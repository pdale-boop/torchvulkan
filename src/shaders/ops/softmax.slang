module softmax;

__exported import core.dtype;
import core.indexing;

// softmax or log_softmax over contiguous rows, one workgroup per row: one pass for the row's max and sum of exponentials
// (online, rescaling the sum when the max grows), one to write the result. Math in T.Accum, rounded once to the output
// type, as the CPU's opmath.

[[vk::constant_id(0)]] const uint WG_x = 256;
[[vk::constant_id(1)]] const uint LOG = 0;
// half or bfloat16 in, float32 out (half_to_float)
[[vk::constant_id(2)]] const uint OUT_ACCUM = 0;
// also write each row's logsumexp (max + log of the sum), as float32, to lse[row]
[[vk::constant_id(3)]] const uint WRITE_LSE = 0;

struct SoftmaxPushConstants
{
    uint64_t input;
    uint64_t output;
    uint64_t lse;
    uint rows;
    uint length;
}

static const int SOFTMAX_SMEM = 256;

// (max, sum of exp(x - max)) with x folded in; -inf adds nothing (and would make exp(-inf - -inf) NaN)
void fold<A : ITorchVulkan>(inout A m, inout A s, A x)
{
    // a NaN makes the row's max NaN, as the CPU's max does, so the whole row comes out NaN
    if (!A.ext_eq(m, m)) return;
    if (!A.ext_eq(x, x))
    {
        m = x;
        return;
    }
    if (A.ext_gt(x, m))
    {
        // exp(x - x) is 1, or NaN for x = +inf, as the CPU's exp(x - max) is
        s = A.ext_add(A.ext_mul(s, A.ext_exp(A.ext_sub(m, x))), A.ext_exp(A.ext_sub(x, x)));
        m = x;
    }
    else if (!A.ext_eq(x, A.ext_neg(positiveInf<A>())))
    {
        s = A.ext_add(s, A.ext_exp(A.ext_sub(x, m)));
    }
}

A positiveInf<A : ITorchVulkan>() { return A.castFromF32(asfloat(0x7f800000u)); }

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void softmax_main<T : ITorchVulkan>(
    uint3 group : SV_GroupID,
    uint3 li : SV_GroupThreadID,
    uniform SoftmaxPushConstants pcs)
{
    typealias A = T.Accum;
    static groupshared A smem_max[SOFTMAX_SMEM];
    static groupshared A smem_sum[SOFTMAX_SMEM];

    uint row = flat_group(group);
    if (row >= pcs.rows) return;
    uint lane = li.x;
    T* x = (T*)pcs.input + uint64_t(row) * pcs.length;

    A m = A.ext_neg(positiveInf<A>());
    A s = A.ext_zero();
    for (uint i = lane; i < pcs.length; i += WG_x) fold<A>(m, s, x[i].toAccum());

    smem_max[lane] = m;
    smem_sum[lane] = s;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG_x / 2; stride > 0; stride /= 2)
    {
        if (lane < stride)
        {
            A m1 = smem_max[lane], m2 = smem_max[lane + stride];
            A s1 = smem_sum[lane], s2 = smem_sum[lane + stride];
            A mx = !A.ext_eq(m1, m1) ? m1 : !A.ext_eq(m2, m2) ? m2 : A.ext_max(m1, m2);
            // a side with nothing in it (max -inf) contributes nothing
            A neg_inf = A.ext_neg(positiveInf<A>());
            A t1 = A.ext_eq(m1, neg_inf) ? A.ext_zero() : A.ext_mul(s1, A.ext_exp(A.ext_sub(m1, mx)));
            A t2 = A.ext_eq(m2, neg_inf) ? A.ext_zero() : A.ext_mul(s2, A.ext_exp(A.ext_sub(m2, mx)));
            smem_max[lane] = mx;
            smem_sum[lane] = A.ext_add(t1, t2);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    m = smem_max[0];
    s = smem_sum[0];
    A log_s = A.ext_log(s);
    // T.Accum is float32 for every dtype this kernel is built for
    if (WRITE_LSE == 1 && lane == 0) ((A*)pcs.lse)[row] = A.ext_add(m, log_s);

    for (uint i = lane; i < pcs.length; i += WG_x)
    {
        A shifted = A.ext_sub(x[i].toAccum(), m);
        A v = LOG == 1 ? A.ext_sub(shifted, log_s) : A.ext_div(A.ext_exp(shifted), s);
        uint64_t o = uint64_t(row) * pcs.length + i;
        if (OUT_ACCUM == 1) ((A*)pcs.output)[o] = v;
        else ((T*)pcs.output)[o] = T.fromAccum(v);
    }
}
