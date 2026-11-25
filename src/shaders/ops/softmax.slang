module softmax;

__exported import core.dtype;
import core.indexing;

// softmax or log_softmax over contiguous rows, one workgroup per row: one pass for the row's max and sum of exponentials
// (online, rescaling the sum when the max grows), one to write the result. Math in float32 (T.Accum of every dtype this
// is built for), rounded once to the output type, as the CPU's opmath.
// NaN and infinities are told apart by their bits, and the special results are written as bits, so nothing depends on
// float math keeping NaN or inf (Vulkan doesn't promise it without SignedZeroInfNanPreserve). As on the CPU: a NaN or
// +inf makes the row NaN, an all -inf row is NaN, a -inf element gives 0 (log_softmax: -inf).

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
static const uint NEG_INF = 0xff800000u;
static const uint POS_INF = 0x7f800000u;
static const uint QNAN = 0x7fc00000u;
static const float LOWEST = -3.402823466e38f;

bool isFinite(float x) { return (asuint(x) & 0x7fffffffu) < POS_INF; }

// (max, sum of exp(x - max)) with x folded in. Only finite values count: -inf adds nothing, NaN and +inf are flagged
// in special and make the row NaN. m starts at the lowest float, so m and s stay finite.
void fold(inout float m, inout float s, inout uint special, float x)
{
    if (!isFinite(x))
    {
        if (asuint(x) != NEG_INF) special = 1;
        return;
    }
    if (x > m)
    {
        s = s * exp(m - x) + 1.0f;
        m = x;
    }
    else s += exp(x - m);
}

// (m, s) := the combination of (m, s) and (m2, s2); an empty side has s = 0 and contributes nothing
void combine(inout float m, inout float s, float m2, float s2)
{
    float mx = max(m, m2);
    s = s * exp(m - mx) + s2 * exp(m2 - mx);
    m = mx;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void softmax_main<T : ITorchVulkan>(
    uint3 group : SV_GroupID,
    uint3 li : SV_GroupThreadID,
    uniform SoftmaxPushConstants pcs) where T.Accum == float
{
    static groupshared float smem_max[SOFTMAX_SMEM];
    static groupshared float smem_sum[SOFTMAX_SMEM];
    static groupshared uint smem_special[SOFTMAX_SMEM];

    uint row = flat_group(group);
    if (row >= pcs.rows) return;
    uint lane = li.x;
    T* x = (T*)pcs.input + uint64_t(row) * pcs.length;

    float m = LOWEST;
    float s = 0.0f;
    uint special = 0;
    for (uint i = lane; i < pcs.length; i += WG_x) fold(m, s, special, x[i].toAccum());

    smem_max[lane] = m;
    smem_sum[lane] = s;
    smem_special[lane] = special;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG_x / 2; stride > 0; stride /= 2)
    {
        if (lane < stride)
        {
            combine(m, s, smem_max[lane + stride], smem_sum[lane + stride]);
            special |= smem_special[lane + stride];
            smem_max[lane] = m;
            smem_sum[lane] = s;
            smem_special[lane] = special;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    m = smem_max[0];
    s = smem_sum[0];
    special = smem_special[0];
    // s >= 1 once the row has a finite value (its max adds exp(0)), so s == 0 means every value is -inf
    bool nan_row = special != 0 || s == 0.0f;
    float log_s = log(s);
    if (WRITE_LSE == 1 && lane == 0)
        ((float*)pcs.lse)[row] = special != 0 ? asfloat(QNAN) : s == 0.0f ? asfloat(NEG_INF) : m + log_s;

    for (uint i = lane; i < pcs.length; i += WG_x)
    {
        float xi = x[i].toAccum();
        float shifted = xi - m;
        float v = LOG == 1 ? shifted - log_s : exp(shifted) / s;
        if (asuint(xi) == NEG_INF) v = LOG == 1 ? asfloat(NEG_INF) : 0.0f;
        if (nan_row) v = asfloat(QNAN);
        uint64_t o = uint64_t(row) * pcs.length + i;
        if (OUT_ACCUM == 1) ((float*)pcs.output)[o] = v;
        else ((T*)pcs.output)[o] = T.fromAccum(v);
    }
}
