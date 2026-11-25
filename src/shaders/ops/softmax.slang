module softmax;

__exported import core.dtype;
import core.indexing;

// softmax or log_softmax over contiguous rows, one workgroup per row. A thread takes V elements per step (one wide load
// when the rows are aligned); one pass finds the row's max and sum of exponentials (online, rescaling the sum when the
// max grows), a second writes the result, from registers when the row fits in MAX_STEPS steps, else by reading the row
// again. Math in float32 (T.Accum of every dtype this is built for), rounded once to the output type, as the CPU's
// opmath.
// NaN and infinities are told apart by their bits, and the special results are written as bits, so nothing depends on
// float math keeping NaN or inf (Vulkan doesn't promise it without SignedZeroInfNanPreserve). As on the CPU: a NaN or
// +inf makes the row NaN, an all -inf row is NaN, a -inf element gives 0 (log_softmax: -inf).

[[vk::constant_id(0)]] const uint WG_x = 256;
[[vk::constant_id(1)]] const uint LOG = 0;
// half or bfloat16 in, float32 out (half_to_float)
[[vk::constant_id(2)]] const uint OUT_ACCUM = 0;
// also write each row's logsumexp (max + log of the sum), as float32, to lse[row]
[[vk::constant_id(3)]] const uint WRITE_LSE = 0;
// reduce with subgroup arithmetic, else through shared memory
[[vk::constant_id(4)]] const uint SUBGROUP_REDUCE = 0;
// the length is a multiple of V and both tensors are aligned to V elements: one load and one store per step
[[vk::constant_id(5)]] const uint VEC_ALIGNED = 0;
// the row fits in MAX_STEPS steps: keep it in registers instead of reading it twice
[[vk::constant_id(6)]] const uint CACHED = 0;

struct SoftmaxPushConstants
{
    uint64_t input;
    uint64_t output;
    uint64_t lse;
    uint rows;
    uint length;
}

static const uint SOFTMAX_SMEM = 256;
// a compile-time bound: AMD's Windows compiler keeps arrays indexed in spec-constant-bounded loops in scratch memory
static const uint MAX_STEPS = 8;
static const uint NEG_INF = 0xff800000u;
static const uint POS_INF = 0x7f800000u;
static const uint QNAN = 0x7fc00000u;
static const float LOWEST = -3.402823466e38f;

groupshared float smem_max[SOFTMAX_SMEM];
groupshared float smem_sum[SOFTMAX_SMEM];
groupshared uint smem_special[SOFTMAX_SMEM];

bool isFinite(float x) { return (asuint(x) & 0x7fffffffu) < POS_INF; }

// element u of a thread's step: V in a row when aligned, else V loads WG_x apart (each coalesced across the workgroup)
uint stepIndex<let V : int>(uint step, uint lane, int u)
{
    return VEC_ALIGNED == 1 ? (step * WG_x + lane) * V + u : (step * V + u) * WG_x + lane;
}

// AMD's Windows compiler misreads loads through a 16-bit pointer cast to a wider one (#69), so the word pointer comes
// straight from the address. idx must be a multiple of V. Up to 16 bytes, in a uint4 (the compiler drops unused words).
uint4 loadWords<T : ITorchVulkan, let V : int>(uint64_t address, uint idx)
{
    if (V * sizeof(T) == 16) return ((uint4*)address)[idx / V];
    if (V * sizeof(T) == 8) return uint4(((uint2*)address)[idx / V], 0, 0);
    return uint4(((uint*)address)[idx / V], 0, 0, 0);
}

vector<T, V> fromWords<T : ITorchVulkan, let V : int>(uint4 w)
{
    if (V * sizeof(T) == 16) return bit_cast<vector<T, V>>(w);
    if (V * sizeof(T) == 8) return bit_cast<vector<T, V>>(w.xy);
    return bit_cast<vector<T, V>>(w.x);
}

void storeWords<T : ITorchVulkan, let V : int>(uint64_t address, uint idx, vector<T, V> v)
{
    if (V * sizeof(T) == 16) ((uint4*)address)[idx / V] = bit_cast<uint4>(v);
    else if (V * sizeof(T) == 8) ((uint2*)address)[idx / V] = bit_cast<uint2>(v);
    else ((uint*)address)[idx / V] = bit_cast<uint>(v);
}

// A thread's V elements of a step, as stored: one wide load when aligned (kept as words until used: unpacking 16-bit
// values right away makes the compiler wait for each load), else V loads WG_x apart. The loads are clamped into the row
// rather than branched around, so all of a thread's loads can be in flight before it waits.
uint4 loadStepWords<T : ITorchVulkan, let V : int>(uint64_t row_in, uint length, uint step, uint lane)
{
    return loadWords<T, V>(row_in, min(stepIndex<V>(step, lane, 0), length - V));
}

vector<T, V> loadStepElems<T : ITorchVulkan, let V : int>(uint64_t row_in, uint length, uint step, uint lane)
{
    vector<T, V> w;
    [ForceUnroll] for (int u = 0; u < V; ++u) w[u] = ((T*)row_in)[min(stepIndex<V>(step, lane, u), length - 1)];
    return w;
}

vector<T, V> loadStep<T : ITorchVulkan, let V : int>(uint64_t row_in, uint length, uint step, uint lane)
{
    if (VEC_ALIGNED == 1) return fromWords<T, V>(loadStepWords<T, V>(row_in, length, step, lane));
    return loadStepElems<T, V>(row_in, length, step, lane);
}

// the step in float32; past the end of the row the elements read as -inf, which adds nothing and isn't written
vector<float, V> toFloat<T : ITorchVulkan, let V : int>(vector<T, V> w, uint length, uint step, uint lane)
    where T.Accum == float
{
    vector<float, V> v;
    [ForceUnroll] for (int u = 0; u < V; ++u) v[u] = stepIndex<V>(step, lane, u) < length ? w[u].toAccum() : asfloat(NEG_INF);
    return v;
}

// (max, sum of exp(x - max)) with a step folded in. Only finite values count: -inf adds nothing, NaN and +inf are
// flagged in special and make the row NaN. m starts at the lowest float, so m and s stay finite.
void foldStep<let V : int>(inout float m, inout float s, inout uint special, vector<float, V> v)
{
    float step_max = LOWEST;
    [ForceUnroll] for (int u = 0; u < V; ++u)
    {
        if (isFinite(v[u])) step_max = max(step_max, v[u]);
        else if (asuint(v[u]) != NEG_INF) special = 1;
    }
    if (step_max > m)
    {
        s *= exp(m - step_max);
        m = step_max;
    }
    [ForceUnroll] for (int u = 0; u < V; ++u)
    {
        if (isFinite(v[u])) s += exp(v[u] - m);
    }
}

// (m, s) := the combination of (m, s) and (m2, s2); an empty side has s = 0 and contributes nothing
void combine(inout float m, inout float s, float m2, float s2)
{
    float mx = max(m, m2);
    s = s * exp(m - mx) + s2 * exp(m2 - mx);
    m = mx;
}

// every thread gets the workgroup's (m, s, special)
void reduceRow(uint lane, inout float m, inout float s, inout uint special)
{
    if (SUBGROUP_REDUCE == 1)
    {
        float mx = WaveActiveMax(m);
        s = WaveActiveSum(s * exp(m - mx));
        m = mx;
        special = WaveActiveBitOr(special);
        uint subgroup = WaveGetLaneCount();
        if (WG_x <= subgroup) return;
        if (WaveIsFirstLane())
        {
            smem_max[lane / subgroup] = m;
            smem_sum[lane / subgroup] = s;
            smem_special[lane / subgroup] = special;
        }
        GroupMemoryBarrierWithGroupSync();
        m = smem_max[0];
        s = smem_sum[0];
        special = smem_special[0];
        for (uint w = 1; w < WG_x / subgroup; ++w)
        {
            combine(m, s, smem_max[w], smem_sum[w]);
            special |= smem_special[w];
        }
        return;
    }

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
}

void writeStep<T : ITorchVulkan, let V : int>(
    uint64_t row_out, uint length, uint step, uint lane, vector<float, V> v, float m, float s, float log_s, bool nan_row)
    where T.Accum == float
{
    vector<float, V> r;
    [ForceUnroll] for (int u = 0; u < V; ++u)
    {
        float shifted = v[u] - m;
        r[u] = LOG == 1 ? shifted - log_s : exp(shifted) / s;
        if (asuint(v[u]) == NEG_INF) r[u] = LOG == 1 ? asfloat(NEG_INF) : 0.0f;
        if (nan_row) r[u] = asfloat(QNAN);
    }
    if (VEC_ALIGNED == 1)
    {
        uint i = stepIndex<V>(step, lane, 0);
        if (i >= length) return;
        if (OUT_ACCUM == 1)
        {
            storeWords<float, V>(row_out, i, r);
            return;
        }
        vector<T, V> t;
        [ForceUnroll] for (int u = 0; u < V; ++u) t[u] = T.fromAccum(r[u]);
        storeWords<T, V>(row_out, i, t);
        return;
    }
    [ForceUnroll] for (int u = 0; u < V; ++u)
    {
        uint i = stepIndex<V>(step, lane, u);
        if (i < length)
        {
            if (OUT_ACCUM == 1) ((float*)row_out)[i] = r[u];
            else ((T*)row_out)[i] = T.fromAccum(r[u]);
        }
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void softmax_main<T : ITorchVulkan, let V : int>(
    uint3 group : SV_GroupID,
    uint3 li : SV_GroupThreadID,
    uniform SoftmaxPushConstants pcs) where T.Accum == float
{
    uint row = flat_group(group);
    if (row >= pcs.rows) return;
    uint lane = li.x;
    uint64_t row_in = pcs.input + uint64_t(row) * pcs.length * uint64_t(sizeof(T));
    uint64_t row_out = pcs.output + uint64_t(row) * pcs.length * uint64_t(OUT_ACCUM == 1 ? 4 : sizeof(T));
    uint steps = (pcs.length + WG_x * V - 1) / (WG_x * V);

    float m = LOWEST;
    float s = 0.0f;
    uint special = 0;
    // CACHED: the row is read once, every load issued before the first is used
    uint4 words[MAX_STEPS];
    vector<T, V> elems[MAX_STEPS];
    if (CACHED == 1)
    {
        [ForceUnroll] for (uint st = 0; st < MAX_STEPS; ++st)
        {
            if (st >= steps) break;
            if (VEC_ALIGNED == 1) words[st] = loadStepWords<T, V>(row_in, pcs.length, st, lane);
            else elems[st] = loadStepElems<T, V>(row_in, pcs.length, st, lane);
        }
        [ForceUnroll] for (uint st = 0; st < MAX_STEPS; ++st)
        {
            if (st >= steps) break;
            if (VEC_ALIGNED == 1) elems[st] = fromWords<T, V>(words[st]);
            foldStep<V>(m, s, special, toFloat<T, V>(elems[st], pcs.length, st, lane));
        }
    }
    else
    {
        for (uint st = 0; st < steps; ++st)
            foldStep<V>(m, s, special, toFloat<T, V>(loadStep<T, V>(row_in, pcs.length, st, lane), pcs.length, st, lane));
    }

    reduceRow(lane, m, s, special);
    // s >= 1 once the row has a finite value (its max adds exp(0)), so s == 0 means every value is -inf
    bool nan_row = special != 0 || s == 0.0f;
    float log_s = log(s);
    if (WRITE_LSE == 1 && lane == 0)
        ((float*)pcs.lse)[row] = special != 0 ? asfloat(QNAN) : s == 0.0f ? asfloat(NEG_INF) : m + log_s;

    if (CACHED == 1)
    {
        [ForceUnroll] for (uint st = 0; st < MAX_STEPS; ++st)
        {
            if (st >= steps) break;
            writeStep<T, V>(row_out, pcs.length, st, lane, toFloat<T, V>(elems[st], pcs.length, st, lane), m, s, log_s, nan_row);
        }
    }
    else
    {
        for (uint st = 0; st < steps; ++st)
        {
            vector<float, V> v = toFloat<T, V>(loadStep<T, V>(row_in, pcs.length, st, lane), pcs.length, st, lane);
            writeStep<T, V>(row_out, pcs.length, st, lane, v, m, s, log_s, nan_row);
        }
    }
}
