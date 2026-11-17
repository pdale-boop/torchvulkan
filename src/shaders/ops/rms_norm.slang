module rms_norm;

__exported import core.dtype;

/*
RMS normalisation over the last N elements of each row, in one pass over the row's data, optionally
followed by a rotary position embedding (RoPE) of the normalised row:

    y = T(x * rsqrt(mean(x^2) + eps) * weight)        (computed in float32, as PyTorch's rms_norm)
    out = rope(y)                                      (when ROPE_MODE != 0)

RoPE rotates pairs of elements by a 2x2 matrix per pair from freqs_cis (float32 [..., N/2, 2, 2]):

    (o0, o1) = (f00 * y0 + f01 * y1,  f10 * y0 + f11 * y1)

with pairs (2i, 2i+1) for ROPE_MODE 1 (interleaved, as Flux and Z-Image) and (i, i + N/2) for
ROPE_MODE 2 (split halves). y is rounded to T before the rotation and the result is rounded again,
as comfy_kitchen's rms_norm followed by its rope does.

Rows are handled by one subgroup each (ROW_PER_SUBGROUP, for short rows such as attention heads) or
by a whole workgroup each (long rows such as hidden states).
*/

[[vk::constant_id(0)]] public const uint WG_SIZE = 256;
[[vk::constant_id(1)]] public const uint SUBGROUP_SIZE = 32;
[[vk::constant_id(2)]] public const uint ROW_PER_SUBGROUP = 0;
[[vk::constant_id(3)]] public const uint ROPE_MODE = 0;   // 0 none, 1 interleaved pairs, 2 split halves
[[vk::constant_id(4)]] public const uint HAS_WEIGHT = 1;

public struct RmsNormPushConstants
{
    public uint64_t x;          // T [rows, n], contiguous
    public uint64_t weight;     // float [n]
    public uint64_t out;        // T [rows, n], contiguous
    public uint64_t freqs;      // float [freq_rows, n / 2, 2, 2]
    public uint rows;
    public uint n;
    public float eps;
    // rope: x is [d0, dim1, dim2, n], so row r is (r / (dim1 * dim2), (r / dim2) % dim1, r % dim2), and
    // its 2x2 matrices start at freqs + (i0 * fstride0 + i1 * fstride1 + i2 * fstride2) * n / 2 * 4
    // (a stride is 0 where freqs_cis broadcasts)
    public uint dim1;
    public uint dim2;
    public uint fstride0;
    public uint fstride1;
    public uint fstride2;
}

static groupshared float partialSum[WG_SIZE / 4];

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void rms_norm_main<T : ITorchVulkan>(
    uint3 wg : SV_GroupID,
    uint ti : SV_GroupIndex,
    uniform RmsNormPushConstants p)
{
    const uint group_size = ROW_PER_SUBGROUP == 1 ? SUBGROUP_SIZE : WG_SIZE;
    const uint rows_per_wg = WG_SIZE / group_size;
    const uint wg_index = wg.x + wg.y * 65535;
    const uint row = wg_index * rows_per_wg + ti / group_size;
    const uint lane = ti % group_size;
    // in subgroup mode a subgroup past the last row has nothing to do; in workgroup mode the whole
    // workgroup is past it together
    if (row >= p.rows) return;

    T* x = (T*)p.x + row * p.n;
    T* o = (T*)p.out + row * p.n;
    float* w = (float*)p.weight;

    float ss = 0.0;
    for (uint i = lane; i < p.n; i += group_size)
    {
        float v = castValue<T, float>(x[i]);
        ss += v * v;
    }
    ss = WaveActiveSum(ss);
    if (ROW_PER_SUBGROUP == 0)
    {
        if (ti % SUBGROUP_SIZE == 0) partialSum[ti / SUBGROUP_SIZE] = ss;
        GroupMemoryBarrierWithGroupSync();
        ss = 0.0;
        for (uint s = 0; s < WG_SIZE / SUBGROUP_SIZE; ++s) ss += partialSum[s];
    }
    const float rstd = rsqrt(ss / float(p.n) + p.eps);

    if (ROPE_MODE == 0)
    {
        for (uint i = lane; i < p.n; i += group_size)
        {
            float v = castValue<T, float>(x[i]) * rstd;
            if (HAS_WEIGHT == 1) v = v * w[i];
            o[i] = castValue<float, T>(v);
        }
        return;
    }

    const uint half_n = p.n / 2;
    const uint i2 = row % p.dim2;
    const uint i1 = (row / p.dim2) % p.dim1;
    const uint i0 = row / (p.dim1 * p.dim2);
    float* f = (float*)p.freqs + (i0 * p.fstride0 + i1 * p.fstride1 + i2 * p.fstride2) * half_n * 4;
    for (uint i = lane; i < half_n; i += group_size)
    {
        const uint a = ROPE_MODE == 1 ? 2 * i : i;
        const uint b = ROPE_MODE == 1 ? 2 * i + 1 : i + half_n;
        float y0 = castValue<T, float>(x[a]) * rstd;
        float y1 = castValue<T, float>(x[b]) * rstd;
        if (HAS_WEIGHT == 1) { y0 = y0 * w[a]; y1 = y1 * w[b]; }
        // the normalised values are rounded to T, as rms_norm's output is
        y0 = castValue<T, float>(castValue<float, T>(y0));
        y1 = castValue<T, float>(castValue<float, T>(y1));
        float* m = f + i * 4;
        o[a] = castValue<float, T>(m[0] * y0 + m[1] * y1);
        o[b] = castValue<float, T>(m[2] * y0 + m[3] * y1);
    }
}
