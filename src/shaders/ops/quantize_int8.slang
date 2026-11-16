module quantize_int8;

__exported import core.dtype;

// Row-wise int8 quantization of a contiguous [M, K] float tensor, one workgroup per row, in one pass
// over the row's data in device memory (the row is read twice, the second time mostly from cache):
//   scale[r] = max(float(max |x[r, :]|) / 127, 1e-30)
//   q[r, k]  = int8(clamp(round_half_even(T(x[r, k] / T(scale[r]))), -128, 127))
// This is comfy_kitchen's quantize_int8_rowwise step by step, including the division done in T and
// the smallest normal of T standing in for a scale that rounds to zero in T.

[[vk::constant_id(0)]] public const uint WG_SIZE = 256;
[[vk::constant_id(1)]] public const uint SUBGROUP_SIZE = 32;

public struct QuantizeInt8PushConstants
{
    public uint64_t tensor_in;     // T [M, K], rows `row_stride` apart
    public uint64_t tensor_q;      // int8 [M, K], contiguous
    public uint64_t tensor_scale;  // float [M]
    public uint row_stride;
    public uint k_dim;
    public float scale_min;        // the smallest normal of T
    public uint m_dim;
}

static groupshared float partialMax[WG_SIZE / 4];

// a / b rounded to the nearest float, as on the CPU and CUDA. Vulkan lets a float division be off by
// up to 2.5 ulp, which changes the scales' last bit and, near rounding boundaries, the quantized
// values. One correction step with the exact remainder (fma) fixes that for normal values.
float divRounded(float a, float b)
{
    float q = a / b;
    float r = fma(-q, b, a);
    return fma(r, 1.0 / b, q);
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void quantize_int8_rows_main<T : ITorchVulkan>(
    uint3 wg : SV_GroupID,
    uint ti : SV_GroupIndex,
    uniform QuantizeInt8PushConstants p)
{
    const uint row = wg.x + wg.y * 65535;
    if (row >= p.m_dim) return;  // the same for the whole workgroup
    T* x = (T*)p.tensor_in + row * p.row_stride;
    int8_t* q = (int8_t*)p.tensor_q + row * p.k_dim;

    float m = 0.0;
    for (uint k = ti; k < p.k_dim; k += WG_SIZE) m = max(m, abs(castValue<T, float>(x[k])));
    m = WaveActiveMax(m);
    const uint waves = WG_SIZE / SUBGROUP_SIZE;
    if (ti % SUBGROUP_SIZE == 0) partialMax[ti / SUBGROUP_SIZE] = m;
    GroupMemoryBarrierWithGroupSync();
    m = 0.0;
    for (uint i = 0; i < waves; ++i) m = max(m, partialMax[i]);

    const float scale = max(divRounded(m, 127.0), 1e-30);
    if (ti == 0) ((float*)p.tensor_scale)[row] = scale;
    float s = castValue<T, float>(castValue<float, T>(scale));
    if (s == 0.0) s = p.scale_min;

    for (uint k = ti; k < p.k_dim; k += WG_SIZE)
    {
        float v = castValue<T, float>(castValue<float, T>(divRounded(castValue<T, float>(x[k]), s)));
        q[k] = int8_t(clamp(float.ext_round(v), -128.0, 127.0));
    }
}
