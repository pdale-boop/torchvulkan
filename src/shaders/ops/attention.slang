module attention;

__exported import core.dtype;

// Fused attention, softmax(q @ k^T * scale) @ v, without materialising the score matrix.
// Each workgroup owns WG_x query rows of one (batch, head), one row per thread. K and V
// stream through shared memory in tiles of BC keys while each thread keeps a running
// max, running sum and output accumulator (the online softmax from FlashAttention).
// D is the compiled head size; smaller heads are zero-padded up to it.

[[vk::constant_id(0)]] public const uint WG_x = 64;

// q, k, v and out are contiguous [batch * heads, length, head_dim]
public struct AttentionPushConstants
{
    public uint64_t q;
    public uint64_t k;
    public uint64_t v;
    public uint64_t out;
    public uint lq;
    public uint lk;
    public uint head_dim;
    public float scale;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void attention_main<T : ITorchVulkan, let D : int, let BC : int>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform AttentionPushConstants pcs)
{
    static groupshared float k_tile[BC * D];
    static groupshared float v_tile[BC * D];

    uint lane = li.x;
    uint row = wg.x * WG_x + lane;
    bool valid = row < pcs.lq;
    uint hd = pcs.head_dim;

    T* q = (T*)pcs.q;
    T* k = (T*)pcs.k;
    T* v = (T*)pcs.v;
    T* o = (T*)pcs.out;
    uint q_base = (wg.y * pcs.lq + row) * hd;
    uint kv_base = wg.y * pcs.lk * hd;

    float q_row[D];
    float acc[D];
    [ForceUnroll] for (int c = 0; c < D; c++)
    {
        q_row[c] = (valid && uint(c) < hd) ? castValue<T, float>(q[q_base + c]) * pcs.scale : 0.0;
        acc[c] = 0.0;
    }

    float m = -1.0e30;
    float l = 0.0;

    for (uint t0 = 0; t0 < pcs.lk; t0 += BC)
    {
        uint n = min(uint(BC), pcs.lk - t0);

        GroupMemoryBarrierWithGroupSync();
        for (uint idx = lane; idx < uint(BC * D); idx += WG_x)
        {
            uint j = idx / D;
            uint c = idx % D;
            bool in_range = j < n && c < hd;
            uint at = kv_base + (t0 + j) * hd + c;
            k_tile[idx] = in_range ? castValue<T, float>(k[at]) : 0.0;
            v_tile[idx] = in_range ? castValue<T, float>(v[at]) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        if (valid)
        {
            float s[BC];
            float tile_max = m;
            [ForceUnroll] for (int j = 0; j < BC; j++)
            {
                float dot = 0.0;
                [ForceUnroll] for (int c = 0; c < D; c++) dot += q_row[c] * k_tile[j * D + c];
                s[j] = uint(j) < n ? dot : -1.0e30;
                tile_max = max(tile_max, s[j]);
            }

            float correction = exp(m - tile_max);
            l *= correction;
            [ForceUnroll] for (int c = 0; c < D; c++) acc[c] *= correction;

            [ForceUnroll] for (int j = 0; j < BC; j++)
            {
                float p = uint(j) < n ? exp(s[j] - tile_max) : 0.0;
                l += p;
                [ForceUnroll] for (int c = 0; c < D; c++) acc[c] += p * v_tile[j * D + c];
            }
            m = tile_max;
        }
    }

    if (!valid) return;
    float inv_l = 1.0 / l;
    [ForceUnroll] for (int c = 0; c < D; c++)
    {
        if (uint(c) < hd) o[q_base + c] = castValue<float, T>(acc[c] * inv_l);
    }
}
