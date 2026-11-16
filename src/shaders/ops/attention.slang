module attention;

__exported import core.dtype;

// Fused attention, softmax(q @ k^T * scale) @ v, without materialising the score matrix.
// Each workgroup owns WG_x / 4 query rows of one (batch, head). A row is shared by the
// four lanes of a quad: lane p holds head dims p, p + 4, p + 8, ..., and dot products are
// summed across the quad. K and V stream through shared memory in tiles of BC keys while
// each row keeps a running max, running sum and output accumulator (the online softmax
// from FlashAttention). D is the compiled head size; smaller heads are zero-padded.
//
// Splitting the row keeps the per-thread arrays at D / 4. With one thread per row,
// D = 128 gave wrong results on an RDNA2 GPU for any head size above 80.

[[vk::constant_id(0)]] public const uint WG_x = 128;

static const uint ROW_LANES = 4;

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

float quad_sum(float x)
{
    x += QuadReadAcrossX(x);
    return x + QuadReadAcrossY(x);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void attention_main<T : ITorchVulkan, let D : int, let BC : int>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform AttentionPushConstants pcs)
{
    static const int DL = D / ROW_LANES;   // head dims per lane
    static groupshared float k_tile[BC * D];
    static groupshared float v_tile[BC * D];

    uint lane = li.x;
    uint part = lane % ROW_LANES;
    uint row = wg.x * (WG_x / ROW_LANES) + lane / ROW_LANES;
    bool valid = row < pcs.lq;   // the same for all four lanes of a row
    uint hd = pcs.head_dim;

    T* q = (T*)pcs.q;
    T* k = (T*)pcs.k;
    T* v = (T*)pcs.v;
    T* o = (T*)pcs.out;
    uint q_base = (wg.y * pcs.lq + row) * hd;
    uint kv_base = wg.y * pcs.lk * hd;

    float q_part[DL];
    float acc[DL];
    [ForceUnroll] for (int i = 0; i < DL; i++)
    {
        uint c = i * ROW_LANES + part;
        q_part[i] = (valid && c < hd) ? castValue<T, float>(q[q_base + c]) * pcs.scale : 0.0;
        acc[i] = 0.0;
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
                [ForceUnroll] for (int i = 0; i < DL; i++) dot += q_part[i] * k_tile[j * D + i * ROW_LANES + part];
                dot = quad_sum(dot);
                s[j] = uint(j) < n ? dot : -1.0e30;
                tile_max = max(tile_max, s[j]);
            }

            float correction = exp(m - tile_max);
            l *= correction;
            [ForceUnroll] for (int i = 0; i < DL; i++) acc[i] *= correction;

            [ForceUnroll] for (int j = 0; j < BC; j++)
            {
                float p = uint(j) < n ? exp(s[j] - tile_max) : 0.0;
                l += p;
                [ForceUnroll] for (int i = 0; i < DL; i++) acc[i] += p * v_tile[j * D + i * ROW_LANES + part];
            }
            m = tile_max;
        }
    }

    if (!valid) return;
    float inv_l = 1.0 / l;
    [ForceUnroll] for (int i = 0; i < DL; i++)
    {
        uint c = i * ROW_LANES + part;
        if (c < hd) o[q_base + c] = castValue<float, T>(acc[i] * inv_l);
    }
}
