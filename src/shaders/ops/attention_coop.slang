module attention_coop;

__exported import core.dtype;

// Fused attention on cooperative matrices, softmax(q @ k^T * scale) @ v, for float16 and bfloat16
// with head sizes that are multiples of 16 (FlashAttention-2 style).
//
// A workgroup has SUBGROUPS subgroups and owns ROWS query rows of one (batch, head); each subgroup
// owns 16 of them, keeps their Q fragments and its O accumulators in registers. Per tile of BC keys:
//   1. K is staged in shared memory and S = Q K^T is computed on cooperative matrices (fp32
//      accumulators) and stored to shared memory;
//   2. the subgroup's lanes, SUBGROUP_SIZE / 16 per row, update each row's running max and sum
//      (combined with subgroup shuffles), write P = exp(S - max) in T and the row's correction
//      exp(old max - new max), broadcast along a 16-wide row;
//   3. V replaces K in the same shared buffer, and O = O * correction + P V, the correction applied
//      as a component-wise product of cooperative matrices.
// Q and O are padded by the host to a multiple of ROWS rows; keys past the end are masked out.

[[vk::constant_id(0)]] public const uint WG_SIZE = 128;       // SUBGROUPS * SUBGROUP_SIZE
[[vk::constant_id(1)]] public const uint SUBGROUP_SIZE = 32;
// 8 where shared memory allows (each K/V tile then serves twice as many query rows), else 4
[[vk::constant_id(2)]] public const uint SUBGROUPS = 4;

static const uint F = 16;                  // fragment size (16 x 16 x 16)
static const uint BC = 32;                 // keys per tile
static const uint ROWS = SUBGROUPS * F;    // query rows per workgroup

// q and out are [batch * heads, lq_padded, D]; k and v are [batch * heads, lk, D]; all contiguous
public struct AttentionCoopPushConstants
{
    public uint64_t q;
    public uint64_t k;
    public uint64_t v;
    public uint64_t out;
    public uint lq_padded;
    public uint lk;
    public float scale;
    public uint row_offset;  // first query row of this dispatch (large problems are split, see attention.cpp)
}

typealias Acc = linalg::CoopMat<float, MemoryScope.Subgroup, F, F, linalg::CoopMatMatrixUse.MatrixAccumulator>;
typealias FragA<T : ICoopElement> = linalg::CoopMat<T, MemoryScope.Subgroup, F, F, linalg::CoopMatMatrixUse.MatrixA>;
typealias FragB<T : ICoopElement> = linalg::CoopMat<T, MemoryScope.Subgroup, F, F, linalg::CoopMatMatrixUse.MatrixB>;
typealias OutFrag<T : ICoopElement> = linalg::CoopMat<T, MemoryScope.Subgroup, F, F, linalg::CoopMatMatrixUse.MatrixAccumulator>;

// combine a value across the lanes that share a row (adjacent, aligned groups of `lanes`)
float row_max(float x, uint lanes)
{
    for (uint m = 1; m < lanes; m *= 2) x = max(x, WaveReadLaneAt(x, WaveGetLaneIndex() ^ m));
    return x;
}

float row_sum(float x, uint lanes)
{
    for (uint m = 1; m < lanes; m *= 2) x += WaveReadLaneAt(x, WaveGetLaneIndex() ^ m);
    return x;
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void attention_coop_main<T : ITorchVulkan & ICoopElement, let D : int>(
    uint3 wg : SV_GroupID,
    uint ti : SV_GroupIndex,
    uniform AttentionCoopPushConstants pcs)
{
    static const int DF = D / F;                       // fragments along the head dimension
    static groupshared T k_tile[BC * D];               // [key][dim]
    static groupshared T v_tile[BC * D];               // [key][dim]
    static groupshared T p_tile[ROWS * BC];            // [row][key], 16 rows per subgroup
    static groupshared float s_tile[ROWS * BC];        // [row][key]
    static groupshared float c_tile[ROWS * F];         // [row][16 copies of the row's factor]

    uint sg = ti / SUBGROUP_SIZE;
    uint lane = ti % SUBGROUP_SIZE;
    uint lanes = SUBGROUP_SIZE / F;                    // lanes per row in the softmax
    uint row = lane / lanes;                           // this lane's row within the subgroup
    uint part = lane % lanes;
    uint bh = wg.y;
    uint row0 = pcs.row_offset + wg.x * ROWS + sg * F; // this subgroup's first query row

    T* q = (T*)pcs.q;
    T* k = (T*)pcs.k;
    T* v = (T*)pcs.v;
    T* o_ptr = (T*)pcs.out;
    uint q_base = (bh * pcs.lq_padded + row0) * D;
    uint kv_base = bh * pcs.lk * D;
    uint s_base = sg * F * BC;
    uint s_row = s_base + row * BC;
    uint c_row = sg * F * F + row * F;

    FragA<T> qf[DF];
    Acc o[DF];
    [ForceUnroll] for (int d = 0; d < DF; d++)
    {
        qf[d] = FragA<T>();
        qf[d] = qf[d].Load<linalg::CoopMatMatrixLayout.RowMajor>(q, q_base + d * F, D);
        o[d] = Acc(0.0);
    }

    // running max and sum of this lane's row, the same in every lane of the row
    float m = -1.0e30;
    float l = 0.0;

    for (uint t0 = 0; t0 < pcs.lk; t0 += BC)
    {
        uint n = min(BC, pcs.lk - t0);

        GroupMemoryBarrierWithGroupSync();
        for (uint i = ti; i < BC * D; i += WG_SIZE)
        {
            uint j = i / D;
            uint at = kv_base + (t0 + j) * D + (i % D);
            k_tile[i] = j < n ? k[at] : T.ext_zero();
            v_tile[i] = j < n ? v[at] : T.ext_zero();
        }
        GroupMemoryBarrierWithGroupSync();

        // 1. S = Q K^T: K is [key][dim] row-major, so K^T is a column-major fragment with stride D
        [ForceUnroll] for (int j = 0; j < BC / F; j++)
        {
            Acc s = Acc(0.0);
            [ForceUnroll] for (int d = 0; d < DF; d++)
            {
                FragB<T> kf = FragB<T>();
                kf = kf.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(k_tile, j * F * D + d * F, D);
                s = linalg::coopMatMulAdd<float, false>(qf[d], kf, s);
            }
            s.Store<linalg::CoopMatMatrixLayout.RowMajor>(s_tile, s_base + j * F, BC);
        }
        GroupMemoryBarrierWithGroupSync();

        // 2. online softmax, `lanes` lanes per row
        float tile_max = -1.0e30;
        for (uint c = part; c < n; c += lanes) tile_max = max(tile_max, s_tile[s_row + c] * pcs.scale);
        float new_max = max(m, row_max(tile_max, lanes));
        float correction = exp(m - new_max);
        float sum = 0.0;
        for (uint c = part; c < BC; c += lanes)
        {
            float p = c < n ? exp(s_tile[s_row + c] * pcs.scale - new_max) : 0.0;
            sum += p;
            p_tile[s_row + c] = castValue<float, T>(p);
        }
        l = l * correction + row_sum(sum, lanes);
        m = new_max;
        for (uint c = part; c < F; c += lanes) c_tile[c_row + c] = correction;

        // 3. O = O * correction + P V
        GroupMemoryBarrierWithGroupSync();

        Acc corr = Acc();
        corr = corr.Load<linalg::CoopMatMatrixLayout.RowMajor>(c_tile, sg * F * F, F);
        FragA<T> pf[BC / F];
        [ForceUnroll] for (int j = 0; j < BC / F; j++)
        {
            pf[j] = FragA<T>();
            pf[j] = pf[j].Load<linalg::CoopMatMatrixLayout.RowMajor>(p_tile, s_base + j * F, BC);
        }
        [ForceUnroll] for (int d = 0; d < DF; d++)
        {
            Acc acc = o[d] * corr;
            [ForceUnroll] for (int j = 0; j < BC / F; j++)
            {
                FragB<T> vf = FragB<T>();
                vf = vf.Load<linalg::CoopMatMatrixLayout.RowMajor>(v_tile, j * F * D + d * F, D);
                acc = linalg::coopMatMulAdd<float, false>(pf[j], vf, acc);
            }
            o[d] = acc;
        }
    }

    // O / l, with 1 / l broadcast along each row like the correction
    GroupMemoryBarrierWithGroupSync();
    for (uint c = part; c < F; c += lanes) c_tile[c_row + c] = 1.0 / l;
    GroupMemoryBarrierWithGroupSync();
    Acc inv = Acc();
    inv = inv.Load<linalg::CoopMatMatrixLayout.RowMajor>(c_tile, sg * F * F, F);
    [ForceUnroll] for (int d = 0; d < DF; d++)
    {
        OutFrag<T> result = OutFrag<T>(o[d] * inv);
        result.Store<linalg::CoopMatMatrixLayout.RowMajor>(o_ptr, q_base + d * F, D);
    }
}
