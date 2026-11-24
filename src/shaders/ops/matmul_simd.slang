module matmul_simd;

import helpers.gemm_simd;

struct MatmulPushConstants<T : ITorchVulkan>
{
    uint64_t tensor_a;
    uint64_t tensor_b;
    uint64_t tensor_out;
    uint64_t tensor_bias;
    TensorStrides strides_A;
    TensorStrides strides_B;
    TensorStrides strides_C;
    TensorStrides strides_bias;
    uint M;
    uint N;
    uint K;
    uint padding;
    T.PushScalar alpha;
    T.PushScalar beta;
}

[shader("compute")]
[numthreads(WG_x, WG_y, 1)]
void matmul_simd_main<T : ITorchVulkan, let VEC : int>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform MatmulPushConstants<T.Accum> pcs)
{
    typealias Acc = T.Accum;
    typealias Epilogue = LinearEpilogue<Acc, Activation<Acc>, AccumMatrix<T>>;

    static groupshared Acc smemA[TILE_M * (TILE_K + 1)];
    static groupshared vector<Acc, VEC> smemB[TILE_K * (TILE_N / VEC)];

    AccumMatrix<T> a = AccumMatrix<T>(pcs.tensor_a, pcs.strides_A);
    AccumMatrix<T> b = AccumMatrix<T>(pcs.tensor_b, pcs.strides_B);
    AccumMatrix<T> bias = AccumMatrix<T>(pcs.tensor_bias, pcs.strides_bias);
    AccumMatrix<T> out = AccumMatrix<T>(pcs.tensor_out, pcs.strides_C);
    Epilogue epilogue = Epilogue(Acc.fromPushScalar(pcs.alpha), Acc.fromPushScalar(pcs.beta), bias);

    matmulTile<Acc, VEC, AccumMatrix<T>, AccumMatrix<T>, Epilogue, AccumMatrix<T>>(
        li, wg, wg.z, smemA, smemB, a, b, epilogue, out, pcs.M, pcs.N, pcs.K);
}

// float16 matmul on packed dot products: each OpFDot2MixAcc32VALVE multiplies a pair of halves along K and adds both
// products to a float32 accumulator (v_dot2_f32_f16 on AMD), so it does twice the FMAs of matmul_simd_main per instruction
// at the same accumulate precision. Only dispatched where VK_VALVE_shader_mixed_float_dot_product's
// shaderMixedFloatDotProductFloat16AccFloat32 is enabled.
//
// The workgroup is 8 x 16 threads (WG_x along N, WG_y along M) and each thread computes 16 columns by TILE_M / 16 rows.
// Shared memory holds two stages of the tile as K pairs, pair-major: sA[kp][m] and sB[kp][n] are uints of two halves, so
// a thread reads its rows and columns of one pair as uint4s. The next tile's global loads go to registers while this one
// computes, then into the other stage. IS_ALIGNED is ignored (the host passes 0 so the epilogue honours every stride);
// VEC_LOADS says rows are 16-byte aligned along K (and, for an N-major B, 8-byte aligned along N), so tiles inside the
// matrix load 8 halves at a time.

[[vk::constant_id(6)]] const uint B_KMAJOR = 0;
[[vk::constant_id(7)]] const uint VEC_LOADS = 0;

static const int D2_TILE_N = 128;
static const int D2_TILE_K = 16;
static const int D2_KP = D2_TILE_K / 2;
static const int D2_TN = 16;
// pads each pair row by one uint4 so rows stay 16-byte aligned and the transposing stores spread over the banks
static const int D2_PAD = 4;
static const int D2_ROW_A = TILE_M + D2_PAD;
static const int D2_ROW_B = D2_TILE_N + D2_PAD;
static const int D2_STAGE_A = D2_KP * D2_ROW_A / 4;
static const int D2_STAGE_B = D2_KP * D2_ROW_B / 4;

static groupshared uint4 d2A[2 * D2_STAGE_A];
static groupshared uint4 d2B[2 * D2_STAGE_B];

float dot2Acc(uint a, uint b, float c)
{
    half2 ha = bit_cast<half2>(a);
    half2 hb = bit_cast<half2>(b);
    return spirv_asm {
        OpExtension "SPV_VALVE_mixed_float_dot_product";
        OpCapability DotProductFloat16AccFloat32VALVE;
        result:$$float = OpFDot2MixAcc32VALVE $ha $hb $c;
    };
}

uint halfBits(uint64_t address, uint offset)
{
    return uint(((uint16_t*)address)[offset]);
}

uint4 load16(uint64_t address, uint offset)
{
    return *((uint4*)(address + uint64_t(offset) * 2));
}

uint2 load8(uint64_t address, uint offset)
{
    return *((uint2*)(address + uint64_t(offset) * 2));
}

uint4 packPairs(uint2 r0, uint2 r1)
{
    return uint4((r0.x & 0xffffu) | (r1.x << 16), (r0.x >> 16) | (r1.x & 0xffff0000u),
                 (r0.y & 0xffffu) | (r1.y << 16), (r0.y >> 16) | (r1.y & 0xffff0000u));
}

void storePair(uint stage, bool is_a, uint kp, uint i, uint value)
{
    uint word = is_a ? stage * D2_STAGE_A * 4 + kp * D2_ROW_A + i : stage * D2_STAGE_B * 4 + kp * D2_ROW_B + i;
    if (is_a) d2A[word / 4][word % 4] = value;
    else d2B[word / 4][word % 4] = value;
}

// eight halves along K from element offset base + k * k_stride, as four K pairs; zero past the matrix
uint4 loadKRun(uint64_t address, uint base, uint k_stride, uint k, uint K, bool inside, bool vec)
{
    if (vec) return load16(address, base + k);
    uint4 pairs = uint4(0);
    [unroll] for (uint q = 0; q < 4; q++)
    {
        uint lo = (inside && k + 2 * q < K) ? halfBits(address, base + (k + 2 * q) * k_stride) : 0;
        uint hi = (inside && k + 2 * q + 1 < K) ? halfBits(address, base + (k + 2 * q + 1) * k_stride) : 0;
        pairs[q] = lo | (hi << 16);
    }
    return pairs;
}

// one thread's share of the tile loads, kept in registers between the global read and the shared-memory write.
// A (and a K-major B): item i < 2 * rows covers row i / 2, halves 8 * (i % 2) .. +8 of the tile's K, so two lanes
// read 32 contiguous bytes. An N-major B: item i covers pair row i / 32 and columns 4 * (i % 32) .. +4.
struct Dot2Loader
{
    uint64_t a_address, b_address;
    TensorStrides sa, sb;
    uint batch, m0, n0, M, N, K, t;
    uint4 ra[2], rb[2];
    // the interior fast path: byte addresses of this thread's loads at k = 0, and how far they move per tile
    uint64_t pa[2], pb[2];
    uint64_t b_step;

    // items past 2 * TILE_M are idle (a 64-row tile has half as many A rows to load)
    bool active(uint p) { return p == 0 || 2 * TILE_M > WG_x * WG_y; }

    [mutating] void setupFast()
    {
        [unroll] for (uint p = 0; p < 2; p++)
        {
            uint item = t + p * (WG_x * WG_y);
            pa[p] = a_address + uint64_t(batch * sa.batch + (m0 + item / 2) * sa.row + 8 * (item % 2)) * 2;
            if (B_KMAJOR == 1) pb[p] = b_address + uint64_t(batch * sb.batch + (n0 + item / 2) * sb.col + 8 * (item % 2)) * 2;
            else pb[p] = b_address + uint64_t(batch * sb.batch + 2 * (item / 32) * sb.row + n0 + 4 * (item % 32)) * 2;
        }
        b_step = B_KMAJOR == 1 ? uint64_t(D2_TILE_K) * 2 : uint64_t(D2_TILE_K) * sb.row * 2;
    }

    // a tile wholly inside the matrix, with VEC_LOADS: no bounds checks, one 64-bit add per load
    [mutating] void fetchFast(uint tile)
    {
        [unroll] for (uint p = 0; p < 2; p++)
        {
            if (active(p)) ra[p] = *((uint4*)(pa[p] + uint64_t(tile) * D2_TILE_K * 2));
            uint64_t b = pb[p] + uint64_t(tile) * b_step;
            if (B_KMAJOR == 1) rb[p] = *((uint4*)b);
            else rb[p] = packPairs(*((uint2*)b), *((uint2*)(b + uint64_t(sb.row) * 2)));
        }
    }

    [mutating] void fetch(uint k0)
    {
        bool k_full = k0 + D2_TILE_K <= K;
        [unroll] for (uint p = 0; p < 2; p++)
        {
            uint item = t + p * (WG_x * WG_y);
            if (active(p))
            {
                uint m = m0 + item / 2;
                bool vec = VEC_LOADS == 1 && k_full && m0 + TILE_M <= M;
                ra[p] = loadKRun(a_address, batch * sa.batch + m * sa.row, sa.col, k0 + 8 * (item % 2), K, m < M, vec);
            }
            bool vec_b = VEC_LOADS == 1 && k_full && n0 + D2_TILE_N <= N;
            if (B_KMAJOR == 1)
            {
                uint n = n0 + item / 2;
                rb[p] = loadKRun(b_address, batch * sb.batch + n * sb.col, sb.row, k0 + 8 * (item % 2), K, n < N, vec_b);
            }
            else
            {
                uint k = k0 + 2 * (item / 32);
                uint n = n0 + 4 * (item % 32);
                uint base = batch * sb.batch + n * sb.col;
                if (vec_b)
                {
                    rb[p] = packPairs(load8(b_address, base + k * sb.row), load8(b_address, base + (k + 1) * sb.row));
                }
                else
                {
                    [unroll] for (uint v = 0; v < 4; v++)
                    {
                        bool col_in = n + v < N;
                        uint lo = (col_in && k < K) ? halfBits(b_address, base + v * sb.col + k * sb.row) : 0;
                        uint hi = (col_in && k + 1 < K) ? halfBits(b_address, base + v * sb.col + (k + 1) * sb.row) : 0;
                        rb[p][v] = lo | (hi << 16);
                    }
                }
            }
        }
    }

    void stash(uint stage)
    {
        [unroll] for (uint p = 0; p < 2; p++)
        {
            uint item = t + p * (WG_x * WG_y);
            if (active(p))
            {
                [unroll] for (uint q = 0; q < 4; q++) storePair(stage, true, 4 * (item % 2) + q, item / 2, ra[p][q]);
            }
            if (B_KMAJOR == 1)
            {
                [unroll] for (uint q = 0; q < 4; q++) storePair(stage, false, 4 * (item % 2) + q, item / 2, rb[p][q]);
            }
            else
            {
                d2B[stage * D2_STAGE_B + ((item / 32) * D2_ROW_B + 4 * (item % 32)) / 4] = rb[p];
            }
        }
    }
}

[shader("compute")]
[numthreads(WG_x, WG_y, 1)]
void matmul_dot2_main<T : ITorchVulkan>( // built for float16 only; T just matches the other matmul entry points
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform MatmulPushConstants<float> pcs)
{
    typealias Epilogue = LinearEpilogue<float, Activation<float>, AccumMatrix<half>>;
    const int TM = TILE_M / 16;

    AccumMatrix<half> bias = AccumMatrix<half>(pcs.tensor_bias, pcs.strides_bias);
    AccumMatrix<half> out = AccumMatrix<half>(pcs.tensor_out, pcs.strides_C);
    Epilogue epilogue = Epilogue(pcs.alpha, pcs.beta, bias);

    uint batch = wg.z;
    uint m0 = wg.y * TILE_M;
    uint n0 = wg.x * D2_TILE_N;
    uint K = pcs.K;

    Dot2Loader ld;
    ld.a_address = pcs.tensor_a;
    ld.b_address = pcs.tensor_b;
    ld.sa = pcs.strides_A;
    ld.sb = pcs.strides_B;
    ld.batch = batch;
    ld.m0 = m0;
    ld.n0 = n0;
    ld.M = pcs.M;
    ld.N = pcs.N;
    ld.K = K;
    ld.t = li.y * WG_x + li.x;
    ld.ra[0] = uint4(0); ld.ra[1] = uint4(0);
    ld.rb[0] = uint4(0); ld.rb[1] = uint4(0);

    float acc[TM][D2_TN];
    [unroll] for (int i = 0; i < TM; i++)
    {
        [unroll] for (int j = 0; j < D2_TN; j++) acc[i][j] = 0.0;
    }

    uint row0 = li.y * TM;
    uint col0 = li.x * D2_TN;

    // a workgroup whose tile lies inside the matrix loads its full K tiles without bounds checks
    bool interior = VEC_LOADS == 1 && m0 + TILE_M <= pcs.M && n0 + D2_TILE_N <= pcs.N;
    uint full_tiles = interior ? K / D2_TILE_K : 0;
    ld.setupFast();

    if (full_tiles > 0) ld.fetchFast(0);
    else ld.fetch(0);
    ld.stash(0);
    GroupMemoryBarrierWithGroupSync();

    uint stage = 0;
    uint tile = 0;
    for (uint k0 = 0; k0 < K; k0 += D2_TILE_K)
    {
        bool more = k0 + D2_TILE_K < K;
        tile++;
        if (more)
        {
            if (tile < full_tiles) ld.fetchFast(tile);
            else ld.fetch(k0 + D2_TILE_K);
        }

        [unroll] for (uint kp = 0; kp < D2_KP; kp++)
        {
            uint a[TM];
            [unroll] for (int i = 0; i < TM; i += 4)
            {
                uint4 a4 = d2A[stage * D2_STAGE_A + (kp * D2_ROW_A + row0 + i) / 4];
                a[i] = a4.x; a[i + 1] = a4.y; a[i + 2] = a4.z; a[i + 3] = a4.w;
            }
            uint b[D2_TN];
            [unroll] for (int j = 0; j < D2_TN; j += 4)
            {
                uint4 b4 = d2B[stage * D2_STAGE_B + (kp * D2_ROW_B + col0 + j) / 4];
                b[j] = b4.x; b[j + 1] = b4.y; b[j + 2] = b4.z; b[j + 3] = b4.w;
            }

            [unroll] for (int i = 0; i < TM; i++)
            {
                [unroll] for (int j = 0; j < D2_TN; j++) acc[i][j] = dot2Acc(a[i], b[j], acc[i][j]);
            }
        }

        // the other stage was last read before the previous barrier, so it can be refilled now
        if (more) ld.stash(stage ^ 1);
        GroupMemoryBarrierWithGroupSync();
        stage ^= 1;
    }

    [unroll] for (int i = 0; i < TM; i++)
    {
        uint row = m0 + row0 + i;
        if (row >= pcs.M) continue;
        [unroll] for (int j = 0; j < D2_TN; j++)
        {
            uint col = n0 + col0 + j;
            if (col < pcs.N) out.store(batch, row, col, epilogue.apply(batch, row, col, acc[i][j]));
        }
    }
}
