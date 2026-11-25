module gemm_simd;

__exported import core.dtype;
__exported import helpers.gemm;
__exported import helpers.unary_ops;

[[vk::constant_id(0)]] public const uint WG_x = 0;
[[vk::constant_id(1)]] public const uint WG_y = 0;
[[vk::constant_id(2)]] public const uint IS_ALIGNED = 1;
[[vk::constant_id(3)]] public const uint HAS_BIAS = 0;
[[vk::constant_id(4)]] public const uint ACTIVATION = 0;
[[vk::constant_id(5)]] public const uint TILE_M = 128;
// matmul only (conv_simd uses id 6): B's K dimension is the contiguous one (F.linear's w.t()), so its tile loads along K
[[vk::constant_id(7)]] public const uint B_KMAJOR = 0;
// matmul only, with IS_ALIGNED: the rows that loadRow/storeRow move start on VEC-element boundaries (host-checked)
[[vk::constant_id(8)]] public const uint VEC_MEM = 0;

public static const int TILE_N = 128;
public static const int TILE_K = 16;

public struct StridedMatrix<T> : IMatmulLoader<T>, IMatmulStore<T>
{
    public uint64_t address;
    public TensorStrides strides;

    public __init(uint64_t address, TensorStrides strides)
    {
        this.address = address;
        this.strides = strides;
    }

    uint offset(uint batch, uint row, uint col)
    {
        uint col_offset = (IS_ALIGNED == 1) ? col : col * strides.col;
        return batch * strides.batch + row * strides.row + col_offset;
    }

    public T load(uint batch, uint row, uint col)
    {
        T* tensor = (T*)address;
        return tensor[offset(batch, row, col)];
    }

    public void store(uint batch, uint row, uint col, T value)
    {
        T* tensor = (T*)address;
        tensor[offset(batch, row, col)] = value;
    }

    // only called under VEC_MEM, which promises the alignment. The word pointer comes straight from the address: AMD's
    // Windows compiler crashes on, or misreads through, a T* bitcast to a wider pointer (#69, #73).
    public override vector<T, N> loadRow<let N : int>(uint batch, uint row, uint col)
    {
        uint64_t p = address + uint64_t(offset(batch, row, col)) * sizeof(T);
        if (N * sizeof(T) == 16) return bit_cast<vector<T, N>>(loadAligned<16>((uint4*)p));
        if (N * sizeof(T) == 8) return bit_cast<vector<T, N>>(loadAligned<8>((uint2*)p));
        return bit_cast<vector<T, N>>(loadAligned<4>((uint*)p));
    }

    public override void storeRow<let N : int>(uint batch, uint row, uint col, vector<T, N> value)
    {
        uint64_t p = address + uint64_t(offset(batch, row, col)) * sizeof(T);
        if (N * sizeof(T) == 16) storeAligned<16>((uint4*)p, bit_cast<uint4>(value));
        else if (N * sizeof(T) == 8) storeAligned<8>((uint2*)p, bit_cast<uint2>(value));
        else storeAligned<4>((uint*)p, bit_cast<uint>(value));
    }
}

public struct AccumMatrix<T : ITorchVulkan> : IMatmulLoader<T.Accum>, IMatmulStore<T.Accum>
{
    public StridedMatrix<T> matrix;

    public __init(uint64_t address, TensorStrides strides)
    {
        this.matrix = StridedMatrix<T>(address, strides);
    }

    public T.Accum load(uint batch, uint row, uint col)
    {
        return matrix.load(batch, row, col).toAccum();
    }

    public void store(uint batch, uint row, uint col, T.Accum value)
    {
        matrix.store(batch, row, col, T.fromAccum(value));
    }

    public override vector<T.Accum, N> loadRow<let N : int>(uint batch, uint row, uint col)
    {
        vector<T, N> raw = matrix.loadRow<N>(batch, row, col);
        vector<T.Accum, N> v;
        [unroll] for (int i = 0; i < N; ++i) v[i] = raw[i].toAccum();
        return v;
    }

    public override void storeRow<let N : int>(uint batch, uint row, uint col, vector<T.Accum, N> value)
    {
        vector<T, N> raw;
        [unroll] for (int i = 0; i < N; ++i) raw[i] = T.fromAccum(value[i]);
        matrix.storeRow<N>(batch, row, col, raw);
    }
}

public struct LinearEpilogue<T : ITorchVulkan, Act : IUnaryOp<T>, Bias : IMatmulLoader<T>> : IMatmulEpilogue<T>
{
    public T alpha;
    public T beta;
    public Bias bias;

    public __init(T alpha, T beta, Bias bias)
    {
        this.alpha = alpha;
        this.beta = beta;
        this.bias = bias;
    }

    public T apply(uint batch, uint row, uint col, T acc)
    {
        T value = T.ext_mul(acc, alpha);
        if (HAS_BIAS == 1) value = T.ext_add(value, T.ext_mul(bias.load(batch, row, col), beta));
        return Act.eval(value);
    }
}

public struct Activation<T : ITorchVulkan> : IUnaryOp<T>
{
    public static T eval(T a)
    {
        switch (ACTIVATION)
        {
        case 1: return ReluOp<T>.eval(a);
        case 2: return GeluOp<T>.eval(a);
        default: return a;
        }
    }
}

public void matmulTile<
    T : __BuiltinArithmeticType,
    let VEC : int,
    A : IMatmulLoader<T>,
    B : IMatmulLoader<T>,
    E : IMatmulEpilogue<T>,
    S : IMatmulStore<T>>(
    uint3 li,
    uint3 wg,
    uint batch,
    __ref T smemA[TILE_M * (TILE_K + 1)],
    __ref vector<T, VEC> smemB[TILE_K * (TILE_N / VEC)],
    A a,
    B b,
    E epilogue,
    S out,
    uint M,
    uint N,
    uint K)
{
    const int THREAD_SIZE_M = TILE_M / TILE_K;
    const int THREAD_SIZE_N = TILE_N / TILE_K;
    const int VECTORS_PER_TILE_ROW_N = TILE_N / VEC;
    const int VECTORS_PER_TILE_ROW_K = TILE_K / VEC;
    const int VECTORS_PER_THREAD_ROW_M = THREAD_SIZE_N / VEC;

    // 2D: Slang can't emit an array size that multiplies the TILE_M spec constant by VEC
    vector<T, VEC> accum[THREAD_SIZE_M][THREAD_SIZE_N / VEC];
    [unroll] for (int i = 0; i < THREAD_SIZE_M; ++i)
    {
        [unroll] for (int j = 0; j < THREAD_SIZE_N / VEC; ++j) accum[i][j] = vector<T, VEC>(T(0));
    }

    uint thread_idx = li.y * WG_x + li.x;
    uint threads = WG_x * WG_y;
    uint thread_row_start = li.y * THREAD_SIZE_M;
    uint thread_col_start = li.x * THREAD_SIZE_N;

    uint vectors_A = (TILE_M * TILE_K) / VEC;
    uint vectors_B = (TILE_K * TILE_N) / VEC;
    uint loads_per_thread_A = vectors_A / threads;
    uint loads_per_thread_B = vectors_B / threads;

    for (uint k_block = 0; k_block < K; k_block += TILE_K)
    {
        // loads a TILE_M x TILE_K block of A into shared memory
        [unroll] for (int i = 0; i < loads_per_thread_A; ++i)
        {
            uint load_idx = thread_idx + (i * threads);
            uint row_A = load_idx / VECTORS_PER_TILE_ROW_K;
            uint col_A = (load_idx % VECTORS_PER_TILE_ROW_K) * VEC;

            uint global_row = wg.y * TILE_M + row_A;
            uint global_col = k_block + col_A;

            if (IS_ALIGNED == 1)
            {
                vector<T, VEC> loaded_vec = vector<T, VEC>(T(0));
                if (global_row < M && global_col < K)
                {
                    if (VEC_MEM == 1) loaded_vec = a.loadRow<VEC>(batch, global_row, global_col);
                    else [unroll] for (int v = 0; v < VEC; ++v) loaded_vec[v] = a.load(batch, global_row, global_col + v);
                }

                [unroll] for (int v = 0; v < VEC; ++v) smemA[row_A * (TILE_K + 1) + col_A + v] = loaded_vec[v];
            }
            else
            {
                // bounds check each element, so a non-multiple tail is padded with zero
                [unroll] for (int v = 0; v < VEC; ++v)
                {
                    if (global_row < M && (global_col + v) < K)
                    {
                        smemA[row_A * (TILE_K + 1) + col_A + v] = a.load(batch, global_row, global_col + v);
                    }
                    else
                    {
                        smemA[row_A * (TILE_K + 1) + col_A + v] = T(0);
                    }
                }
            }
        }

        if (B_KMAJOR == 1)
        {
            // b is B's transpose (N x K, contiguous along K): load it along K like A, then put each value at its [k][n] place
            [unroll] for (int i = 0; i < loads_per_thread_B; ++i)
            {
                uint load_idx = thread_idx + (i * threads);
                uint n = load_idx / VECTORS_PER_TILE_ROW_K;
                uint k = (load_idx % VECTORS_PER_TILE_ROW_K) * VEC;

                uint global_row = wg.x * TILE_N + n;
                uint global_col = k_block + k;

                vector<T, VEC> loaded_vec = vector<T, VEC>(T(0));
                if (IS_ALIGNED == 1)
                {
                    if (global_row < N && global_col < K)
                    {
                        if (VEC_MEM == 1) loaded_vec = b.loadRow<VEC>(batch, global_row, global_col);
                        else [unroll] for (int v = 0; v < VEC; ++v) loaded_vec[v] = b.load(batch, global_row, global_col + v);
                    }
                }
                else
                {
                    [unroll] for (int v = 0; v < VEC; ++v)
                    {
                        if (global_row < N && (global_col + v) < K) loaded_vec[v] = b.load(batch, global_row, global_col + v);
                    }
                }

                [unroll] for (int v = 0; v < VEC; ++v) smemB[(k + v) * VECTORS_PER_TILE_ROW_N + n / VEC][n % VEC] = loaded_vec[v];
            }
        }
        else
        {
            // loads a TILE_K x TILE_N block of B into shared memory
            [unroll] for (int i = 0; i < loads_per_thread_B; ++i)
            {
                uint load_idx = thread_idx + (i * threads);
                uint row_B = load_idx / VECTORS_PER_TILE_ROW_N;
                uint col_B = (load_idx % VECTORS_PER_TILE_ROW_N) * VEC;

                uint global_row = k_block + row_B;
                uint global_col = wg.x * TILE_N + col_B;

                vector<T, VEC> loaded_vec = vector<T, VEC>(T(0));

                if (IS_ALIGNED == 1)
                {
                    if (global_row < K && global_col < N)
                    {
                        if (VEC_MEM == 1) loaded_vec = b.loadRow<VEC>(batch, global_row, global_col);
                        else [unroll] for (int v = 0; v < VEC; ++v) loaded_vec[v] = b.load(batch, global_row, global_col + v);
                    }
                }
                else
                {
                    [unroll] for (int v = 0; v < VEC; ++v)
                    {
                        if (global_row < K && (global_col + v) < N)
                        {
                            loaded_vec[v] = b.load(batch, global_row, global_col + v);
                        }
                    }
                }

                uint vec_col = col_B / VEC;
                smemB[row_B * VECTORS_PER_TILE_ROW_N + vec_col] = loaded_vec;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // compute the matmul for this tile
        for (int k = 0; k < TILE_K; k++)
        {
            vector<T, VEC> b_frag[THREAD_SIZE_N / VEC];

            [unroll] for (int j = 0; j < THREAD_SIZE_N / VEC; ++j)
            {
                uint base_vec_col = (thread_col_start / VEC) + j;
                b_frag[j] = smemB[k * VECTORS_PER_TILE_ROW_N + base_vec_col];
            }

            [unroll] for (int i = 0; i < THREAD_SIZE_M; ++i)
            {
                T a_val = smemA[(thread_row_start + i) * (TILE_K + 1) + k];

                [unroll] for (int j = 0; j < THREAD_SIZE_N / VEC; ++j)
                {
                    accum[i][j] += b_frag[j] * a_val;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    uint global_row_start = wg.y * TILE_M + thread_row_start;
    uint global_col_start = wg.x * TILE_N + thread_col_start;

    // write the results back to global memory
    [unroll] for (int i = 0; i < THREAD_SIZE_M; ++i)
    {
        uint g_row = global_row_start + i;

        [unroll] for (int j_vec = 0; j_vec < VECTORS_PER_THREAD_ROW_M; ++j_vec)
        {
            uint g_col_base = global_col_start + (j_vec * VEC);
            vector<T, VEC> result_vec = accum[i][j_vec];

            if (IS_ALIGNED == 1)
            {
                if (g_row < M && g_col_base < N)
                {
                    if (VEC_MEM == 1)
                    {
                        [unroll] for (int v = 0; v < VEC; ++v) result_vec[v] = epilogue.apply(batch, g_row, g_col_base + v, result_vec[v]);
                        out.storeRow<VEC>(batch, g_row, g_col_base, result_vec);
                    }
                    else
                    {
                        [unroll] for (int v = 0; v < VEC; ++v)
                        {
                            uint g_col = g_col_base + v;
                            out.store(batch, g_row, g_col, epilogue.apply(batch, g_row, g_col, result_vec[v]));
                        }
                    }
                }
            }
            else
            {
                [unroll] for (int v = 0; v < VEC; ++v)
                {
                    uint g_col = g_col_base + v;
                    if (g_row < M && g_col < N)
                    {
                        out.store(batch, g_row, g_col, epilogue.apply(batch, g_row, g_col, result_vec[v]));
                    }
                }
            }
        }
    }
}
