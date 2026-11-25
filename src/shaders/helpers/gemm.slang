module gemm;

public struct TensorStrides
{
    public uint batch;
    public uint row;
    public uint col;
    public uint padding;
}

public interface IMatmulLoader<T>
{
    T load(uint batch, uint row, uint col);

    // N consecutive elements of a row; a loader over memory that can promise their alignment overrides this with one load
    vector<T, N> loadRow<let N : int>(uint batch, uint row, uint col)
    {
        vector<T, N> v;
        [unroll] for (int i = 0; i < N; ++i) v[i] = load(batch, row, col + i);
        return v;
    }
}

public interface IMatmulStore<T>
{
    void store(uint batch, uint row, uint col, T value);

    void storeRow<let N : int>(uint batch, uint row, uint col, vector<T, N> value)
    {
        [unroll] for (int i = 0; i < N; ++i) store(batch, row, col + i, value[i]);
    }
}

public interface IMatmulEpilogue<T>
{
    T apply(uint batch, uint row, uint col, T acc);
}
