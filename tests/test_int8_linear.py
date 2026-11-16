"""torch.ops.torchvulkan.quantize_int8_rowwise and int8_scaled_mm, the two halves of an int8 linear
layer with per-row activation scales (as in comfy_kitchen's int8_linear). Both are compared exactly
with the same math run step by step on the CPU."""
import pytest
import torch
import torchvulkan  # noqa: F401  registers the vulkan device and the torchvulkan ops

DEVICE = 'vulkan'
FLOATS = [torch.float32, torch.float16, torch.bfloat16]


def supported(dtype):
    if dtype == torch.bfloat16 and not torchvulkan.is_bf16_supported():
        pytest.skip('bfloat16 is not supported on this device')


# the CPU reference: comfy_kitchen's eager quantize_int8_rowwise
def quantize_reference(x):
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    s = scale.to(x.dtype)
    s = torch.where(s == 0, torch.full_like(s, torch.finfo(x.dtype).tiny), s)
    return (x / s).round_().clamp_(-128.0, 127.0).to(torch.int8), scale


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('shape', [(1, 7), (5, 64), (77, 3840), (3, 2, 1001), (513, 256)])
def test_quantize_int8_rowwise(dtype, shape):
    supported(dtype)
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(shape, generator=g) * 3).to(dtype)
    x[..., 0, :] = 0  # an all-zero row
    q_ref, s_ref = quantize_reference(x)
    q, s = torch.ops.torchvulkan.quantize_int8_rowwise(x.to(DEVICE))
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    assert q.shape == x.shape and s.shape == (*x.shape[:-1], 1)
    torch.testing.assert_close(s.cpu(), s_ref, rtol=0, atol=0)
    torch.testing.assert_close(q.cpu(), q_ref, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', FLOATS)
def test_quantize_int8_rowwise_strided_rows(dtype):
    supported(dtype)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(40, 300, generator=g).to(dtype)[:, 20:220]  # rows 300 apart
    q_ref, s_ref = quantize_reference(x)
    q, s = torch.ops.torchvulkan.quantize_int8_rowwise(x.to(DEVICE)[:, :])
    torch.testing.assert_close(s.cpu(), s_ref, rtol=0, atol=0)
    torch.testing.assert_close(q.cpu(), q_ref, rtol=0, atol=0)


SHAPES = [(1, 256, 17), (33, 100, 50), (64, 64, 64), (77, 384, 300), (130, 257, 70)]


@pytest.fixture(params=['default', 'scalar'])
def kernel_path(request, monkeypatch):
    # 'default' runs the fused cooperative-matrix kernel where the device has one; 'scalar' forces
    # the fallback (the scalar _int_mm and separate ops)
    if request.param == 'scalar':
        monkeypatch.setenv('TORCHVULKAN_INT_MM_SCALAR', '1')
    else:
        monkeypatch.delenv('TORCHVULKAN_INT_MM_SCALAR', raising=False)
    return request.param


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('m,k,n', SHAPES)
@pytest.mark.parametrize('per_channel', [False, True])
@pytest.mark.parametrize('with_bias', [False, True])
def test_int8_scaled_mm(kernel_path, dtype, m, k, n, per_channel, with_bias):
    supported(dtype)
    g = torch.Generator().manual_seed(m * 1000 + n)
    a = torch.randint(-128, 128, (m, k), dtype=torch.int8, generator=g)
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=g)  # a Linear weight [N, K]
    a_scale = torch.rand(m, 1, generator=g) * 0.01 + 1e-4
    w_scale = torch.rand(n if per_channel else 1, generator=g) * 0.01 + 1e-4
    bias = torch.randn(n, generator=g).to(dtype) if with_bias else None

    expected = (torch._int_mm(a, w.t()).float() * (a_scale * w_scale.reshape(1, -1))).to(dtype)
    if bias is not None:
        expected = expected + bias
    result = torch.ops.torchvulkan.int8_scaled_mm(
        a.to(DEVICE), w.to(DEVICE).t(), a_scale.to(DEVICE), w_scale.to(DEVICE),
        None if bias is None else bias.to(DEVICE), dtype)
    assert result.dtype == dtype and result.shape == (m, n)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_int8_scaled_mm_row_major_b():
    g = torch.Generator().manual_seed(2)
    a = torch.randint(-128, 128, (40, 96), dtype=torch.int8, generator=g)
    b = torch.randint(-128, 128, (96, 72), dtype=torch.int8, generator=g)
    a_scale = torch.rand(40, generator=g) * 0.01
    b_scale = torch.rand(1, generator=g) * 0.01
    expected = (torch._int_mm(a, b).float() * (a_scale.reshape(-1, 1) * b_scale)).to(torch.float32)
    result = torch.ops.torchvulkan.int8_scaled_mm(a.to(DEVICE), b.to(DEVICE), a_scale.to(DEVICE), b_scale.to(DEVICE), None, torch.float32)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_int8_scaled_mm_checks_shapes():
    a = torch.zeros(4, 8, dtype=torch.int8, device=DEVICE)
    b = torch.zeros(8, 6, dtype=torch.int8, device=DEVICE)
    with pytest.raises(RuntimeError, match='a_scale'):
        torch.ops.torchvulkan.int8_scaled_mm(a, b, torch.ones(3, device=DEVICE), torch.ones(1, device=DEVICE))
    with pytest.raises(RuntimeError, match='b_scale'):
        torch.ops.torchvulkan.int8_scaled_mm(a, b, torch.ones(4, device=DEVICE), torch.ones(5, device=DEVICE))
