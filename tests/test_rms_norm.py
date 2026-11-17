"""rms_norm on the vulkan device (a native kernel) and torch.ops.torchvulkan.rms_norm_rope, checked
against the same math on the CPU."""
import pytest
import torch
import torch.nn.functional as F
import torchvulkan  # noqa: F401  registers the vulkan device and the torchvulkan ops

DEVICE = "vulkan"
DTYPES = [torch.float32, torch.float16] + ([torch.bfloat16] if torchvulkan.is_bf16_supported() else [])
# one rounding step of each dtype (the GPU sums in a different order)
TOL = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}


def close(out, ref, dtype):
    torch.testing.assert_close(out.cpu().float(), ref.float(), rtol=TOL[dtype], atol=TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape,normalized", [
    ((7, 128), (128,)),            # one subgroup per row
    ((3, 50, 30, 128), (128,)),    # attention heads
    ((2, 33, 3840), (3840,)),      # hidden states: one workgroup per row
    ((5, 1001), (1001,)),          # odd length
    ((4, 6, 40), (6, 40)),         # several normalised dimensions
])
@pytest.mark.parametrize("with_weight", [True, False])
def test_rms_norm_matches_cpu(dtype, shape, normalized, with_weight):
    g = torch.Generator().manual_seed(len(shape) * 100 + shape[-1])
    x = (torch.randn(shape, generator=g) * 3).to(dtype)
    w = torch.rand(normalized, generator=g).to(dtype) if with_weight else None
    ref = F.rms_norm(x.float(), normalized, None if w is None else w.float(), 1e-6).to(dtype)
    out = F.rms_norm(x.to(DEVICE), normalized, None if w is None else w.to(DEVICE), 1e-6)
    assert out.dtype == dtype and out.shape == x.shape
    close(out, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_rms_norm_default_eps_and_zero_rows(dtype):
    x = torch.randn(6, 64).to(dtype)
    x[2] = 0
    ref = F.rms_norm(x.float(), (64,), eps=torch.finfo(dtype).eps).to(dtype)
    out = F.rms_norm(x.to(DEVICE), (64,))
    close(out, ref, dtype)


def test_rms_norm_non_contiguous_input():
    x = torch.randn(40, 300)[:, 50:178]
    ref = F.rms_norm(x, (128,), eps=1e-6)
    close(F.rms_norm(x.to(DEVICE), (128,), eps=1e-6), ref, torch.float32)


def rope_reference(x, w, freqs, eps, split_half):
    """comfy_kitchen's rms_rope: rms_norm (rounded to x's dtype), then the 2x2 rotations in float32."""
    y = F.rms_norm(x.float(), (x.shape[-1],), w.float(), eps).to(x.dtype).float()
    n = x.shape[-1]
    if split_half:
        y0, y1 = y[..., : n // 2], y[..., n // 2:]
    else:
        y0, y1 = y[..., 0::2], y[..., 1::2]
    f = freqs.float()
    o0 = f[..., 0, 0] * y0 + f[..., 0, 1] * y1
    o1 = f[..., 1, 0] * y0 + f[..., 1, 1] * y1
    out = torch.cat([o0, o1], -1) if split_half else torch.stack([o0, o1], -1).flatten(-2)
    return out.to(x.dtype)


def make_freqs(shape, g):
    angles = torch.rand(*shape, generator=g) * 6.28
    c, s = torch.cos(angles), torch.sin(angles)
    return torch.stack([c, -s, s, c], -1).reshape(*shape, 2, 2)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("x_shape,freqs_shape", [
    ((1, 300, 6, 128), (1, 300, 1, 64)),    # [batch, tokens, heads, dim] (Z-Image)
    ((2, 4, 77, 64), (2, 1, 77, 32)),       # [batch, heads, tokens, dim] (Flux), freqs per batch
    ((2, 50, 3, 128), (1, 50, 1, 64)),      # freqs broadcast over the batch
])
@pytest.mark.parametrize("split_half", [False, True])
def test_rms_norm_rope_matches_cpu(dtype, x_shape, freqs_shape, split_half):
    g = torch.Generator().manual_seed(x_shape[1] * 10 + x_shape[3])
    x = (torch.randn(x_shape, generator=g) * 2).to(dtype)
    w = torch.rand(x_shape[-1], generator=g).to(dtype)
    freqs = make_freqs(freqs_shape, g)
    ref = rope_reference(x, w, freqs, 1e-6, split_half)
    out = torch.ops.torchvulkan.rms_norm_rope(x.to(DEVICE), w.to(DEVICE), freqs.to(DEVICE), 1e-6, split_half)
    assert out.dtype == dtype and out.shape == x.shape
    close(out, ref, dtype)


def test_rms_norm_rope_checks_freqs_shape():
    x = torch.randn(1, 8, 2, 16, device=DEVICE)
    with pytest.raises(RuntimeError, match="freqs_cis"):
        torch.ops.torchvulkan.rms_norm_rope(x, None, torch.zeros(1, 8, 1, 7, 2, 2, device=DEVICE), 1e-6, False)
