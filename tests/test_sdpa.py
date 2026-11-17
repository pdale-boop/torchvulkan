"""scaled_dot_product_attention on the vulkan device, checked against PyTorch's CPU implementation.

The fused kernel covers inference without a mask, dropout or causal masking, for head sizes up to
512 (compiled as 64, 128, 256 and 512; smaller heads are padded). Masked and causal calls go
through PyTorch's own implementation and are checked here too.
"""
import pytest
import torch
import torch.nn.functional as F
import torchvulkan

DTYPES = [torch.float32, torch.float16] + ([torch.bfloat16] if torchvulkan.is_bf16_supported() else [])
TOL = {torch.float32: 2e-4, torch.float16: 4e-3, torch.bfloat16: 3e-2}


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("heads,lq,lk,head_dim", [
    (6, 200, 150, 64),    # split by heads
    (2, 700, 600, 128),   # one head over the cap: split by query rows
    (3, 130, 90, 512),
])
def test_split_dispatches_match_one_dispatch(monkeypatch, dtype, heads, lq, lk, head_dim):
    # TORCHVULKAN_ATTENTION_CHUNK caps the work per dispatch (in units of 2^20 rows x keys x head
    # size), as on integrated GPUs, where one long dispatch trips the GPU watchdog
    g = torch.Generator().manual_seed(heads * lq)
    q, k, v = (torch.randn(1, heads, n, head_dim, generator=g).to(dtype).to("vulkan") for n in (lq, lk, lk))
    monkeypatch.setenv("TORCHVULKAN_ATTENTION_CHUNK", "1000000000")
    whole = F.scaled_dot_product_attention(q, k, v).cpu()
    monkeypatch.setenv("TORCHVULKAN_ATTENTION_CHUNK", "1")
    split = F.scaled_dot_product_attention(q, k, v).cpu()
    torch.testing.assert_close(split, whole, rtol=0, atol=0)


def compare(q, k, v, dtype, **kwargs):
    ref = F.scaled_dot_product_attention(q, k, v, **kwargs)
    on_device = {key: val.to("vulkan") if isinstance(val, torch.Tensor) else val for key, val in kwargs.items()}
    out = F.scaled_dot_product_attention(*(t.to(dtype).to("vulkan") for t in (q, k, v)), **on_device)
    out = out.float().cpu()
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < TOL[dtype]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("batch,heads,lq,lk,head_dim", [
    (1, 4, 128, 128, 64),   # self-attention
    (2, 8, 100, 77, 80),    # cross-attention to text tokens, head size padded to 128
    (1, 2, 70, 33, 128),    # lengths that don't fill a tile
    (1, 2, 5, 1, 40),       # a single key
    (2, 2, 64, 64, 160),    # the 256 kernel
    (1, 1, 96, 96, 512),    # VAE mid-block attention: the 512 kernel, 8 lanes per row
])
def test_fused_matches_cpu(dtype, batch, heads, lq, lk, head_dim):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, lq, head_dim)
    k = torch.randn(batch, heads, lk, head_dim)
    v = torch.randn(batch, heads, lk, head_dim)
    compare(q, k, v, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_non_contiguous_inputs(dtype):
    # diffusion models build q, k and v as [batch, length, heads, dim] and transpose
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 90, 8, 64).transpose(1, 2) for _ in range(3))
    compare(q, k, v, dtype)


def test_explicit_scale():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 50, 64) for _ in range(3))
    compare(q, k, v, torch.float32, scale=0.3)


@pytest.mark.parametrize("case", ["causal", "bool mask", "float mask"])
def test_masked_and_causal_fall_back_correctly(case):
    torch.manual_seed(0)
    q = torch.randn(1, 4, 64, 64)
    k, v = torch.randn(1, 4, 48, 64), torch.randn(1, 4, 48, 64)
    if case == "causal":
        compare(q, k, v, torch.float32, is_causal=True)
    elif case == "bool mask":
        compare(q, k, v, torch.float32, attn_mask=torch.rand(64, 48) > 0.3)
    else:
        compare(q, k, v, torch.float32, attn_mask=torch.randn(64, 48))


def test_fused_does_not_materialise_scores():
    # the full score matrix would be 4 * 2048 * 2048 * 4 bytes = 64 MiB; the output is 2 MiB
    q, k, v = (torch.randn(1, 4, 2048, 64, device="vulkan") for _ in range(3))
    torchvulkan.synchronize()
    torchvulkan.reset_peak_memory_stats()
    before = torchvulkan.memory_allocated()
    out = F.scaled_dot_product_attention(q, k, v)
    torchvulkan.synchronize()
    extra = torchvulkan.max_memory_allocated() - before
    assert out.shape == q.shape
    assert extra < 16 * 2 ** 20, f"peak extra memory {extra / 2 ** 20:.0f} MiB"
