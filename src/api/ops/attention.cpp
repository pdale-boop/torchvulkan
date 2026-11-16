#include <torch/extension.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <cmath>
#include <limits>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

// threads per workgroup; the shader gives each query row 4 lanes, so 32 rows per workgroup
constexpr uint32_t ATTENTION_WG_SIZE = 128;
constexpr uint32_t ATTENTION_ROWS = ATTENTION_WG_SIZE / 4;

// PyTorch's own composite implementation, which is what runs without this kernel.
// It converts boolean masks before reaching the math path, so call it rather than
// _scaled_dot_product_attention_math directly.
at::Tensor sdpa_math(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, double dropout_p, bool is_causal,
    std::optional<double> scale, bool enable_gqa)
{
    return at::native::scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

// The fused kernel covers inference without a mask, dropout or causal masking, for
// [..., length, head_dim] inputs with matching batch dims and head sizes up to 256.
bool fused_attention_supported(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const std::optional<at::Tensor>& attn_mask, double dropout_p, bool is_causal, bool enable_gqa)
{
    if (attn_mask.has_value() && attn_mask->defined()) return false;
    if (dropout_p > 0.0 || is_causal || enable_gqa) return false;
    if (at::GradMode::is_enabled() && (q.requires_grad() || k.requires_grad() || v.requires_grad())) return false;

    at::ScalarType dtype = q.scalar_type();
    if (dtype != at::kFloat && dtype != at::kHalf && dtype != at::kBFloat16) return false;
    if (k.scalar_type() != dtype || v.scalar_type() != dtype) return false;

    if (q.dim() < 3 || k.dim() != q.dim() || v.dim() != q.dim()) return false;
    if (q.sizes().slice(0, q.dim() - 2) != k.sizes().slice(0, k.dim() - 2)) return false;
    if (k.sizes() != v.sizes()) return false;

    int64_t head_dim = q.size(-1);
    if (k.size(-1) != head_dim || head_dim > 256 || head_dim == 0) return false;
    if (q.size(-2) == 0 || k.size(-2) == 0) return false;

    // the shader indexes with 32-bit offsets
    constexpr int64_t max_index = std::numeric_limits<uint32_t>::max();
    return q.numel() < max_index && k.numel() < max_index;
}

torchvulkan::Kernel attention_kernel_for(int64_t head_dim)
{
    if (head_dim <= 64) return torchvulkan::Kernel::ATTENTION_64;
    if (head_dim <= 128) return torchvulkan::Kernel::ATTENTION_128;
    return torchvulkan::Kernel::ATTENTION_256;
}

at::Tensor scaled_dot_product_attention_vulkan(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa)
{
    if (!fused_attention_supported(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa))
        return sdpa_math(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);

    DeviceContext* device = get_device_context(query);
    int64_t lq = query.size(-2);
    int64_t lk = key.size(-2);
    int64_t head_dim = query.size(-1);
    int64_t batch_heads = query.numel() / (lq * head_dim);

    if (batch_heads > device->properties.limits.maxComputeWorkGroupCount[1])
        return sdpa_math(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);

    at::Tensor q = query.reshape({batch_heads, lq, head_dim}).contiguous();
    at::Tensor k = key.reshape({batch_heads, lk, head_dim}).contiguous();
    at::Tensor v = value.reshape({batch_heads, lk, head_dim}).contiguous();
    at::Tensor out = at::empty({batch_heads, lq, head_dim}, q.options());

    float softmax_scale = scale.has_value() ? static_cast<float>(*scale) : 1.0f / std::sqrt(static_cast<float>(head_dim));

    SpecializationBuilder spd{};
    spd.push(ATTENTION_WG_SIZE);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(q))
       .push(get_tensor_address(k))
       .push(get_tensor_address(v))
       .push(get_output_address(out))
       .push(static_cast<uint32_t>(lq))
       .push(static_cast<uint32_t>(lk))
       .push(static_cast<uint32_t>(head_dim))
       .push(softmax_scale);

    device->shader_manager->dispatchShader(
        ShaderKey{attention_kernel_for(head_dim), query.scalar_type()},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {q, k, v},
        /* write = */ {out},
        static_cast<uint32_t>((lq + ATTENTION_ROWS - 1) / ATTENTION_ROWS),
        static_cast<uint32_t>(batch_heads),
        1
    );

    return out.view(query.sizes());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("scaled_dot_product_attention", &scaled_dot_product_attention_vulkan);
}
