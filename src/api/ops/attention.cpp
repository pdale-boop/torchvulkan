#include <torch/extension.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/autotuner.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

// threads per workgroup; each query row gets 4 lanes (8 for the 512 kernel)
constexpr uint32_t ATTENTION_WG_SIZE = 128;
constexpr int64_t ATTENTION_MAX_HEAD_DIM = 512;

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
// [..., length, head_dim] inputs with matching batch dims and head sizes up to 512.
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
    if (k.size(-1) != head_dim || head_dim > ATTENTION_MAX_HEAD_DIM || head_dim == 0) return false;
    if (q.size(-2) == 0 || k.size(-2) == 0) return false;

    // the shader indexes with 32-bit offsets
    constexpr int64_t max_index = std::numeric_limits<uint32_t>::max();
    return q.numel() < max_index && k.numel() < max_index;
}

torchvulkan::Kernel attention_kernel_for(int64_t head_dim)
{
    if (head_dim <= 64) return torchvulkan::Kernel::ATTENTION_64;
    if (head_dim <= 128) return torchvulkan::Kernel::ATTENTION_128;
    if (head_dim <= 256) return torchvulkan::Kernel::ATTENTION_256;
    return torchvulkan::Kernel::ATTENTION_512;
}

// lanes per query row, matching the RL constant in scripts/config.py
uint32_t attention_row_lanes(int64_t head_dim)
{
    return head_dim > 256 ? 8 : 4;
}

constexpr int64_t ATTENTION_COOP_SUBGROUPS = 4;  // SUBGROUPS in attention_coop.slang
constexpr int64_t ATTENTION_COOP_ROWS = ATTENTION_COOP_SUBGROUPS * 16;  // query rows per workgroup
constexpr int64_t ATTENTION_COOP_KEYS = 32;  // BC in attention_coop.slang

bool env_flag(const char* name)
{
    const char* v = std::getenv(name);
    return v && v[0] != '\0' && v[0] != '0';
}

// One way to run the fused attention: its dispatch, and its output [batch * heads, rows, head_dim],
// whose first lq rows are the result.
struct AttentionPlan
{
    TuneDispatch dispatch;
    at::Tensor out;
    int64_t lq = 0;

    at::Tensor result() const { return out.size(1) == lq ? out : out.narrow(1, 0, lq).contiguous(); }
};

// The cooperative-matrix kernel: float16 or bfloat16, head sizes 64 and 128, on devices with 16x16x16
// T x T -> float32 cooperative matrices and subgroup shuffles (the softmax combines lanes with them).
bool coop_applies(DeviceContext* device, at::ScalarType dtype, int64_t head_dim)
{
    if (!device->support_coopmat || !device->support_subgroup_shuffle) return false;
    if ((dtype != at::kHalf && dtype != at::kBFloat16) || (head_dim != 64 && head_dim != 128)) return false;

    bool has_fragment = false;
    for (const CoopMatConfig& c : device->cache.getCoopMatConfig(dtype, dtype, at::kFloat, at::kFloat)) {
        if (c.m == 16 && c.n == 16 && c.k == 16) has_fragment = true;
    }
    const uint32_t subgroup = device->default_subgroup_size;
    const uint64_t element = c10::elementSize(dtype);
    const uint64_t shared = 2 * ATTENTION_COOP_KEYS * head_dim * element            // K and V tiles
                          + ATTENTION_COOP_ROWS * ATTENTION_COOP_KEYS * (element + 4) // P and S
                          + ATTENTION_COOP_ROWS * 16 * 4;                             // row factors
    return has_fragment && subgroup > 0 &&
           ATTENTION_COOP_SUBGROUPS * subgroup <= device->properties.limits.maxComputeWorkGroupInvocations &&
           shared <= device->properties.limits.maxComputeSharedMemorySize;
}

// q, k and v are contiguous [batch * heads, length, head_dim]
AttentionPlan coop_plan(DeviceContext* device, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, float softmax_scale)
{
    const int64_t batch_heads = q.size(0);
    const int64_t lq = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t lq_padded = (lq + ATTENTION_COOP_ROWS - 1) / ATTENTION_COOP_ROWS * ATTENTION_COOP_ROWS;
    const uint32_t subgroup = device->default_subgroup_size;

    // whole 16-row fragments are loaded and stored, so q and the output are padded to whole
    // workgroups. Query rows are independent and the padded rows' output is discarded, so the
    // padding needs no particular contents.
    at::Tensor q_padded = q;
    if (lq_padded != lq) {
        q_padded = at::empty({batch_heads, lq_padded, head_dim}, q.options());
        q_padded.narrow(1, 0, lq).copy_(q);
    }

    AttentionPlan plan;
    plan.lq = lq;
    plan.out = at::empty({batch_heads, lq_padded, head_dim}, q.options());
    TuneDispatch& d = plan.dispatch;
    d.key = ShaderKey{head_dim == 64 ? torchvulkan::Kernel::ATTENTION_COOP_64 : torchvulkan::Kernel::ATTENTION_COOP_128, q.scalar_type()};
    d.subgroupSize = subgroup;
    d.spec.push(static_cast<uint32_t>(ATTENTION_COOP_SUBGROUPS * subgroup)).push(subgroup);
    d.push.push(get_tensor_address(q_padded))
          .push(get_tensor_address(k))
          .push(get_tensor_address(v))
          .push(get_output_address(plan.out))
          .push(static_cast<uint32_t>(lq_padded))
          .push(static_cast<uint32_t>(k.size(1)))
          .push(softmax_scale)
          .push(static_cast<uint32_t>(0));
    d.read = {q_padded, k, v};
    d.write = {plan.out};
    d.groupX = static_cast<uint32_t>(lq_padded / ATTENTION_COOP_ROWS);
    d.groupY = static_cast<uint32_t>(batch_heads);
    return plan;
}

// The scalar kernel needs quad operations (or shuffles, for 8 lanes per row).
bool scalar_applies(DeviceContext* device, int64_t head_dim)
{
    return attention_row_lanes(head_dim) == 4 ? device->support_subgroup_quad : device->support_subgroup_shuffle;
}

AttentionPlan scalar_plan(DeviceContext* device, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, float softmax_scale)
{
    const int64_t batch_heads = q.size(0);
    const int64_t lq = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t rows_per_group = ATTENTION_WG_SIZE / attention_row_lanes(head_dim);

    AttentionPlan plan;
    plan.lq = lq;
    plan.out = at::empty({batch_heads, lq, head_dim}, q.options());
    TuneDispatch& d = plan.dispatch;
    d.key = ShaderKey{attention_kernel_for(head_dim), q.scalar_type()};
    d.subgroupSize = device->default_subgroup_size;
    d.spec.push(ATTENTION_WG_SIZE);
    d.push.push(get_tensor_address(q))
          .push(get_tensor_address(k))
          .push(get_tensor_address(v))
          .push(get_output_address(plan.out))
          .push(static_cast<uint32_t>(lq))
          .push(static_cast<uint32_t>(k.size(1)))
          .push(static_cast<uint32_t>(head_dim))
          .push(softmax_scale);
    d.read = {q, k, v};
    d.write = {plan.out};
    d.groupX = static_cast<uint32_t>((lq + rows_per_group - 1) / rows_per_group);
    d.groupY = static_cast<uint32_t>(batch_heads);
    return plan;
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
    // like every other op, the dtype must be supported on the device (bf16 isn't on Turing or llvmpipe)
    if (!is_dtype_supported(device, query.scalar_type()))
        return sdpa_math(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
    int64_t lq = query.size(-2);
    int64_t lk = key.size(-2);
    int64_t head_dim = query.size(-1);
    int64_t batch_heads = query.numel() / (lq * head_dim);

    if (batch_heads > device->properties.limits.maxComputeWorkGroupCount[1])
        return sdpa_math(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);

    at::Tensor q = query.reshape({batch_heads, lq, head_dim}).contiguous();
    at::Tensor k = key.reshape({batch_heads, lk, head_dim}).contiguous();
    at::Tensor v = value.reshape({batch_heads, lk, head_dim}).contiguous();
    float softmax_scale = scale.has_value() ? static_cast<float>(*scale) : 1.0f / std::sqrt(static_cast<float>(head_dim));

    // TORCHVULKAN_ATTENTION_SCALAR=1 or _COOP=1 forces one kernel (for comparisons)
    bool coop = coop_applies(device, query.scalar_type(), head_dim) && !env_flag("TORCHVULKAN_ATTENTION_SCALAR");
    bool scalar = scalar_applies(device, head_dim) && !(coop && env_flag("TORCHVULKAN_ATTENTION_COOP"));
    if (!coop && !scalar)
        return sdpa_math(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);

    // Cooperative matrices win where the GPU has matrix units, but some drivers emulate them (an
    // RDNA2 GPU under Windows, where the coop kernel was slower), so when both kernels apply the
    // autotuner times them once per device and problem class.
    bool use_coop = coop;
    if (coop && scalar && device->tuner) {
        TuneSpace space;
        space.params = {{"coop", {0, 1}}};
        auto bucket = [](int64_t x) -> uint64_t { uint64_t b = 1; while (b < static_cast<uint64_t>(x) && b < 65536) b <<= 1; return b; };
        const uint64_t tuned_id = bucket(lq) | (bucket(lk) << 17) | (static_cast<uint64_t>(head_dim) << 34);
        const ShaderKey key{head_dim == 64 ? torchvulkan::Kernel::ATTENTION_COOP_64 : torchvulkan::Kernel::ATTENTION_COOP_128, query.scalar_type()};
        const TuneConfig& chosen = device->tuner->select(key, tuned_id, space, [&](const TuneConfig& cfg) {
            return (cfg[0] == 1 ? coop_plan(device, q, k, v, softmax_scale) : scalar_plan(device, q, k, v, softmax_scale)).dispatch;
        });
        if (!chosen.empty()) use_coop = chosen[0] == 1;
    }

    AttentionPlan plan = use_coop ? coop_plan(device, q, k, v, softmax_scale) : scalar_plan(device, q, k, v, softmax_scale);
    const TuneDispatch& d = plan.dispatch;
    device->shader_manager->dispatchShader(d.key, d.subgroupSize, d.spec.build(), d.push.build(), d.read, d.write, d.groupX, d.groupY, d.groupZ);
    return plan.result().view(query.sizes());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("scaled_dot_product_attention", &scaled_dot_product_attention_vulkan);
}
