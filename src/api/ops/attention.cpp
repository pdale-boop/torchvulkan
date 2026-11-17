#include <torch/extension.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
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

constexpr int64_t ATTENTION_COOP_KEYS = 32;  // BC in attention_coop.slang

bool env_flag(const char* name)
{
    const char* v = std::getenv(name);
    return v && v[0] != '\0' && v[0] != '0';
}

// One way to run the fused attention: its output [batch * heads, rows, head_dim], whose first lq
// rows are the result, and the dispatch for any range of heads and query rows (row_start a multiple
// of row_step), so that a large problem can run as several dispatches.
struct AttentionPlan
{
    at::Tensor out;
    int64_t lq = 0;
    int64_t heads = 0;
    int64_t rows = 0;      // query rows the kernel covers (lq, or lq padded to whole workgroups)
    int64_t row_step = 1;  // query rows per workgroup
    std::function<TuneDispatch(int64_t head_start, int64_t head_count, int64_t row_start, int64_t row_count)> chunk;

    at::Tensor result() const { return out.size(1) == lq ? out : out.narrow(1, 0, lq).contiguous(); }
};

// The most work (query rows x keys x head size, summed over heads) one dispatch may do. A single
// long dispatch can trip the GPU watchdog: the i915 preemption timeout (~640 ms) killed the context
// on an Intel HD 530 at 16 heads x 1024 tokens, and Windows resets a GPU after 2 s (TDR). Dispatches
// are preemption points, so the work is split by heads, then by query rows. The caps keep a dispatch
// to ~0.25 s on an HD 530 (integrated) and under ~1 s on a slow discrete GPU, while ordinary shapes
// stay one dispatch on a fast one. TORCHVULKAN_ATTENTION_CHUNK sets the cap in units of 2^20.
int64_t attention_dispatch_work_cap(DeviceContext* device)
{
    if (const char* env = std::getenv("TORCHVULKAN_ATTENTION_CHUNK"); env && env[0])
        return std::max<int64_t>(1, std::strtoll(env, nullptr, 10)) << 20;
    switch (device->properties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return int64_t(1) << 26;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return std::numeric_limits<int64_t>::max();  // no watchdog
    default: return int64_t(1) << 36;
    }
}

// heads and query rows per dispatch under the work cap
struct AttentionSplit { int64_t heads; int64_t rows; };

AttentionSplit attention_split(DeviceContext* device, const AttentionPlan& plan, int64_t lk, int64_t head_dim)
{
    const int64_t cap = attention_dispatch_work_cap(device);
    const int64_t per_row = lk * head_dim;
    if (plan.rows > cap / per_row) {
        // one head is already too much: split its query rows
        return {1, std::min(plan.rows, std::max<int64_t>(plan.row_step, cap / per_row / plan.row_step * plan.row_step))};
    }
    return {std::min(plan.heads, std::max<int64_t>(1, cap / (plan.rows * per_row))), plan.rows};
}

// the first dispatch of the split, which the autotuner times (so tuning never runs a dispatch
// longer than the cap either)
TuneDispatch first_attention_chunk(DeviceContext* device, const AttentionPlan& plan, int64_t lk, int64_t head_dim)
{
    const AttentionSplit s = attention_split(device, plan, lk, head_dim);
    return plan.chunk(0, s.heads, 0, s.rows);
}

void run_attention(DeviceContext* device, const AttentionPlan& plan, int64_t lk, int64_t head_dim)
{
    const AttentionSplit s = attention_split(device, plan, lk, head_dim);
    for (int64_t h = 0; h < plan.heads; h += s.heads) {
        for (int64_t r = 0; r < plan.rows; r += s.rows) {
            TuneDispatch d = plan.chunk(h, std::min(s.heads, plan.heads - h), r, std::min(s.rows, plan.rows - r));
            device->shader_manager->dispatchShader(d.key, d.subgroupSize, d.spec.build(), d.push.build(), d.read, d.write, d.groupX, d.groupY, d.groupZ);
        }
    }
}

// The cooperative-matrix kernel: float16 or bfloat16, head sizes 64 and 128, on devices with 16x16x16
// T x T -> float32 cooperative matrices and subgroup shuffles (the softmax combines lanes with them).
// Subgroups per workgroup for the cooperative-matrix kernel, or 0 where it doesn't apply. Each
// subgroup owns 16 query rows. 8 subgroups share each staged K/V tile between twice as many rows
// as 4 and were ~18% faster on an RTX 5070 Ti (24 x 4608 x 128 bf16: 24.9 -> 20.4 ms), but their
// P/S tiles need more shared memory (exactly the 48 KB NVIDIA offers at head size 128), so devices
// with less get 4.
int64_t coop_subgroups(DeviceContext* device, at::ScalarType dtype, int64_t head_dim)
{
    if (!device->support_coopmat || !device->support_subgroup_shuffle) return 0;
    if ((dtype != at::kHalf && dtype != at::kBFloat16) || (head_dim != 64 && head_dim != 128)) return 0;

    bool has_fragment = false;
    for (const CoopMatConfig& c : device->cache.getCoopMatConfig(dtype, dtype, at::kFloat, at::kFloat)) {
        if (c.m == 16 && c.n == 16 && c.k == 16) has_fragment = true;
    }
    const uint32_t subgroup = device->default_subgroup_size;
    if (!has_fragment || subgroup == 0) return 0;
    const uint64_t element = c10::elementSize(dtype);
    for (int64_t subgroups : {8, 4}) {
        const uint64_t rows = subgroups * 16;
        const uint64_t shared = 2 * ATTENTION_COOP_KEYS * head_dim * element  // K and V tiles
                              + rows * ATTENTION_COOP_KEYS * (element + 4)    // P and S
                              + rows * 16 * 4;                                // row factors
        if (subgroups * subgroup <= device->properties.limits.maxComputeWorkGroupInvocations &&
            shared <= device->properties.limits.maxComputeSharedMemorySize)
            return subgroups;
    }
    return 0;
}

// q, k and v are contiguous [batch * heads, length, head_dim]
AttentionPlan coop_plan(DeviceContext* device, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, float softmax_scale)
{
    const int64_t batch_heads = q.size(0);
    const int64_t lq = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t subgroups = coop_subgroups(device, q.scalar_type(), head_dim);
    const int64_t rows_per_group = subgroups * 16;
    const int64_t lq_padded = (lq + rows_per_group - 1) / rows_per_group * rows_per_group;
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
    plan.heads = batch_heads;
    plan.rows = lq_padded;
    plan.row_step = rows_per_group;
    plan.out = at::empty({batch_heads, lq_padded, head_dim}, q.options());
    const at::Tensor out = plan.out;
    const uint64_t elem = c10::elementSize(q.scalar_type());
    const int64_t lk = k.size(1);
    plan.chunk = [=](int64_t head_start, int64_t head_count, int64_t row_start, int64_t row_count) {
        // a chunk of heads starts further into each tensor; its rows start at row_offset
        const uint64_t q_skip = head_start * lq_padded * head_dim * elem;
        const uint64_t kv_skip = head_start * lk * head_dim * elem;
        TuneDispatch d;
        d.key = ShaderKey{head_dim == 64 ? torchvulkan::Kernel::ATTENTION_COOP_64 : torchvulkan::Kernel::ATTENTION_COOP_128, q.scalar_type()};
        d.subgroupSize = subgroup;
        d.spec.push(static_cast<uint32_t>(subgroups * subgroup)).push(subgroup).push(static_cast<uint32_t>(subgroups));
        d.push.push(get_tensor_address(q_padded) + q_skip)
              .push(get_tensor_address(k) + kv_skip)
              .push(get_tensor_address(v) + kv_skip)
              .push(get_output_address(out) + q_skip)
              .push(static_cast<uint32_t>(lq_padded))
              .push(static_cast<uint32_t>(lk))
              .push(softmax_scale)
              .push(static_cast<uint32_t>(row_start));
        d.read = {q_padded, k, v};
        d.write = {out};
        d.groupX = static_cast<uint32_t>((row_count + rows_per_group - 1) / rows_per_group);
        d.groupY = static_cast<uint32_t>(head_count);
        return d;
    };
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
    plan.heads = batch_heads;
    plan.rows = lq;
    plan.row_step = rows_per_group;
    plan.out = at::empty({batch_heads, lq, head_dim}, q.options());
    const at::Tensor out = plan.out;
    const uint64_t elem = c10::elementSize(q.scalar_type());
    const int64_t lk = k.size(1);
    const uint32_t subgroup = device->default_subgroup_size;
    plan.chunk = [=](int64_t head_start, int64_t head_count, int64_t row_start, int64_t row_count) {
        const uint64_t q_skip = head_start * lq * head_dim * elem;
        const uint64_t kv_skip = head_start * lk * head_dim * elem;
        TuneDispatch d;
        d.key = ShaderKey{attention_kernel_for(head_dim), q.scalar_type()};
        d.subgroupSize = subgroup;
        d.spec.push(ATTENTION_WG_SIZE);
        d.push.push(get_tensor_address(q) + q_skip)
              .push(get_tensor_address(k) + kv_skip)
              .push(get_tensor_address(v) + kv_skip)
              .push(get_output_address(out) + q_skip)
              .push(static_cast<uint32_t>(lq))
              .push(static_cast<uint32_t>(lk))
              .push(static_cast<uint32_t>(head_dim))
              .push(softmax_scale)
              .push(static_cast<uint32_t>(row_start))
              .push(static_cast<uint32_t>(0));
        d.read = {q, k, v};
        d.write = {out};
        d.groupX = static_cast<uint32_t>((row_count + rows_per_group - 1) / rows_per_group);
        d.groupY = static_cast<uint32_t>(head_count);
        return d;
    };
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
    bool coop = coop_subgroups(device, query.scalar_type(), head_dim) > 0 && !env_flag("TORCHVULKAN_ATTENTION_SCALAR");
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
            const AttentionPlan p = cfg[0] == 1 ? coop_plan(device, q, k, v, softmax_scale) : scalar_plan(device, q, k, v, softmax_scale);
            return first_attention_chunk(device, p, lk, head_dim);
        });
        if (!chosen.empty()) use_coop = chosen[0] == 1;
    }

    AttentionPlan plan = use_coop ? coop_plan(device, q, k, v, softmax_scale) : scalar_plan(device, q, k, v, softmax_scale);
    run_attention(device, plan, lk, head_dim);
    return plan.result().view(query.sizes());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("scaled_dot_product_attention", &scaled_dot_product_attention_vulkan);
}
