#include <torch/extension.h>
#include <c10/util/irange.h>
#include <limits>
#include <optional>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

namespace {

constexpr uint32_t RMS_NORM_WG_SIZE = 256;

bool is_norm_float(at::ScalarType t) { return t == at::kFloat || t == at::kHalf || t == at::kBFloat16; }

// rope geometry for rms_norm_kernel: x is [d0, dim1, dim2, n] and freqs a contiguous float32
// [f0, f1, f2, n / 2, 2, 2] broadcast over it (each f either matches x or is 1); strides count
// rows of n / 2 matrices
struct RopeLayout
{
    uint32_t mode = 0;  // 0 none, 1 interleaved pairs, 2 split halves
    at::Tensor freqs;
    int64_t dim1 = 1, dim2 = 1;
    int64_t fstride0 = 0, fstride1 = 0, fstride2 = 0;
};

// x: contiguous [rows, n] in a float dtype. One subgroup per row when the row is short, so that a
// [tokens * heads, head_dim] input fills the GPU, else one workgroup per row.
at::Tensor rms_norm_kernel(const at::Tensor& x, const std::optional<at::Tensor>& weight, double eps, const RopeLayout& rope)
{
    const int64_t rows = x.size(0);
    const int64_t n = x.size(1);
    at::Tensor out = at::empty_like(x);
    if (rows == 0) return out;

    DeviceContext* device = get_device_context(x);
    const uint32_t sg = device->default_subgroup_size;
    TORCH_CHECK(sg >= 4 && RMS_NORM_WG_SIZE % sg == 0, "rms_norm: unsupported subgroup size ", sg);
    const bool row_per_subgroup = n <= 4 * static_cast<int64_t>(sg) * 8;  // up to 8 elements per lane
    const int64_t rows_per_wg = row_per_subgroup ? RMS_NORM_WG_SIZE / sg : 1;

    at::Tensor w;
    if (weight && weight->defined()) w = weight->to(at::kFloat).contiguous();
    at::Tensor freqs = rope.mode != 0 ? rope.freqs : at::Tensor();

    SpecializationBuilder spec{};
    spec.push(RMS_NORM_WG_SIZE)
        .push(sg)
        .push(static_cast<uint32_t>(row_per_subgroup))
        .push(rope.mode)
        .push(static_cast<uint32_t>(w.defined()));
    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(x))
       .push(w.defined() ? get_tensor_address(w) : (uint64_t)0)
       .push(get_tensor_address(out))
       .push(freqs.defined() ? get_tensor_address(freqs) : (uint64_t)0)
       .push(static_cast<uint32_t>(rows))
       .push(static_cast<uint32_t>(n))
       .push(static_cast<float>(eps))
       .push(static_cast<uint32_t>(rope.dim1))
       .push(static_cast<uint32_t>(rope.dim2))
       .push(static_cast<uint32_t>(rope.fstride0))
       .push(static_cast<uint32_t>(rope.fstride1))
       .push(static_cast<uint32_t>(rope.fstride2));

    std::vector<at::Tensor> read = {x};
    if (w.defined()) read.push_back(w);
    if (freqs.defined()) read.push_back(freqs);
    const int64_t groups = (rows + rows_per_wg - 1) / rows_per_wg;
    const uint32_t gx = static_cast<uint32_t>(std::min<int64_t>(groups, 65535));
    const uint32_t gy = static_cast<uint32_t>((groups + 65534) / 65535);
    device->shader_manager->dispatchShader(ShaderKey{torchvulkan::Kernel::RMS_NORM, x.scalar_type()}, sg,
        spec.build(), pcs.build(), read, {out}, gx, gy, 1);
    return out;
}

// rms_norm's eps when none is given: the input dtype's machine epsilon (torch.finfo(dtype).eps)
double default_rms_eps(at::ScalarType t)
{
    switch (t) {
    case at::kHalf: return 0.0009765625;
    case at::kBFloat16: return 0.0078125;
    case at::kDouble: return std::numeric_limits<double>::epsilon();
    default: return std::numeric_limits<float>::epsilon();
    }
}

// PyTorch's composite rms_norm (at::native::rms_norm_symint is not exported on Windows), for the
// cases the kernel does not take: other dtypes, autograd
at::Tensor rms_norm_composite(const at::Tensor& input, int64_t dims, const std::optional<at::Tensor>& weight, std::optional<double> eps)
{
    const bool upcast = input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16;
    at::Tensor x = upcast ? input.to(at::kFloat) : input;
    const double e = eps.has_value() ? *eps : default_rms_eps(input.scalar_type());
    std::vector<int64_t> reduce_dims;
    for (int64_t i = input.dim() - dims; i < input.dim(); ++i) reduce_dims.push_back(i);
    at::Tensor out = x * at::rsqrt(x.pow(2).mean(reduce_dims, /*keepdim=*/true).add_(e));
    if (weight && weight->defined()) out = out * *weight;
    return out.to(input.scalar_type());
}

// aten::rms_norm, which PyTorch otherwise runs as eight kernels (upcast, pow, mean, add, rsqrt, two
// multiplies, downcast). Same math: float32 throughout, weight applied in float32, one rounding.
at::Tensor rms_norm_vulkan(const at::Tensor& input, c10::SymIntArrayRef normalized_shape, const std::optional<at::Tensor>& weight, std::optional<double> eps)
{
    const int64_t dims = static_cast<int64_t>(normalized_shape.size());
    const bool supported = is_norm_float(input.scalar_type()) && dims >= 1 && dims <= input.dim() &&
                           (!weight || !weight->defined() || is_norm_float(weight->scalar_type())) &&
                           !(at::GradMode::is_enabled() && (input.requires_grad() || (weight && weight->defined() && weight->requires_grad())));
    if (!supported || dims > input.dim()) return rms_norm_composite(input, std::min<int64_t>(dims, input.dim()), weight, eps);

    int64_t n = 1;
    for (const auto i : c10::irange(dims)) {
        TORCH_CHECK(input.sym_size(input.dim() - dims + i) == normalized_shape[i],
            "rms_norm: normalized_shape ", normalized_shape, " does not match the input's trailing dimensions ", input.sym_sizes());
        n *= normalized_shape[i].expect_int();
    }
    if (weight && weight->defined())
        TORCH_CHECK(weight->numel() == n, "rms_norm: weight has ", weight->numel(), " elements, expected ", n);
    if (n == 0 || n > std::numeric_limits<uint32_t>::max() || input.numel() > std::numeric_limits<uint32_t>::max())
        return rms_norm_composite(input, dims, weight, eps);

    const double e = eps.has_value() ? *eps : default_rms_eps(input.scalar_type());
    at::Tensor x = input.contiguous().view({-1, n});
    return rms_norm_kernel(x, weight, e, RopeLayout{}).view(input.sizes());
}

// torchvulkan::rms_norm_rope(x, weight, freqs_cis, eps, split_half): rms_norm over the last dimension
// of a 4D x [d0, d1, d2, head_dim] (as [batch, tokens, heads, head_dim] or [batch, heads, tokens,
// head_dim]), then a rotary embedding with freqs_cis, a [f0, f1, f2, head_dim / 2, 2, 2] tensor of
// 2x2 matrices broadcast over x's first three dimensions, in one kernel. split_half rotates
// (i, i + head_dim / 2) instead of (2i, 2i + 1).
at::Tensor rms_norm_rope_vulkan(const at::Tensor& x, const std::optional<at::Tensor>& weight, const at::Tensor& freqs_cis, double eps, bool split_half)
{
    TORCH_CHECK(x.dim() == 4, "rms_norm_rope: expected a 4D x, got ", x.sizes());
    TORCH_CHECK(is_norm_float(x.scalar_type()), "rms_norm_rope: expected float32, float16 or bfloat16, got ", x.scalar_type());
    const int64_t n = x.size(3);
    TORCH_CHECK(n % 2 == 0, "rms_norm_rope: head_dim must be even, got ", n);
    TORCH_CHECK(freqs_cis.dim() == 6 && freqs_cis.size(3) == n / 2 && freqs_cis.size(4) == 2 && freqs_cis.size(5) == 2,
        "rms_norm_rope: expected freqs_cis [., ., ., ", n / 2, ", 2, 2], got ", freqs_cis.sizes());
    for (int d = 0; d < 3; ++d)
        TORCH_CHECK(freqs_cis.size(d) == 1 || freqs_cis.size(d) == x.size(d),
            "rms_norm_rope: freqs_cis ", freqs_cis.sizes(), " does not broadcast to x ", x.sizes());
    if (weight && weight->defined())
        TORCH_CHECK(weight->numel() == n, "rms_norm_rope: weight has ", weight->numel(), " elements, expected ", n);
    TORCH_CHECK(x.numel() <= std::numeric_limits<uint32_t>::max(), "rms_norm_rope: tensor too large");

    RopeLayout rope;
    rope.mode = split_half ? 2 : 1;
    rope.freqs = freqs_cis.to(x.device(), at::kFloat).contiguous();
    rope.dim1 = x.size(1);
    rope.dim2 = x.size(2);
    const int64_t f1 = freqs_cis.size(1), f2 = freqs_cis.size(2);
    rope.fstride2 = f2 > 1 ? 1 : 0;
    rope.fstride1 = f1 > 1 ? f2 : 0;
    rope.fstride0 = freqs_cis.size(0) > 1 ? f1 * f2 : 0;
    return rms_norm_kernel(x.contiguous().view({-1, n}), weight, eps, rope).view(x.sizes());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("rms_norm", &rms_norm_vulkan);
}

TORCH_LIBRARY_FRAGMENT(torchvulkan, m) {
    m.def("rms_norm_rope(Tensor x, Tensor? weight, Tensor freqs_cis, float eps=1e-6, bool split_half=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchvulkan, PrivateUse1, m) {
    m.impl("rms_norm_rope", &rms_norm_rope_vulkan);
}
