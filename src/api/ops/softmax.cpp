#include <torch/extension.h>
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

namespace torchvulkan {

bool softmax_rows_supported(const at::Tensor& input)
{
    const at::ScalarType dtype = input.scalar_type();
    return (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) && is_dtype_supported(input, dtype) &&
           input.numel() <= std::numeric_limits<int32_t>::max();
}

void softmax_rows(const at::Tensor& input, const at::Tensor& out, bool log, const at::Tensor& lse, bool masked_zero)
{
    TORCH_CHECK(input.is_contiguous() && out.is_contiguous() && softmax_rows_supported(input), "torchvulkan [ERROR]: softmax_rows needs contiguous float rows.");
    const int64_t length = input.dim() == 0 ? 1 : input.size(-1);
    const int64_t rows = input.numel() / std::max<int64_t>(length, 1);
    if (rows == 0) return;
    const bool out_accum = out.scalar_type() == at::kFloat && input.scalar_type() != at::kFloat;
    TORCH_CHECK(out.numel() == input.numel() && (out_accum || out.scalar_type() == input.scalar_type()),
                "torchvulkan [ERROR]: softmax_rows' out must be the input's shape, in its dtype or float32.");
    const bool write_lse = lse.defined();
    TORCH_CHECK(!write_lse || (lse.is_contiguous() && lse.scalar_type() == at::kFloat && lse.numel() == rows),
                "torchvulkan [ERROR]: softmax_rows' lse must be one contiguous float32 per row.");
    TORCH_CHECK(!masked_zero || !log, "torchvulkan [ERROR]: softmax_rows' masked_zero is for probabilities.");

    DeviceContext* device = get_device_context(input);
    // a thread takes vec elements per step; about two steps per thread, from one subgroup (64 without subgroup
    // arithmetic) up to 256 threads a row
    const int64_t vec = get_dtype_vec_size(input.scalar_type());
    const uint32_t subgroup = device->default_subgroup_size;
    const bool subgroup_reduce = device->support_subgroup_arithmetic && subgroup > 0 && subgroup <= 256;
    uint32_t workgroup = subgroup_reduce ? subgroup : 64;
    while (workgroup < 256 && workgroup * vec * 2 < length) workgroup *= 2;
    const int64_t steps = (length + workgroup * vec - 1) / (workgroup * vec);
    const bool cached = steps <= 8; // MAX_STEPS in softmax.slang
    const uint64_t in_bytes = vec * input.element_size(), out_bytes = vec * out.element_size();
    const bool vec_aligned = length % vec == 0 && get_tensor_address(input) % in_bytes == 0 &&
                             get_output_address(out) % out_bytes == 0;
    SpecializationBuilder spd{};
    spd.push(workgroup)
        .push(uint32_t(log))
        .push(uint32_t(out_accum))
        .push(uint32_t(write_lse))
        .push(uint32_t(subgroup_reduce))
        .push(uint32_t(vec_aligned))
        .push(uint32_t(cached))
        .push(uint32_t(masked_zero));
    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(input))
        .push(get_output_address(out))
        .push(write_lse ? get_output_address(lse) : uint64_t(0))
        .push(uint32_t(rows))
        .push(uint32_t(length));
    std::vector<at::Tensor> written{out};
    if (write_lse) written.push_back(lse);
    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::SOFTMAX, input.scalar_type()},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {input},
        /* write = */ written,
        static_cast<uint32_t>(rows), 1, 1
    );
}

} // namespace torchvulkan
