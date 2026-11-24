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

void softmax_rows(const at::Tensor& input, const at::Tensor& out, bool log, const at::Tensor& lse)
{
    TORCH_CHECK(input.is_contiguous() && out.is_contiguous() && softmax_rows_supported(input), "torchvulkan [ERROR]: softmax_rows needs contiguous float rows.");
    const int64_t length = input.dim() == 0 ? 1 : input.size(-1);
    const int64_t rows = input.numel() / std::max<int64_t>(length, 1);
    if (rows == 0) return;
    const bool out_accum = out.scalar_type() == at::kFloat && input.scalar_type() != at::kFloat;
    const bool write_lse = lse.defined();
    TORCH_CHECK(!write_lse || (lse.is_contiguous() && lse.scalar_type() == at::kFloat && lse.numel() == rows),
                "torchvulkan [ERROR]: softmax_rows' lse must be one contiguous float32 per row.");

    DeviceContext* device = get_device_context(input);
    const uint32_t workgroup = length >= 1024 ? 256 : length >= 256 ? 128 : 64;
    SpecializationBuilder spd{};
    spd.push(workgroup).push(uint32_t(log)).push(uint32_t(out_accum)).push(uint32_t(write_lse));
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
