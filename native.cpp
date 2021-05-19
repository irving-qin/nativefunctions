
#include <torch/extension.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward_cpu(
    const at::Tensor & input,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    at::IntArrayRef normalized_shape,
#endif
    const at::Tensor & weight,
    const at::Tensor & bias,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
#else
    int64_t M, int64_t N,
#endif
    double eps) {

#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    return at::native::layer_norm_cpu(input, normalized_shape, weight, bias, eps);
#else
    return at::native::layer_norm_cpu(input, weight, bias, M, N, eps);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_cpu(
    const at::Tensor & grad_out,
    const at::Tensor & input,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    at::IntArrayRef normalized_shape,
#endif
    const at::Tensor & mean,
    const at::Tensor & rstd,
    const at::Tensor & weight,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    const at::Tensor & bias,
#else
    int64_t M, int64_t N,
#endif
    std::array<bool,3> output_mask) {

#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    return at::native::layer_norm_backward_cpu(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
#else
    return at::native::layer_norm_backward_cpu(grad_out, input, mean, rstd, weight, M, N, output_mask);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward_cuda(
    const at::Tensor & input,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    at::IntArrayRef normalized_shape,
#endif
    const at::Tensor & weight,
    const at::Tensor & bias,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
#else
    int64_t M, int64_t N,
#endif
    double eps) {
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    return at::native::layer_norm_cuda(input, normalized_shape, weight, bias, eps);
#else
    return at::native::layer_norm_cuda(input, weight, bias, M, N, eps);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_cuda(
    const at::Tensor & grad_out,
    const at::Tensor & input,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    at::IntArrayRef normalized_shape,
#endif
    const at::Tensor & mean,
    const at::Tensor & rstd,
    const at::Tensor & weight,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    const at::Tensor & bias,
#else
    int64_t M, int64_t N,
#endif
    std::array<bool,3> output_mask) {

#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    return at::native::layer_norm_backward_cuda(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
#else
    return at::native::layer_norm_backward_cuda(grad_out, input, mean, rstd, weight, M, N, output_mask);
#endif
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_forward_cpu", &layer_norm_forward_cpu, "layer norm forward (cpu version)");
    m.def("layer_norm_backward_cpu", &layer_norm_backward_cpu, "layer norm backward (cpu version)");
    m.def("layer_norm_forward_cuda", &layer_norm_forward_cuda, "layer norm forward (cuda version)");
    m.def("layer_norm_backward_cuda", &layer_norm_backward_cuda, "layer norm backward (cuda version)");
}

