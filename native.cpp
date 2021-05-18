
#include <torch/extension.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward_cpu(
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor & bias,
    int64_t M, int64_t N, double eps) {
    return at::native::layer_norm_cpu(input, weight, bias, M, N, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_layer_norm_cpu(
    const at::Tensor & grad_out,
    const at::Tensor & input,
    const at::Tensor & mean,
    const at::Tensor & rstd,
    const at::Tensor & weight,
    int64_t M, int64_t N, std::array<bool,3> output_mask) {
    return at::native::layer_norm_backward_cpu(grad_out, input, mean, rstd, weight, M, N, output_mask);
}

//std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward_cuda(
//    const at::Tensor & input,
//    const at::Tensor & weight,
//    const at::Tensor & bias,
//    int64_t M, int64_t N, double eps) {
//    return at::native::layer_norm_cuda(input, weight, bias, M, N, eps);
//}
//
//std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_layer_norm_cuda(
//    const at::Tensor & grad_out,
//    const at::Tensor & input,
//    const at::Tensor & mean,
//    const at::Tensor & rstd,
//    const at::Tensor & weight,
//    int64_t M, int64_t N, std::array<bool,3> output_mask) {
//    return at::layer_norm_backward_cuda(grad_out, input, mean, rstd, weight, M, N, output_mask);
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_forward_cpu",  &layer_norm_forward_cpu, "layer norm forward (cpu version)");
    m.def("layer_norm_backward_cpu", &backward_layer_norm_cpu, "layer norm backward (cpu version)");
    //m.def("layer_norm_forward_cuda", &layer_norm_forward_cuda, "layer norm forward (cuda version)");
    //m.def("layer_norm_backward_cuda",&backward_layer_norm_cuda, "layer norm backward (cuda version)");
}

