
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import native

class layer_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, training):
        N = 1
        if isinstance(normalized_shape, int):
            N = normalized_shape
        elif isinstance(normalized_shape, (list, tuple)):
            for i in normalized_shape:
                N *= i
        else:
            raise RuntimeError("unexpected type of normalized_shape".format(type(normalized_shape)))
        M = x.nelement() // N

        if x.is_cuda:

            y, mean, rstd = native.layer_norm_forward_cuda(x, weight, bias, M, N, eps)
        else:
            y, mean, rstd = native.layer_norm_forward_cpu(x, weight, bias, M, N, eps)

        if training:
            ctx.layer_norm_input = x
            ctx.layer_norm_parameters = (mean, rstd, weight, M, N)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.layer_norm_input
        mean, rstd, weight, M, N = ctx.layer_norm_parameters

        #output_mask = [True, True, True]
        output_mask = ctx.needs_input_grad #[True, True, True]
        print(output_mask)
        if grad_output.is_cuda:
            grad_input, grad_weight, grad_bias = native.layer_norm_backward_cuda(grad_output, x, mean, rstd, weight, M, N, output_mask)
        else:
            grad_input, grad_weight, grad_bias = native.layer_norm_backward_cpu(grad_output, x, mean, rstd, weight, M, N, output_mask)
        ctx.layer_norm_input = None
        ctx.layer_norm_parameters = None
        return grad_input, None, grad_weight, grad_bias, None, None

class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        nn.LayerNorm.__init__(self, normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        y = layer_norm.apply(x, self.normalized_shape, self.weight, self.bias, self.eps, self.training)
        return y


if __name__ == "__main__":
    seed = 2809
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic=True #https://github.com/pytorch/pytorch/issues/8019

    model = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            LayerNorm([64,56,56])
            )
    print(model)

    #model = model.cuda()
    model.train()
    iteration = 10
    for i in range(iteration):
        print("index: ", i)
        x = torch.rand(512,64,56,56)
        x = x - 0.5
        #x = x.cuda()

        y = model(x)
        z = y.sum()
        z.backward()

