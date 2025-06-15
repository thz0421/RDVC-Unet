#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _triple
from torch.autograd.function import once_differentiable

import D3D

class DeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias,
                stride, padding, dilation, group, deformable_groups, im2col_step):
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.kernel_size = _triple(weight.shape[2:5])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = D3D.deform_conv_forward(input, weight, bias,
                                         offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1],ctx.kernel_size[2],
                                         ctx.stride[0], ctx.stride[1],ctx.stride[2],
                                         ctx.padding[0], ctx.padding[1],ctx.padding[2],
                                         ctx.dilation[0], ctx.dilation[1],ctx.dilation[2],
                                         ctx.group,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = \
            D3D.deform_conv_backward(input, weight,
                                     bias,
                                     offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                     ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                     ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                     ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                                     ctx.group,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None


def calc_3d_deformconv_flops(input_shape, deform_conv_module):
    """
    计算3D可变形卷积的FLOPs
    input_shape: (N, C_in, D_in, H_in, W_in)
    deform_conv_module: 你的DeformConvPack_d_new实例
    """
    N, Cin, D_in, H_in, W_in = input_shape
    Cout = deform_conv_module.out_channels if hasattr(deform_conv_module, 'out_channels') else \
    deform_conv_module.weight.shape[0]
    Kd, Kh, Kw = deform_conv_module.kernel_size if hasattr(deform_conv_module, 'kernel_size') else (3, 3, 3)
    stride = deform_conv_module.stride if hasattr(deform_conv_module, 'stride') else (1, 1, 1)
    padding = deform_conv_module.padding if hasattr(deform_conv_module, 'padding') else (1, 1, 1)
    dilation = deform_conv_module.dilation if hasattr(deform_conv_module, 'dilation') else (1, 1, 1)

    # 计算输出尺寸
    def calc_out_dim(in_size, kernel_size, pad, stride, dil):
        return (in_size + 2 * pad - dil * (kernel_size - 1) - 1) // stride + 1

    D_out = calc_out_dim(D_in, Kd, padding[0], stride[0], dilation[0])
    H_out = calc_out_dim(H_in, Kh, padding[1], stride[1], dilation[1])
    W_out = calc_out_dim(W_in, Kw, padding[2], stride[2], dilation[2])

    # FLOPs = 2 * Cout * Cin * Kd * Kh * Kw * D_out * H_out * W_out * N
    flops = 2 * Cout * Cin * Kd * Kh * Kw * D_out * H_out * W_out * N
    return flops
