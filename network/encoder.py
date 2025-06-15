import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial

from torch import batch_norm

from .dcn.modules.deform_conv import *
import functools


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(self.weight.size())
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x
class InstanceNormND(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        # Expecting x: (B, C, ...)
        dims = list(range(2, x.ndim))  # [2, 3] for 4D; [2, 3, 4] for 5D
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        shape = [1, -1] + [1] * (x.ndim - 2)
        x = self.weight.view(*shape) * x + self.bias.view(*shape)
        return x



class RDVC(nn.Module):
    def __init__(self, nf):
        super(RDVC, self).__init__()

        self.dcn0 = nn.Sequential(

            LayerNorm(nf, eps=1e-6, data_format='channels_first'),
            DVC(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HWD')
        )
        self.dcn1 = nn.Sequential(
            LayerNorm(nf, eps=1e-6, data_format='channels_first'),
            DVC(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HWD')
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x

class ResBlockStandard(nn.Module):
    def __init__(self, nf):
        super(ResBlockStandard, self).__init__()
        self.dcn0 = nn.Sequential(
            nn.Conv3d(nf, nf, kernel_size=3, padding=1, stride=1),
            LayerNorm(nf, eps=1e-6, data_format='channels_first')
        )
        self.dcn1 = nn.Sequential(
            nn.Conv3d(nf, nf, kernel_size=3, padding=1, stride=1),
            LayerNorm(nf, eps=1e-6, data_format='channels_first')
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x


class encoder(nn.Module):

    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],deform_layers=None):
        super().__init__()
        self.deform_layers = deform_layers if deform_layers is not None else [0,1,2,3]
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
              LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i in range(4):
            layers = []
            for j in range(depths[i]):

                if i in self.deform_layers:
                    layers.append(RDVC(dims[i]))
                else:
                    layers.append(ResBlockStandard(dims[i]))
            stage = nn.Sequential(*layers)
            self.stages.append(stage)
        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return layers

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 0:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
            else:
                outs.append(x)   # 每层结束不加layerNorm，因为模块中已经加了
        return tuple(outs)
    def forward(self, x):
        x = self.forward_features(x)
        return x


