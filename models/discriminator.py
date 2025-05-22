# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

""" Res2Net implementation is adapted from https://github.com/wenet-e2e/wespeaker.
    Some modifications from the original architecture:
    1. Smaller kernel size for the input layer
    2. Smaller expansion in BasicBlockRes2Net
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from . import pooling_layers
from torch.nn.utils import spectral_norm

class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = "inplace" if self.inplace else ""
        return self.__class__.__name__ + " (" + inplace_str + ")"


class BasicBlockRes2Net(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockRes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(
            in_planes, width * scale, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    def __init__(
        self,
        block=BasicBlockRes2Net,
        num_blocks=[3, 4, 6, 3],
        m_channels=32,
        feat_dim=80,
        embedding_size=192,
        pooling_func="TSTP",
        two_emb_layer=False,
        **kwargs
    ):
        super(Res2Net, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)

        self.n_stats = 1 if pooling_func == "TAP" or pooling_func == "TSDP" else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion
        )
        if feat_dim == 257:
            linear_in = 33792 
        else:
            linear_in = self.stats_dim * block.expansion * self.n_stats        
        self.seg_1 = nn.Linear(linear_in, embedding_size)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


def xavier_init_layer(in_size, out_size=None, spec_norm=True, layer_type=nn.Linear, **kwargs):
    "Create a layer with spectral norm, xavier uniform init and zero bias"
    if out_size is None:
        out_size = in_size
    layer = layer_type(in_size, out_size, **kwargs)
    if spec_norm:
        layer = spectral_norm(layer)
    # initialization
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)
    return layer


class ResNetDiscriminator(nn.Module):
    def __init__(self, model_conf, activation=nn.LeakyReLU) -> None:
        super().__init__()
        self.activation = activation(negative_slope=0.3)
        feat_dim = model_conf.get('feat_dim', 257)
        embedding_size = model_conf.get('embedding_size', 192)
        out_size = model_conf.get('out_size', 1)
        self.random = model_conf.get('ranom', False)
        if self.random:
            feat_dim = feat_dim * 2
        self.backbone = Res2Net(feat_dim=feat_dim, embedding_size=embedding_size)
        self.linear = xavier_init_layer(in_size=embedding_size, out_size=out_size)
    
    def forward(self, x):
        if self.random:
            x = torch.cat((x, torch.randn_like(x)), dim=2)
        out = self.backbone(x)
        out = self.activation(out)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = OmegaConf.create({"feat_dim": 80})
    conf.merge_with_cli()
    x = torch.zeros(10, 1, conf.feat_dim, 300)
    # x = torch.zeros(10, 300, 80)
    model = Res2Net(feat_dim=conf.feat_dim, embedding_size=192, pooling_func="TSTP")
    model.eval()
    out = model(x)
    print(out.shape)  # torch.Size([10, 192])

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))  # 4.03M
