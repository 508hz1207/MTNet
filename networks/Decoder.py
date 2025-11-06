import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Type
from .DeformableDecoder import DeformableRoadDecoder
from .basic_blocks import *
nonlinearity = partial(F.relu, inplace=True)

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



class new_map_decoder(nn.Module):
    def __init__(self,decoder_1dconv=4,filters=[32,64, 128, 256]):
        # activation = nn.GELU
        activation = nn.ReLU()
        super().__init__()
        #####
        self.map_decoder = nn.ModuleList()
        conv1=nn.Sequential(
            nn.Conv2d(filters[3], filters[3], kernel_size=(1, 1), stride=1, padding=0),
            LayerNorm2d(filters[3]),
            activation, )#nn.ReLU()
        self.map_decoder.append(conv1)
        if decoder_1dconv == 0:
            self.decoder = DecoderBlock
        elif decoder_1dconv == 2:
            self.decoder = DecoderBlock1DConv2
        elif decoder_1dconv == 4:#进入该处
            self.decoder = DecoderBlock1DConv4

        map_decoder4 = self.decoder(filters[3], filters[2])  # filters =[32,64, 128, 256]
        self.map_decoder.append(map_decoder4)

        map_decoder3 = self.decoder(filters[2], filters[1])
        self.map_decoder.append(map_decoder3)

        map_decoder2 = self.decoder(filters[1], filters[0])
        self.map_decoder.append(map_decoder2)

        map_decoder1 = nn.Sequential(nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2), )
        self.map_decoder.append(map_decoder1)

