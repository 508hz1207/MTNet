
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from torchvision import models
from .resnet import *
from .basic_blocks import *
from .LGFF_module import LGFF
from .vit_encoder import *
from .curfnet import Curfres
from .center_bridge import *
from .Decoder import *
class LGC_Encoder(nn.Module):
    def __init__(
        self,
        config,
        encoder_1dconv=0,
        decoder_1dconv=4,
        num_classes=1,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:

        super().__init__()

        self.cen_encoder=EncoderViT(
                in_chans=in_chans,
                depth=depth,
                embed_dim=embed_dim,
                img_size=img_size,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                num_heads=num_heads,
                patch_size=patch_size,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                global_attn_indexes=global_attn_indexes,
                window_size=window_size,
                out_chans=out_chans,
            )
        self.config=config
        #=======================================================================
        #curfnet build
        self.curfnet=Curfres(config,num_classes=num_classes,num_channels=in_chans, encoder_1dconv=encoder_1dconv,decoder_1dconv=decoder_1dconv)
        #====================================
        #local  and global feature interactive fusion
        self.LGFF=LGFF(dlink_encoder_in_channel=[64,128,256,512],
                       embed_dim=embed_dim,
                       img_size=img_size,
                       mlp_ratio=mlp_ratio,
                       norm_layer=norm_layer,
                       num_heads=num_heads,
                       patch_size=patch_size,
                       qkv_bias=qkv_bias,
                       use_rel_pos=use_rel_pos,
                       global_attn_indexes=global_attn_indexes,
                       window_size=window_size,
                       act_layer= nn.GELU,
                                   )

        if self.config.USE_SCATT:
            self.curf_center_bridge = SCAtt(in_channels=512)
        if self.config.USE_SCMA:
            self.curf_center_bridge =SCMA(in_channels=512)
        if self.config.USE_VSSM_SS2D:
            self.vssm_module =VSSM_SS2D(in_channels=768,expansion_factor=2, lambda_factor=2)
        self.map_decoder = new_map_decoder(decoder_1dconv=4,filters=[32,64, 128, 256])

        # 记录梯度
        self.gradient = []
        # 记录输出的特征图
        self.output = []

    # use for feature map in heatmap_Grad_internet.py
    def save_grad(self, grad):
        self.gradient.append(grad)

    def get_grad(self):
        a = self.gradient
        b = self.gradient[-1]
        return self.gradient[-1].cpu().data

    def get_feature(self):
        return self.output[-1][0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_x=x# 10,5,512,512

        x = self.cen_encoder.patch_embed(x)#10,5(c),512,512->10,32,32,768(c)
        if self.cen_encoder.pos_embed is not None:
            x = x + self.cen_encoder.pos_embed#10,32,32,768(c)

        if self.curfnet.num_channels > 3:
            add_gps_osm = self.curfnet.addconv(res_x.narrow(1, 3,  self.curfnet.num_channels - 3))#(10,64,256,256)
            res_x = self.curfnet.firstconv(res_x.narrow(1, 0, 3))#convolution for only image#(10,64,256,256)
            res_x = res_x + add_gps_osm#add the trajectory and image conv result #(10,64,256,256)
        else:#I,T
            res_x = self.curfnet.firstconv(res_x)#(2,64,256,256)
        res_x = self.curfnet.firstbn(res_x)#(10,64,256,256)
        res_x = self.curfnet.firstrelu(res_x)#(10,64,256,256)
        res_x = self.curfnet.firstmaxpool(res_x)#(10,64,128,128)
        #firstly fusion# 2 vit block
        for blk in self.cen_encoder.blocks[0:2]:  # 2,4,4,2
            x = blk(x)  ##10,32,32,768(c)->#10,32,32,768(c)
        res_e_x1 = self.curfnet.encoder1(res_x)#(10,64,128,128)    #3resnetblock
        if self.config.USE_LGFF:
            res_e_x1,x=self.LGFF.lgffs[0](res_e_x1,x)

        for blk in self.cen_encoder.blocks[2:6]:#2,4,4,2
            x = blk(x)##10,32,32,768(c)->#10,32,32,768(c)
        res_e_x2 = self.curfnet.encoder2(res_e_x1)#(10,128,64,64)    #4resnetblock
        if self.config.USE_LGFF:
            res_e_x2, x = self.LGFF.lgffs[1](res_e_x2, x)
        for blk in self.cen_encoder.blocks[6:10]:#2,4,4,2
            x = blk(x)##10,32,32,768(c)->#10,32,32,768(c)
        res_e_x3 = self.curfnet.encoder3(res_e_x2)#(10,256,32,32)    #6resnetblock
        if self.config.USE_LGFF:
            res_e_x3, x = self.LGFF.lgffs[2](res_e_x3, x)
        for blk in self.cen_encoder.blocks[10:]:#2,4,4,2
            x = blk(x)##10,32,32,768(c)->#10,32,32,768(c)
        res_e_x4 = self.curfnet.encoder4(res_e_x3)#(10,512,16,16)    #3resnetblock
        if self.config.USE_LGFF:
            res_x4, x = self.LGFF.lgffs[3](res_e_x4, x)
        else:
            res_x4=res_e_x4

        if self.config.USE_SCATT or self.config.USE_SCMA:
            res_cen_x=self.curf_center_bridge(res_x4)##10,512,16,16->10,512,16,16
        else:
            res_cen_x=res_x4##10,512,16,16->10,512,16,16 map_decoder

        if  self.config.USE_VSSM_SS2D:
            x= self.vssm_module(x)#10,32,32,768(c)->10,768,32,32
        else:
            x=x.permute(0, 3, 1, 2)#10,32,32,768(c)->10,768,32,32


        if self.config.USE_RESENCODER_ADD_VITENCODER:#elect surface decoder add map decoder
            x = self.cen_encoder.neck(x)  # 10,768,32,32->cnn->10,256(c),32,32
            res_d4 = self.curfnet.decoder4(res_cen_x) + res_e_x3 + x  # 10,512,16,16->(10,256,32,32)
            x=self.map_decoder.map_decoder[0](x)#conv 10,256(c),32,32->10,256(c),32,32
            vit_d4 = self.map_decoder.map_decoder[1](x + res_d4)  # 10,256(c),32,32->10,128,64,64
            res_d3 = self.curfnet.decoder3(res_d4) + res_e_x2 + vit_d4  # (10,256,32,32)->(10,128,64,64)
            vit_d3 = self.map_decoder.map_decoder[2](vit_d4 + res_d3)  # 10,128,64,64->10,64,128,128
            res_d2 = self.curfnet.decoder2(res_d3) + res_e_x1 + vit_d3  # (10,128,64,64)->(10,64,128,128)
            vit_d2 = self.map_decoder.map_decoder[3](vit_d3 + res_d2)  # 10,64,128,128->10,32,256,256

            res_d1 = self.curfnet.decoder1(res_d2)  # (10,64,128,128)->(10,64,256,256)
            res_out = self.curfnet.finaldeconv1(res_d1)  # (10,64,256,256)->(10,32,512,512)
            res_out = self.curfnet.finalrelu1(res_out)  # (10,32,512,512)
            res_out = self.curfnet.finalconv2(res_out)  # (10,32,512,512)
            res_out = self.curfnet.finalrelu2(res_out)  # (10,32,512,512)
            res_out = self.curfnet.finalconv3(res_out)  # (10,1,512,512)no process nn.sigmoid()

            return res_out
        else:
            res_d4 = self.curfnet.decoder4(res_cen_x) #+ res_e_x3  # 10,512,16,16->(10,256,32,32)
            res_d3 = self.curfnet.decoder3(res_d4) #+ res_e_x2  # (10,256,32,32)->(10,128,64,64)
            res_d2 = self.curfnet.decoder2(res_d3) #+ res_e_x1  # (10,128,64,64)->(10,64,128,128)
            res_d1 = self.curfnet.decoder1(res_d2)  # (10,64,128,128)->(10,64,256,256)
            res_out = self.curfnet.finaldeconv1(res_d1)  # (10,64,256,256)->(10,32,512,512)
            res_out = self.curfnet.finalrelu1(res_out)  # (10,32,512,512)
            res_out = self.curfnet.finalconv2(res_out)  # (10,32,512,512)
            res_out = self.curfnet.finalrelu2(res_out)  # (10,32,512,512)
            res_out = self.curfnet.finalconv3(res_out)  # (10,1,512,512)
            x = self.cen_encoder.neck(x)  # 10,768,32,32->cnn->10,256(c),32,32
            x = self.map_decoder.map_decoder[0](x)  # conv 10,256(c),32,32->10,256(c),32,32
            vit_d4 = self.map_decoder.map_decoder[1](x)  # 10,256(c),32,32->10,128,64,64
            vit_d3 = self.map_decoder.map_decoder[2](vit_d4)  # 10,128,64,64->10,64,128,128
            vit_d2 = self.map_decoder.map_decoder[3](vit_d3)  # 10,64,128,128->8,32,256,256
            return res_out

