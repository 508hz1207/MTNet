import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from .vit_encoder import Block
class LGFF(nn.Module):
    def __init__(self,
                 dlink_encoder_in_channel=[64,128,256,512],
                 embed_dim=768,
                 img_size=512,
                 mlp_ratio=4,
                 norm_layer= nn.LayerNorm,
                 num_heads=12,
                 patch_size=16,
                 qkv_bias=True,
                 use_rel_pos=False,
                 global_attn_indexes: Tuple[int, ...] = (),
                 window_size=0,
                 act_layer: Type[nn.Module] = nn.GELU,
                 rel_pos_zero_init: bool = True,
                 ):
        super().__init__()
        self.lgffs=nn.ModuleList()
        for i in range(4):
            lgff = LGFF_BASE(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                in_channel=dlink_encoder_in_channel,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                order=i,
                img_size=img_size,
            )
            self.lgffs.append(lgff)

    def forward(self, local_feature, global_feature):
        pass

class LGFF_BASE(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        in_channel: list[int] = [64, 128, 256, 512],
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        order:int=0,
        img_size=512,
    ) -> None:
        super().__init__()



        if img_size==512:
            vit_adapt_output_size=32
        elif img_size==256:
            vit_adapt_output_size = 16
        else:
            pass
        if order==3:
            self.vit_adapt=nn.ConvTranspose2d(in_channels=in_channel[order], out_channels=in_channel[order], kernel_size=4, stride=2, padding=1)

            self.conv1=nn.Sequential( nn.Conv2d(in_channel[order], 768, kernel_size=(1,1), stride=1, padding=0),
                nn.BatchNorm2d(768))
        else:
            self.vit_adapt=nn.AdaptiveMaxPool2d(output_size=(vit_adapt_output_size,vit_adapt_output_size))#(32,32)

            self.conv1=nn.Sequential( nn.Conv2d(in_channel[order], 768, kernel_size=(1,1), stride=1, padding=0),
                nn.BatchNorm2d(768))
        self.att= Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size,
                input_size=input_size,
            )

        conT_kernel_size=[4,4]
        conT_stride=[4,2]
        conT_padding=[0,1]
        if img_size==512:
            dlink_adapt_output_size=16
        elif img_size==256:
            dlink_adapt_output_size = 8
        else:
            pass
        if order==0 or order==1:
            self.dlink_condT = nn.ConvTranspose2d(in_channels=768, out_channels=768, kernel_size=conT_kernel_size[order],
                                                  stride=conT_stride[order], padding=conT_padding[order])

            self.conv2 = nn.Sequential(nn.Conv2d(768, in_channel[order], kernel_size=(1, 1), stride=1, padding=0),
                                       nn.BatchNorm2d(in_channel[order]))
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channel[order]*2, in_channel[order], kernel_size=(1, 1), stride=1, padding=0),
                nn.BatchNorm2d(in_channel[order]))

        elif order==2:
            self.dlink_condT = nn.Sequential(nn.Conv2d(768, 768, kernel_size=(1, 1), stride=1, padding=0),
                                       nn.BatchNorm2d(768))

            self.conv2 = nn.Sequential(nn.Conv2d(768, in_channel[order], kernel_size=(1, 1), stride=1, padding=0),
                                       nn.BatchNorm2d(in_channel[order]))
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channel[order]*2, in_channel[order], kernel_size=(1, 1), stride=1, padding=0),
                nn.BatchNorm2d(in_channel[order]))
        else: #
            self.dlink_condT = nn.AdaptiveMaxPool2d(output_size=(dlink_adapt_output_size, dlink_adapt_output_size))

            self.conv2 = nn.Sequential(nn.Conv2d(768, in_channel[order], kernel_size=(1, 1), stride=1, padding=0),
                                       nn.BatchNorm2d(in_channel[order]))
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channel[order]*2, in_channel[order], kernel_size=(1, 1), stride=1, padding=0),
                nn.BatchNorm2d(in_channel[order]))


        pass
    def forward(self,local_feature,global_feature):

        local_feature_adp=self.vit_adapt(local_feature)
        local_feature_con=self.conv1(local_feature_adp)
        local_feature_con=local_feature_con.permute(0,2,3,1)
        global_feature_fusion=self.att(global_feature+local_feature_con)
        global_feature=global_feature.permute(0,3,1,2)
        global_feature=self.dlink_condT(global_feature)
        global_feature=self.conv2(global_feature)
        a1=torch.concat((global_feature,local_feature),dim=1)
        local_feature_fusion=self.conv3(torch.concat((global_feature,local_feature),dim=1))


        return local_feature_fusion,global_feature_fusion