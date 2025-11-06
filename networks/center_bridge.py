import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import time
import math
import copy
from functools import partial
from networks.vmamba import mamba_init

from networks.mamba_simple import Mamba
try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, alpha_init=0):
        super(SpatialAttentionModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float))

    def forward(self, X):

        C, H, W = X.size(1), X.size(2), X.size(3)

        A = self.conv1(X)
        B = self.conv2(X)
        D = self.conv3(X)

        N = H * W
        b=A.size(0)
        A = A.view(A.size(0), C, -1).contiguous()
        B = B.view(B.size(0), C, -1).contiguous()
        D = D.view(D.size(0), C, -1).contiguous()

        A_T = A.permute(0, 2, 1).contiguous()
        M = torch.bmm(A_T, B)
        M = F.softmax(M, dim=-1)

        weighted_C = torch.bmm(M, D.permute(0, 2, 1).contiguous() )
        weighted_C = weighted_C.permute(0, 2, 1).view(D.size(0), C, H, W).contiguous()

        E = self.alpha * weighted_C

        return E


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, beta_init=0):
        super(ChannelAttentionModule, self).__init__()


        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float))

    def forward(self, X):#10,512,16,16
        C, H, W = X.size(1), X.size(2), X.size(3)#
        N = H * W
        X_reshaped = X.view(X.size(0), C, -1).contiguous()
        S = torch.bmm(X_reshaped, X_reshaped.permute(0, 2, 1).contiguous() )
        S = F.softmax(S, dim=-1)
        weighted_X = torch.bmm(S.permute(0, 2, 1).contiguous() , X_reshaped)
        weighted_X = weighted_X.view(X.size(0), C, H, W).contiguous()
        E = self.beta * weighted_X
        return E


class SCAtt(nn.Module):
    def __init__(self,in_channels=512):
        super().__init__()
        self.spatial_att=SpatialAttentionModule(in_channels)
        self.channels_att=ChannelAttentionModule(in_channels)
        self.conv1=nn.Conv2d(in_channels*3, in_channels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self,x):#10,512,16,16
        spatial_att_x=self.spatial_att(x)#10,512,16,16->10,512,16,16
        channels_att_x=self.channels_att(x)#10,512,16,16->10,512,16,16
        x=torch.concat((x,spatial_att_x,channels_att_x),dim=1)#10,1536,16,16
        x=self.conv1(x)
        return x
class LayerNorm2d1(nn.Module):
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

class SCMA(nn.Module):
    def __init__(self,in_channels=512):
        super().__init__()
        self.spatial_att=SpatialAttentionModule(in_channels)
        self.channels_att=ChannelAttentionModule(in_channels)
        self.scale=0.125
        self.conv1=nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=1, padding=0)
        self.bn1=nn.BatchNorm2d(in_channels)

    def forward(self,x):#
        temp_x=x
        batch,channel,height,width=x.size(0),x.size(1),x.size(2),x.size(3)
        spatial_att_x=self.spatial_att(x)
        channels_att_x=self.channels_att(x)

        q=spatial_att_x.permute(0,2,3,1).contiguous()#10,512,16,16->10,16,16,512
        q=q.view(batch,-1,channel).contiguous()#10,16,16,512->10,256,512
        k=channels_att_x.permute(0,2,3,1).contiguous()#10,512,16,16->10,16,16,512
        k=k.view(batch,-1,channel).contiguous()#10,16,16,512->10,256,512
        v=x.permute(0,2,3,1).contiguous()#10,512,16,16->10,16,16,512
        v=v.view(batch,-1,channel).contiguous()#10,16,16,512->10,256,512
        attn = (q * self.scale) @ k.transpose(-2, -1)#10,256,256
        attn = attn.softmax(dim=-1)#10,256,256

        x = (attn @ v).view(batch, height, width, -1).permute(0, 3,1,2)#10,256,256x10,256,512->10,512,16,16
        x=self.bn1(x+temp_x)
        x=self.conv1(x)
        return x




class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
class SS2Dv0(nn.Module):
    def __init__(
            self,
            d_model=768,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # ======================
            dropout=0.0,
            # ======================
            seq=False,
            force_fp32=True,
            **kwargs,
    ):

        if 'channel_first' not in list(kwargs.keys()):
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group =1 #4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)  # 96
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank  # 6

        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)  # 96,96X2
        self.act: nn.Module = act_layer()  # nn.SiLU
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,  # 96
            out_channels=d_inner,  # 96
            groups=d_inner,  # 96
            bias=conv_bias,  # True
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]  # 96,8
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )#k_group=4,

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)  # 96 96
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        x = x.permute(0, 2, 3, 1).contiguous()
        t = x
        x = x.view(-1, c).contiguous()
        x = self.in_proj(x)
        p_t = self.in_proj(t)
        x = x.view(b, h, w, c * 2).contiguous()
        judge = torch.equal(x, p_t)  # True
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend="mamba")

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        """ 四个不同的遍历路径 """
        # 堆叠输入张量 x 的两个视角（原始和转置）, [b, 2, d, l],6,2,96,4096
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        # 拼接 x_hwwh 和 其翻转
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  6,4,96,4096

        # 将 xs 通过权重矩阵 self.x_proj_weight 进行投影
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)  # 6,4,8,4096
        if hasattr(self, "x_proj_bias"):  # no run
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)

        """ x --投影-> delta, B, C 矩阵 """
        # 由投影后的x分别得到 delta, B, C 矩阵, '(B, L, D) -> (B, L, N)'
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)  # 6,4,8,4096->6,4,6,4096 / 6,4,1,4096 / 6,4,1,4096
        # 将 dts（delta） 通过权重矩阵 self.dt_projs_weight 进行投影, '(B, L, N) -> (B, L, D)'
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)  # 6,4,6,4096->6,4,96,4096

        xs = xs.view(B, -1, L)  # (b, k * d, l)  6,4,96,4096->6,384,4096
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l) 6,4,96,4096->6,384,4096
        Bs = Bs.contiguous()  # (b, k, d_state, l)  6,4,1,4096->6,4,1,4096
        Cs = Cs.contiguous()  # (b, k, d_state, l)  6,4,1,4096->6,4,1,4096

        As = -self.A_logs.float().exp()  # (k * d, d_state) 384,1 ->384,1
        Ds = self.Ds.float()  # (k * d)  384 ->384
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d) 4,96->384

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:  # run  xs 6,384,4096->6,384,4096, dts 6,384,4096->6,384,4096,
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)  # Bs 6,4,1,4096->6,4,1,4096, Cs 6,4,1,4096->6,4,1,4096

        if seq:  # no run
            out_y = []
            """ 选择性扫描 """
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:  # run
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)  # 在 selective_scan 函数中进行离散化操作  out_y=6,4,96,4096
        assert out_y.dtype == torch.float
        """ 四种遍历路径叠加 (Mamba之后) """
        # token位置还原
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)  # 6,4,96,4096->6,2,96,4096
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1,
                                                                                                L)  # 6,4,96,4096->6,96,4096
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1,
                                                                                                   L)  # 6,2,96,4096->6,96,4096
        # 四种状态叠加
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y  # 6,96,4096
        # 还原形状，方便输出
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C) 6,96,4096->6,4096,96
        y = self.out_norm(y).view(B, H, W, -1)  ##6,4096,96->6,64,64,96

        y = y * z  # 6,64,64,96*6,64,64,96->6,64,64,96
        out = self.dropout(self.out_proj(y))  # 6,64,64,96->6,64,64,96
        out = out.permute(0, 3, 1, 2).contiguous()  # 6,64,64,96->6,96,64,64
        return out  # 6,96,64,64
class VSSM_SS2D(nn.Module):
    def __init__(self, in_channels, expansion_factor, lambda_factor):
        super(VSSM_SS2D, self).__init__()

        self.lambda_factor = lambda_factor

        # Linear projection layers ϕ1, ϕ2
        self.ϕ1 = nn.Linear(in_channels, in_channels * self.lambda_factor)#in_channels=768
        self.ϕ2 = nn.Linear(in_channels, in_channels )

        # 1x1 Depth-wise Convolution (DWConv)
        self.conv1 = nn.Conv2d(in_channels*self.lambda_factor, in_channels, kernel_size=1, groups=in_channels, bias=False)

        # SiLU activation
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()

        #use thess2dv0
        self.norm1=LayerNorm2d(768)
        self.ss2dv0 = SS2Dv0(
                d_model=768, #hidden_dim
                d_state=1, #ssm_d_state
                ssm_ratio=1.0,#ssm_ratio
                dt_rank="auto",#ssm_dt_rank
                # ==========================
                dropout=0.0,#ssm_drop_rate
                # ==========================
                seq=False,
                # ==========================
                force_fp32=True,
                channel_first=True,
            )
        self.drop_path = DropPath(0.0)
        self.norm2 = LayerNorm2d(768)
        self.norm3 = LayerNorm2d(768)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False)
    def forward(self, x):#10,32,32,768(c)
        b,h,w,c=x.size(0), x.size(1), x.size(2), x.size(3)
        xN = F.layer_norm(x, x.size()[1:])#10,32,32,768->10,32,32,768
        N = h * w
        xN = xN.view( -1,c).contiguous()  # (batch_size*h*w, C) 10,32,32,768->10240,768
        x_expanded = self.ϕ1(xN)  # (batch_size*h*w, C)*(C,out_channels) 10240,768->10240,1536
        x_expanded=x_expanded.view(b,h,w,-1).contiguous() #10240,1536->10,32,32,1536
        x_expanded=x_expanded.permute(0,3,1,2).contiguous() #10,32,32,1536->10,1536,32,32
        h1 = self.conv1(x_expanded)  # Apply Depth-wise convolution 10,1536,32,32->10,768,32,32
        h1 = self.silu1(h1)  # SiLU activation  10,768,32,32->10,768,32,32

        # h1 = self.ss2dv0(h1)  # Apply 2D-selective scan module (SSM) 10,768,32,32->10,768,32,32
        h1 = h1 + self.drop_path(self.ss2dv0(self.norm1(h1)))  # 10,768,32,32->10,768,32,32 run 6,96,64,64->6,96,64,64
        h1 = self.norm2(h1)  # Layer Normalization 10,768,32,32->10,768,32,32

        h2 = self.ϕ2(xN)  # Apply linear projection ϕ2   10240,768->10240,768
        h2=h2.view(b,h,w,-1).contiguous() #10240,768->10,32,32,768
        h2=h2.permute(0,3,1,2).contiguous() #10,32,32,768->10,768,32,32
        h2 = self.silu2(h2)  # SiLU activation 10,768,32,32)
        hout = h1 * h2  # Hadamard product (element-wise multiplication)#10,768,32,32
        hout=self.norm3(hout)#10,768,32,32->10,768,32,32
        hout=self.conv2(hout)#10,768,32,32->10,768,32,32
        return hout



