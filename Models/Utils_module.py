import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dropblock import DropBlock2D
class EFFat(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) + self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + x


class convMixerLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            groups=in_channel,
            kernel_size=kernel_size,
            stride=1,
            padding="same"
        )
        self.eff = EFFat(out_channel)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.activation = nn.SiLU()
        self.batchnorm1 = nn.BatchNorm2d(in_channel)
        self.batchnorm2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        ori = x
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.eff(self.batchnorm1(x)) + ori
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.batchnorm2(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class Layernorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Channel_attention(nn.Module):
    def __init__(self, c, reduction=16) -> None:
        super().__init__()
        self.fcap1 = nn.Conv2d(c, c//reduction,1)
        self.fcmp1 = nn.Conv2d(c, c//reduction, 1)
        self.fcap2 = nn.Conv2d(c//reduction, c, 1)
        self.fcmp2 = nn.Conv2d(c//reduction, c, 1)
        self.acg = nn.Conv2d(c//reduction, c, 1)
    def forward(self, x):
        f1 = F.relu(self.fcap1(x))
        f2 = F.relu(self.fcmp1(x))
        f = self.acg(f1 + f2)
        out = F.sigmoid(f + self.fcap2(f1) + self.fcmp2(f2))*x
        return out

class CTblock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3) -> None:
        super().__init__()
        self.mixspatial = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, padding="same", groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            Channel_attention(in_channel),
            nn.Conv2d(in_channel, out_channel, 3, padding="same"),
            nn.BatchNorm2d(out_channel),
        )
        self.mixchannel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        y = F.relu(self.mixspatial(x) + self.mixchannel(x))
        return y
class UpDownstream(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        self.scale = scale
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = Layernorm(out_channel)
        self.ac = nn.GELU()
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.ac(x)
        bn, c, h, w = x.shape
        x = F.interpolate(x, size=(int(h*self.scale), int(w*self.scale)), mode="bilinear")
        return x

class NormMode(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        """nhận đầu vào là một tensor cxhxw

        Returns:
            - vector key: d
            - Tensor value: c1xh1xw1
        """
        self.norm = UpDownstream(scale, in_channel, out_channel)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.mg = nn.AdaptiveMaxPool2d((1,1))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.GELU(),
            nn.Conv2d(in_channel, out_channel, 1)
        )

    def forward(self, x):
        v = self.norm(x).unsqueeze(1) # (bs, 1, c, h, w)
        k = self.mlp(self.avg(x)+self.mg(x)).view(x.shape[0], 1, -1) #(bs,1, c)
        return v, k


class AttentionDC(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        self.normfm1 = NormMode(scale[0], in_channel[0], out_channel)
        self.normfm2 = NormMode(scale[1], in_channel[1], out_channel)
        self.normfm3 = NormMode(scale[2], in_channel[2], out_channel)
        self.normfm4 = NormMode(scale[3], in_channel[3], out_channel)
        self.normfmDecode = NormMode(scale[3], out_channel, out_channel)
        self.mlp = nn.Linear(out_channel*2, out_channel)


    def forward(self, feature_maps):
        fm1, fm2, fm3, fm4, fmdecode = feature_maps
        v1, k1 = self.normfm1(fm1)
        v2, k2 = self.normfm2(fm2)
        v3, k3 = self.normfm3(fm3)
        v4, k4 = self.normfm4(fm4)
        vd, qd = self.normfmDecode(fmdecode) #(bs, 1, c)
        K = torch.cat([k1, k2, k3, k4], dim=1) #(bs, 4, c)
        K = torch.cat([K, qd.expand_as(K)], dim=2) #(bs, 4, 2c)
        atten = F.softmax(self.mlp(K), dim=1).unsqueeze(-1).unsqueeze(-1)   #(bs, 4, c, 1, 1)
        V = torch.cat([v1,v2,v3,v4], dim=1) #(bs, 4, c, h, w)
        V = V*atten #(bs, 4, c, h, w)
        V = torch.sum(V, dim=1) #(bs, c, h, w)
#         print(V.shape, vd.squeeze(1).shape)
        V = torch.cat([V, vd.squeeze(1)], dim=1)
        return V
    
class SimpleAttentionDC(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        self.normfm1 = UpDownstream(scale[0], in_channel[0], out_channel)
        self.normfm2 = UpDownstream(scale[1], in_channel[1], out_channel)
        self.normfm3 = UpDownstream(scale[2], in_channel[2], out_channel)
        self.normfm4 = UpDownstream(scale[3], in_channel[3], out_channel)

        self.axial_conv = nn.Sequential(
            nn.Conv2d(out_channel*4, out_channel*4, (7,1), dilation=(3,1), padding="same"),
            nn.Conv2d(out_channel*4, out_channel*4, (1,7), dilation=(1,3), padding="same"),
            Layernorm(out_channel*4),
            nn.GELU(),
        )


    def forward(self, feature_maps):
        fm1, fm2, fm3, fm4 = feature_maps
        v1 = self.normfm1(fm1)
        v2 = self.normfm2(fm2)
        v3 = self.normfm3(fm3)
        v4 = self.normfm4(fm4)
        V = torch.cat([v1,v2,v3,v4], dim=1) #(bs, c, h, w)
        V = self.axial_conv(V) #(bs, c, h, w)
        
        return V

class Layernorm(nn.Module):
    def __init__(self, in_channels):
        super(Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(in_channels)
    def forward(self, x):
        bn, c, h, w = x.shape
        x_view = x.view(bn, c, -1).transpose(1, 2)
        x = self.layernorm(x_view).transpose(2, 1)
        x = x.view(bn, c, h, w)
        return x


class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding="same"),
            Layernorm(out_channels),
            nn.GELU(),
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1, padding="same"),
            Layernorm(out_channels),
            nn.GELU(),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class MixSpatial(nn.Module):
    def __init__(self, in_channel, p=0.1):
        super(MixSpatial, self).__init__()
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 7, dilation=5, padding="same", groups=in_channel),
            nn.Dropout2d(p),
            nn.BatchNorm2d(in_channel)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 5, dilation=3, padding="same", groups=in_channel),
            nn.Dropout2d(p),
            nn.BatchNorm2d(in_channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, dilation=1, padding="same", groups=in_channel),
            nn.Dropout2d(p),
            nn.BatchNorm2d(in_channel)
        )
        self.gelu = nn.GELU()
    def forward(self, x):
        y = self.conv7x7(x) + self.conv5x5(x) + self.conv3x3(x)
        return x + self.gelu(y)

class MixChannel(nn.Module):
    def __init__(self, in_channel, expand_factor=2, reduction = 16):
        super(MixChannel, self).__init__()
        expand_out_channel = in_channel*expand_factor
        self.amplify_channel = nn.Sequential(nn.Conv2d(in_channel, expand_out_channel, 1),
                                    nn.BatchNorm2d(expand_out_channel),
                                    nn.GELU())
        self.se = SEModule(expand_out_channel, reduction=reduction)
        self.mixing_channel = nn.Sequential(nn.Conv2d(expand_out_channel, in_channel, 1),
                                        nn.BatchNorm2d(in_channel),
                                        nn.GELU())
    def forward(self, x):
        y = self.amplify_channel(x)
        y = self.se(y)
        y = self.mixing_channel(y)
        return y + x

class DownStage(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownStage, self).__init__()
        self.patches_merge = nn.Sequential(nn.Conv2d(in_channel, out_channel, 2, 2), Layernorm(out_channel), nn.GELU())
    def forward(self, x):
        return self.patches_merge(x)

class DownStagePluss(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownStagePluss, self).__init__()
        self.patches_merge = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(in_channel, out_channel, 1, 1), Layernorm(out_channel), nn.GELU())
    def forward(self, x):
        return self.patches_merge(x)


class MixModule(nn.Module):
    def __init__(self, in_channel, expand_factor=2):
        super().__init__()
        self.mixspa = MixSpatial(in_channel)
        self.mixchan = MixChannel(in_channel, expand_factor)

    def forward(self, x):
        x = self.mixspa(x)
        x = self.mixchan(x)
        return x

class Gap(nn.Module):
    def __init__(self, in_channel):
        super(Gap, self).__init__()
        self.conv1x1x1 = nn.Conv2d(in_channel, in_channel//2, 1)
        self.conv1x1x2 = nn.Conv2d(in_channel, in_channel//2, 1)
    def forward(self, x):
        x1 = self.conv1x1x1(x)
        x2 = self.conv1x1x2(x)
        return x1*x2


class Spatial_pool_mix(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(Spatial_pool_mix, self).__init__()
        self.layer_num1 = Layernorm(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels*2, 1)
        self.avpool = nn.AvgPool2d(kernel_size, padding=kernel_size//2, stride=1)
        self.gap = Gap(in_channels*2)
    def forward(self, x):
        ori = x
        x = self.layer_num1(x)
        x = self.conv1x1(x)
        x = F.gelu(x)
        x = self.avpool(x)
        x = F.gelu(x)
        x = self.gap(x)
        return x+ori


class PoolModule(nn.Module):
    def __init__(self, in_channles):
        super(PoolModule, self).__init__()
        self.spatial_mix = Spatial_pool_mix(in_channles)
        self.channel_mix = MixChannel(in_channles)
    def forward(self, x):
        x = self.spatial_mix(x)
        x = self.channel_mix(x)
        return x

class MLKA_Ablation(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2*n_feats

        self.n_feats= n_feats
        self.i_feats = i_feats

        self.norm = Layernorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        k = 2

        #Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 7, 1, 7//2, groups= n_feats//k),
            nn.Conv2d(n_feats//k, n_feats//k, 9, stride=1, padding=(9//2)*4, groups=n_feats//k, dilation=4),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 5, 1, 5//2, groups= n_feats//k),
            nn.Conv2d(n_feats//k, n_feats//k, 7, stride=1, padding=(7//2)*3, groups=n_feats//k, dilation=3),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))
        '''self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k),
            nn.Conv2d(n_feats//k, n_feats//k, 5, stride=1, padding=(5//2)*2, groups=n_feats//k, dilation=2),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))'''

        #self.X3 = nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k)
        self.X5 = nn.Conv2d(n_feats//k, n_feats//k, 5, 1, 5//2, groups= n_feats//k)
        self.X7 = nn.Conv2d(n_feats//k, n_feats//k, 7, 1, 7//2, groups= n_feats//k)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))


    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        #u_1, u_2, u_3= torch.chunk(u, 3, dim=1)
        a_1, a_2 = torch.chunk(a, 2, dim=1)

        a = torch.cat([self.LKA7(a_1)*self.X7(a_1), self.LKA5(a_2)*self.X5(a_2)], dim=1)

        x = self.proj_last(x*a)*self.scale + shortcut

        return x

class SMmodule(nn.Module):
    def __init__(self, in_channel, expand = 2, kernel_size=7, kernel_local=3, drop_rate=0.1, dilate =1):
        super(SMmodule, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size, padding="same", groups=in_channel, dilation=dilate)
        self.conv1x1 = nn.Conv2d(in_channel, in_channel*expand, 1)
        self.conv_local = nn.Conv2d(in_channel*expand, in_channel, kernel_local, padding="same")
        self.drp = DropBlock2D(block_size=3, drop_prob=drop_rate)
        self.ln1 = Layernorm(in_channel)
        self.ln2 = Layernorm(in_channel*expand)
    def forward(self, x):
        ori = x
        x = self.conv(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.drp(x)
        x = self.conv1x1(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.conv_local(x)
        return x+ori

class SMmoduleplus(nn.Module):
    def __init__(self, in_channel, expand = 2, kernel_size=7, kernel_local=3, drop_rate=0.1, dilate =1):
        super(SMmoduleplus, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size, padding="same", groups=in_channel, dilation=dilate)
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, 1)
        self.conv_local = nn.Conv2d(in_channel, in_channel, kernel_local, padding="same")
        self.drp = DropBlock2D(block_size=3, drop_prob=drop_rate)
        self.ln1 = Layernorm(in_channel)
        self.ln2 = Layernorm(in_channel)
        self.ln3 = Layernorm(in_channel)
    def forward(self, x):
        ori = x
        x = self.conv_local(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = x + ori
        ori = x
        x = self.conv(x)
        x = self.ln3(x)
        x = F.gelu(x)
        x = x + ori
        x = self.conv1x1(x)
        x = self.ln2(x)
        x = F.gelu(x)
        return x

class SMmoduleplusplus(nn.Module):
    def __init__(self, in_channel, kernel_size=7, kernel_local=3, drop_rate=0.1, dilate =1):
        super(SMmoduleplusplus, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size, padding="same", groups=in_channel, dilation=dilate)
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, 1)
        self.conv_local = nn.Conv2d(in_channel, in_channel, kernel_local, groups=1, padding="same")
        self.drp = DropBlock2D(block_size=3, drop_prob=drop_rate)
        self.ln1 = Layernorm(in_channel)
        self.ln2 = Layernorm(in_channel)
        self.ln3 = Layernorm(in_channel)
    def forward(self, x):
        oriin = x
        x = self.conv_local(x)
        x = self.ln1(x)
        x = F.gelu(x)
        ori = x
        x = self.conv(x)
        x = self.ln3(x)
        x = F.gelu(x)
        x = x + ori + oriin
        # x = torch.cat([oriin, x], dim=1)
        x = self.conv1x1(x)
        x = self.ln2(x)
        x = F.gelu(x)
        return x

class SMmodulesimple(nn.Module):
    def __init__(self, in_channel, kernel_size=3, drop_rate=0.01, dilate =1):
        super(SMmodulesimple, self).__init__()
        self.conv_local = nn.Conv2d(in_channel, in_channel, kernel_size, groups=1, dilation=dilate, padding="same")
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, 1)
        self.drp = DropBlock2D(block_size=3, drop_prob=drop_rate)
        self.ln1 = Layernorm(in_channel)
        self.ln2 = Layernorm(in_channel)
    def forward(self, x):
        ori = x
        x = self.conv_local(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.conv1x1(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = x + ori
        return x
    
    

if __name__ == "__main__":
    sm = SMmoduleplusplus(32)
    x = torch.rand(2, 32, 32, 32)
    out = sm(x)
    print(out.shape)