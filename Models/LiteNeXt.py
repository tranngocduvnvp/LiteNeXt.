import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Utils_module import SMmoduleplusplus, SMmodulesimple, Layernorm, DownStagePluss,\
      UpDownstream, AttentionDC, SimpleAttentionDC, RB

class LiteNeXt(nn.Module):
    def __init__(self, in_channel = 16, conf_layer_encoder = [1,1,1,1], kernel_sizes=[3,3,3,3], dilate = [1, 1, 1, 1]):
        super(LiteNeXt, self).__init__()
        self.patches_parition = nn.Sequential(nn.Conv2d(3,in_channel, 2, 2), Layernorm(in_channel), nn.GELU()) #H/4xW/4xC
        self.stage1 = nn.Sequential(*[SMmoduleplusplus(in_channel, kernel_size=kernel_sizes[0], dilate = dilate[0]) for i in range(conf_layer_encoder[0])]) #H/4xW/4xC
        self.stage2 = nn.Sequential(*([DownStagePluss(in_channel, in_channel*2)] + [SMmoduleplusplus(in_channel*2, kernel_size=kernel_sizes[1], dilate = dilate[1]) for i in range(conf_layer_encoder[1])])) #H/8xW/8x2C
        self.stage3 = nn.Sequential(*([DownStagePluss(in_channel*2, in_channel*4)] + [SMmoduleplusplus(in_channel*4, kernel_size=kernel_sizes[2], dilate = dilate[2]) for i in range(conf_layer_encoder[2])])) #H/16xW/16x4C
        self.stage4 = nn.Sequential(*([DownStagePluss(in_channel*4, in_channel*8)] + [SMmoduleplusplus(in_channel*8, kernel_size=kernel_sizes[3], dilate = dilate[3]) for i in range(conf_layer_encoder[3])])) #H/32xW/32x8C
        self.bottle_neck = nn.Sequential(RB(in_channel*8, in_channel*8))
        self.rb4 = UpDownstream(2, 16*in_channel, in_channel)
        self.head = nn.Conv2d(in_channel, 1, 1)
        self.attention = AttentionDC([1, 2, 4, 8], [in_channel,in_channel*2, in_channel*4, in_channel*8], in_channel*8)

    def forward(self, x):
        x = self.patches_parition(x) #H/4xC
        e1 = self.stage1(x) #H/4xC
        e2 = self.stage2(e1) #H/8x2C
        e3 = self.stage3(e2) #H/16x4C
        e4 = self.stage4(e3) #H/32x8C
        #--------------------------------#
        d4 = self.bottle_neck(e4)
        d3 = self.attention([e1, e2, e3, e4, d4])
        d1 = self.rb4(d3) # HxC
        y = self.head(d1) #Hx1
        return y
    

if __name__ == "__main__":
    litenext = LiteNeXt()
    x = torch.rand(2,3,256,256)
    out = litenext(x)
    print(out.shape)