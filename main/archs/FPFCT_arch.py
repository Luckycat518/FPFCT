import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs import LightSR_block as LightSR_block



@ARCH_REGISTRY.register()
class LightSR(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat,
                 upscale,
                 ):
        super(LightSR, self).__init__()


        nf = num_feat
        nb = 1


        self.edge_map=LightSR_block.DEA(nf,nb)

        self.upscale = upscale
        self.fea_conv = LightSR_block.conv_layer(num_in_ch, nf, kernel_size=3)
        # num_modules = 5
        num_modules = 4
        # num_modules = 3
        self.B1 = LightSR_block.FPF(in_channels=nf)
        self.B2 = LightSR_block.FPF(in_channels=nf)
        self.B3 = LightSR_block.FPF(in_channels=nf)
        self.B4 = LightSR_block.FPF(in_channels=nf)
        # self.B5 = LightSR_block.FEB(in_channels=nf)
        self.c = LightSR_block.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = LightSR_block.conv_layer(nf, nf, kernel_size=3)
        upsample_block = LightSR_block.pixelshuffle_block
        self.upsampler = upsample_block(nf, num_out_ch, upscale_factor=upscale)
        self.scale_idx = 0




    def forward(self, input):
        bi = F.interpolate(input, scale_factor=self.upscale, mode='bicubic', align_corners=False) #这里把FeNet的上采样加上去了

        out_fea = self.fea_conv(input)

        edge_map =self.edge_map(out_fea)

        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        # out_B5 = self.B5(out_B4)
        # out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4,out_B5], dim=1))  #concatenation 拼接

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea


        out_lr = out_lr + edge_map


        output = self.upsampler(out_lr)

        return output+bi

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx




