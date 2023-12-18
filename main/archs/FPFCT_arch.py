import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
# from basicsr.archs import block as B, block  #block里面写的函数，调用的时候需要B.或者block.
from basicsr.archs import FPFCT_block as FPFCT_block
# from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer


@ARCH_REGISTRY.register()
class FPFCT(nn.Module):
    #主网络
    def __init__(self,
                 num_in_ch, #输入的通道数为3。
                 num_out_ch, #输出的通道数为3。
                 num_feat, #中间特征的通道数50。
                 upscale,
                 ): #图像平均值以RGB顺序表示。
        super(FPFCT, self).__init__()


        nf = num_feat #中间特征的通道数。
        nb = 1 #DEA块的数量

        ###############边缘细节提取部分####################
        self.edge_map=FPFCT_block.DAB(nf,nb)  #边缘提取部分

        self.upscale = upscale
        self.fea_conv = FPFCT_block.conv_layer(num_in_ch, nf, kernel_size=3) #输入通道为RGB3通道 输出特征为50通道中间特征通道数 3×3的卷积
        num_modules = 4  # 模块数量
        self.B1 = FPFCT_block.FPF(in_channels=nf) #特征提取块
        self.B2 = FPFCT_block.FPF(in_channels=nf) #特征提取块
        self.B3 = FPFCT_block.FPF(in_channels=nf) #特征提取块
        self.B4 = FPFCT_block.FPF(in_channels=nf) #特征提取块
        self.c = FPFCT_block.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu') #激活函数
        self.LR_conv = FPFCT_block.conv_layer(nf, nf, kernel_size=3) #Conv2d(50, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        upsample_block = FPFCT_block.pixelshuffle_block #双三次上采样模块
        self.upsampler = upsample_block(nf, num_out_ch, upscale_factor=upscale)  #上采样模块
        self.scale_idx = 0




    def forward(self, input):
        bi = F.interpolate(input, scale_factor=self.upscale, mode='bicubic', align_corners=False) #这里把FeNet的上采样加上去了

        out_fea = self.fea_conv(input) #3×3的卷积     提取浅层特征

        edge_map =self.edge_map(out_fea) #通过边缘提取网路得到边缘细节图

        out_B1 = self.B1(out_fea) #特征提取块
        out_B2 = self.B2(out_B1) #特征提取块
        out_B3 = self.B3(out_B2) #特征提取块
        out_B4 = self.B4(out_B3) #特征提取块

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        ########加上边缘细节部分#############
        out_lr = out_lr + edge_map


        output = self.upsampler(out_lr)
        return output+bi

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

if __name__ == '__main__':
    x = torch.randn((16,3,64,64))
    model = FPFCT(num_in_ch=3, num_out_ch=3,num_feat=50,upscale=2)
    out = model(x)
    print(out.shape)







