import torch.nn as nn
import torch.nn.functional as F #函数，由def function ()定义，是一个固定的运算公式
# from basicsr.archs import SGBlock,FNet,Spartial_Attention,SwinT
from basicsr.archs import SwinT
import functools
# from basicsr.archs import block as B
import torch



#定义卷积层 输入通道 输出通道 卷积核尺寸 步长 扩展 组
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation #计算padding
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    return nn.Sequential(#400epoch 32.726 28.623
        nn.Conv2d(in_channels, int(in_channels * 0.5), 1, stride, bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5 * 0.5), 1, 1, bias=True),
        nn.Conv2d(int(in_channels * 0.5 * 0.5), int(in_channels * 0.5), (1, 3), 1, (0, 1),
                           bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5), (3, 1), 1, (1, 0), bias=True),
        nn.Conv2d(int(in_channels * 0.5), out_channels, 1, 1, bias=True)
    )


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

#定义激活函数 类型
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

#融合特征模块
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)  #Conv2d(40, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv_f = conv(f, f, kernel_size=1) #Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv_max = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0) #Conv2d(12, 12, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_ = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = conv(f, n_feats, kernel_size=1) #Conv2d(12, 40, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid() #Sigmoid()
        self.relu = nn.ReLU(inplace=True) #ReLU(inplace=True)

    def forward(self, x): #输入特性x
        c1_ = (self.conv1(x)) #x通过1×1卷积提取特征
        c1 = self.conv2(c1_) #Conv(S=2) 步长为2的卷积层
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3) #最大池化
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) #双线性插值
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf) #逐元素相加
        m = self.sigmoid(c4) #Sigmoid激活函数

        return x * m #返回x乘m

#中间级联的块（模型中的子块）
class FPF(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(FPF, self).__init__()
        self.rc = self.remaining_channels = in_channels  #剩余通道=输入通道
        # self.c = B.conv_block(in_channels, in_channels, kernel_size=1, act_type='lrelu')  # 激活函数
        self.lfe = LFE(num_feat=50, compress_ratio=3, squeeze_factor=30)  # LFE 局部特征提取
        self.swinT = SwinT.SwinT() #深度为2的Transformer层
        self.c1_r = conv_layer(in_channels, self.rc, 3)  # 输入通道数 剩余通道数 卷积核为3
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        shortcut = input ###加上残差结构
        # local context 局部特征提取
        LFE = self.lfe(input) #局部特征提取
        input = self.swinT(input) + LFE  #通过Swin Transformer Layer 深度为2
        input = input + shortcut #相加之后的结果
        out_fused = self.esa(self.c1_r(input)) #融合模块
        out_fused = self.c1_r(out_fused)  # 融合模块
        return out_fused


#双三次上采样
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride) #Conv2d(50, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    pixel_shuffle = nn.PixelShuffle(upscale_factor) #PixelShuffle(upscale_factor=4)
    return sequential(conv, pixel_shuffle)


# 局部特征提取
class ECAM(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ECAM, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  #全局平均池化
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),  #卷积1×1
            nn.ReLU(inplace=True),  #激活函数
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),  #卷积1×1
            nn.Sigmoid()) #激活函数
    def forward(self, x):
        y = self.attention(x)
        return x * y  #通道注意力作用到特征图上  变形版注意力

# 局部特征提取块
class LFE(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(LFE, self).__init__()

        self.lfe = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 1),
            nn.GELU(),  #激活函数
            nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 1),
            ECAM(num_feat, squeeze_factor) #增强注意力块
            )
    def forward(self, x):
        return self.lfe(x)

##########动态获取边缘细节####################
class DAB(nn.Module):
    def __init__(self,nf,nb):
        super(DEA, self).__init__()
        # nf = 50 #中间特征图通道数
        # nb = 4 #块的数量

        # DAB
        DAB_block_f = functools.partial(DAB, nf=nf)

        # 第一个卷积层---提取图像的浅层特征
        self.conv_first = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # 中间的DAB块
        self.DAB_trunk = make_layer(DAB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) #3×3卷积

        self.conv_last = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):

        fea = self.conv_first(x) #第一层提取浅层特征的卷积层
        trunk = self.trunk_conv(self.DAB_trunk(fea)) #主干（DAB块）
        fea = fea - trunk #浅层特征减去通过主干之后的特征
        out = self.conv_last(fea) #最后一个卷积层
        return out

##########动态获取边缘细节####################

#定义层的数量和块的数量
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

#像素级别的注意力
class PA(nn.Module):
    #  PA表示像素注意力
    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1) #1×1卷积
        self.sigmoid = nn.Sigmoid() #激活

    def forward(self, x):
        y = self.conv(x) #1×1卷积
        y = self.sigmoid(y) #激活
        out = torch.mul(x, y) #对两个张量进行逐元素乘法
        return out


# 注意力分支
class AttentionBranch(nn.Module):

    def __init__(self, nf, k_size=3):
        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3卷积
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True) #激活
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 卷积 nf->nf
        self.sigmoid = nn.Sigmoid() #激活
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 卷积
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 卷积

    def forward(self, x):
        y = self.k1(x) #3×3卷积
        y = self.lrelu(y) #ReLU激活
        y = self.k2(y) #1×1卷积进行通道变维
        y = self.sigmoid(y) #激活

        out = torch.mul(self.k3(x), y) #对两个张量进行逐元素乘法（输入x进行3×3卷积的结果和y进行逐元素乘法
        out = self.k4(out) #两者逐元素相乘的结果在通过3×3卷积

        return out

#DAB 模块
class DAB(nn.Module):

    def __init__(self, nf, reduction=4, K=2, t=30):
        super(DAB, self).__init__()
        self.t = t # t的值为30
        self.K = K # K的值为2

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False) #1×1卷积
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False) #1×1卷积
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True) #激活

        self.avg_pool = nn.AdaptiveAvgPool2d(1) #平均池化

        # Attention Dropout Module 注意力下降模块（动态调整过程）
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),  #线性层   输入：中间特征通道数   输出：中间特征通道数÷4
            nn.ReLU(inplace=True), #激活
            nn.Linear(nf // reduction, self.K, bias=False), #线性层   输入：中间特征通道数÷4  输出：K=2
        )

        # 注意力分支
        self.attention = AttentionBranch(nf)
        # 非注意力分支
        self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)


    def forward(self, x):
        residual = x #x用于后面的残差连接
        a, b, c, d = x.shape #输入的维度

        x = self.conv_first(x) #首先进行1×1卷积进行通道数变维
        x = self.lrelu(x) #激活

        # 动态调节机制--生成一组向量
        y = self.avg_pool(x).view(a, b) #对输入的特征进行平均池化
        y = self.ADM(y) #进行生成动态调节权值-----输出通道为2
        ax = F.softmax(y / self.t, dim=1)  #归一化操作（y÷t）---得到两个权值 ax[ : ]

        attention = self.attention(x) #输入特征x进行注意力过程
        non_attention = self.non_attention(x) #输入特征x进行非注意力过程

        # 注意力过程×ax[1] + 非注意力过程×ax[0]
        x = attention * ax[:, 0].view(a, 1, 1, 1) + non_attention * ax[:, 1].view(a, 1, 1, 1)
        x = self.lrelu(x) #激活

        out = self.conv_last(x) #1×1卷积进行变维、
        out += residual #和原始特征x进行残差连接

        return out


if __name__ == '__main__':
    x = torch.randn((32,50,64,64))
    model = DEA(nf=50,nb=4)
    out = model(x)
    print(out.shape)

