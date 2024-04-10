import torch
import torch.nn as nn
import functools
import numpy as np
from CBAM import CbamAttention


#######网络权重参数初始化#######
def weights_init(m):
    classname = m.__class__.__name__
    # 遍历网络中每个模块类名
    # 卷积层=权重高斯分布初始化，偏置设置为0
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    # 批标准化层=对权重进行正态分布初始化，偏置设置为0
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
##返回一个指定规范化类型及其参数的规范化层
def get_norm_layer(norm_type='instance'):
    # batch=批标准化层
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    # instance=实例标准化层
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

########接收参数并返回所需生成器#######
def define_G(input_nc,output_nc,ngf=64,norm="batch",use_dropout=False):
    netG=None
    use_gpu=torch.cuda.is_available()
    norm_layer=get_norm_layer(norm_type=norm)

    netG=ResnetGenerator(input_nc, output_nc,ngf=64,norm_layer=norm_layer, 
                        use_dropout=use_dropout, n_blocks=9)

    netG.apply(weights_init)
    return netG


#######接收参数并返回所需判别器#######
def define_D(input_nc, ndf=64, n_layers_D=3, norm='batch', use_sigmoid=False):
    netD = None
    use_gpu = torch.cuda.is_available()
    norm_layer = get_norm_layer(norm_type=norm)



    netD = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D,
    	                  norm_layer=norm_layer, use_sigmoid=use_sigmoid)

    netD.apply(weights_init)
    return netD


class ResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        # 判断规范化层norm_layer是否为实例标准化(IN)类型
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # 定义了首层卷积和归一化操作，包括反射填充层、卷积层*1、归一化层、ReLU激活函数
        model = [
            nn.ReflectionPad2d(3),  # 上下左右填充3像素
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]


        # 下采样*2 将输入图像的尺寸缩小为原来的一半
        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),

        ]

        # 残差网络 若干ResNet块组成
        # 每个ResNet块包括了一个跳跃连接和仿射变换、归一化和ReLU激活等层
        for i in range(n_blocks):
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            ]
        # 上采样操作*2 第一个上采样操作后，引入了CBAM注意力机制
        model += [
            # CBAM
            CbamAttention(in_channels=256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
        ]

        # 输出层 对上一层的输出进行卷积、反射填充和Tanh()激活函数处理
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    # 前向传播
    def forward(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.to(device)

        output = self.model(input)

        return output

#######ResNet残差块网络#######
class ResnetBlock(nn.Module):

	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()
        # 存储了三种常用的填充加卷积操作组合
        # 根据 padding_type 的选择，获取对应的padding函数和卷积层来构建卷积块
		padAndConv = {
			'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
			'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
			'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		}

		try:
			blocks = padAndConv[padding_type] + [
				norm_layer(dim),
				nn.ReLU(True)
            ] + [
				nn.Dropout(0.5)
			] if use_dropout else [] + padAndConv[padding_type] + [
				norm_layer(dim)
			]
		except:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		self.conv_block = nn.Sequential(*blocks)


	def forward(self, x):
		out = x + self.conv_block(x)
		return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        # 判断规范化层norm_layer是否为实例标准化(IN)类型
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4 # 卷积核宽度
        padw = int(np.ceil((kw - 1) / 2)) # 填充像素
        #卷积 全局判别
        sequence = [
            # 第一层卷积和LeakyReLU激活函数
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        #局部判别
        for n in range(1, n_layers): # 逐层
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8) # 计算特征个数（第一层nf_mult=1）
            sequence += [
                # 卷积操作、归一化层和LeakyReLU激活函数
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            # 输出层卷积和Sigmoid激活函数（use_sigmoid=True时）
            # 或无激活函数（use_sigmoid=False时）
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        return self.model(input)


