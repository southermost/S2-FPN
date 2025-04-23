import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter
#_--------------------------------------------


# 定义 SiLU 激活函数
class SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)

# 定义 Conv2d + BatchNorm + Activation 的模块
class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            SiLU()
        )

# 定义 Squeeze-and-Excitation 模块
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1)
        self.activation = SiLU()
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale

# 定义 MBConv 模块
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio, drop_prob=0.0):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.block = nn.Sequential(
            Conv2dNormActivation(in_channels, hidden_dim, kernel_size=1),
            Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, max(1, int(in_channels * se_ratio))),
            Conv2dNormActivation(hidden_dim, out_channels, kernel_size=1)
        )
        self.stochastic_depth = nn.Identity() if drop_prob == 0 else nn.Dropout(drop_prob)

    def forward(self, x):
        return self.stochastic_depth(self.block(x))

# 定义 EfficientNet-B0 主体
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 定义 EfficientNet-B0 特征层
        self.base_layers = nn.Sequential(
            Conv2dNormActivation(3, 32, kernel_size=3, stride=2, padding=1),  # 输入层
            MBConv(32, 16, expand_ratio=1, kernel_size=3, stride=1, se_ratio=0.25),  # Block 1
            MBConv(16, 24, expand_ratio=6, kernel_size=3, stride=2, se_ratio=0.25),  # Block 2
            MBConv(24, 40, expand_ratio=6, kernel_size=5, stride=2, se_ratio=0.25),  # Block 3
            MBConv(40, 80, expand_ratio=6, kernel_size=3, stride=2, se_ratio=0.25),  # Block 4
            MBConv(80, 112, expand_ratio=6, kernel_size=5, stride=1, se_ratio=0.25), # Block 5
            MBConv(112, 192, expand_ratio=6, kernel_size=5, stride=2, se_ratio=0.25),# Block 6
            MBConv(192, 320, expand_ratio=6, kernel_size=3, stride=1, se_ratio=0.25),# Block 7
            Conv2dNormActivation(320, 1280, kernel_size=1) # 输出层
        )

        # 按层分割特征层，便于提取
        self.layer_down0 = nn.Sequential(*self.base_layers[:2])  # 特征层1
        self.layer_down1 = nn.Sequential(*self.base_layers[2:3])  # 特征层2
        self.layer_down2 = nn.Sequential(*self.base_layers[3:4])  # 特征层3
        self.layer_down3 = nn.Sequential(*self.base_layers[4:6])  # 特征层4
        self.layer_down4 = nn.Sequential(*self.base_layers[6:9])  # 特征层5
    #
    #     # 定义分类层
    #     self.avgpool = nn.AdaptiveAvgPool2d(1)
    #     self.classifier = nn.Linear(1280, num_classes)
    #
    # def forward(self, x):
    #     # 分层提取特征
    #     x0 = self.layer_down0(x)
    #     x1 = self.layer_down1(x0)
    #     x2 = self.layer_down2(x1)
    #     x3 = self.layer_down3(x2)
    #     x4 = self.layer_down4(x3)
    #
    #     # 分类
    #     x = self.avgpool(x4).flatten(1)
    #     x = self.classifier(x)
    #     return x, [x0, x1, x2, x3, x4]  # 返回最终输出和分层特征

# # 示例用法
# model = EfficientNetB0(num_classes=1000)
# print(model)

# --------------------------------------------------------------


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class Attention(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()


class AttentionAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionAggregationModule, self).__init__()
        self.convblk = ConvBnRelu(in_chan, out_chan, ksize=1, stride=1, pad=0)
        self.conv_atten = Attention(out_chan)

    def forward(self, s5, s4, s3, s2):
        fcat = torch.cat([s5, s4, s3, s2], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_out = atten + feat
        return feat_out



class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)



class A2FPN(nn.Module):
    def __init__(
            self,
            class_num=6,
            encoder_channels=[1280, 112, 40, 24],  # EfficientNet 的通道数变化，1280 为最终输出通道
            pyramid_channels=64,
            segmentation_channels=64,
            dropout=0.2,
    ):
        super().__init__()
        self.name = 'A2FPNWithEfficientNet'

        # # EfficientNet B0 作为骨干网络
        # self.base_model = models.efficientnet_b0(pretrained=True)
        # print(self.base_model)
        # # print(self.base_model)
        # # self.base_model = efficientnet_b0(num_classes=class_num)
        #
        # # EfficientNet 提供 features 属性，包含提取的特征层
        # self.base_layers = self.base_model.features
        # self.layer_down0 = self.base_layers[0:2]  # 特征层1
        # self.layer_down1 = self.base_layers[2:3]  # 特征层2
        # self.layer_down2 = self.base_layers[3:4]  # 特征层3
        # self.layer_down3 = self.base_layers[4:6]  # 特征层4
        # self.layer_down4 = self.base_layers[6:9]  # 特征层5
        base_model = EfficientNetB0(num_classes=4)
        self.layer_down0 = base_model.layer_down0   # 特征层1
        self.layer_down1 = base_model.layer_down1  # 特征层2
        self.layer_down2 = base_model.layer_down2  # 特征层3
        self.layer_down3 = base_model.layer_down3  # 特征层4
        self.layer_down4 = base_model.layer_down4 # 特征层5
        # 修改 conv1 的输入通道数为 1280
        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        self.attention = AttentionAggregationModule(segmentation_channels * 4, segmentation_channels * 4)
        self.final_conv = nn.Conv2d(segmentation_channels * 4, class_num, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x):
        # 获取 EfficientNet 提取的特征层
        c1 = self.layer_down0(x)
        c2 = self.layer_down1(c1)
        c3 = self.layer_down2(c2)
        c4 = self.layer_down3(c3)
        c5 = self.layer_down4(c4)
        # print(f"c1: {c1.shape}, c2: {c2.shape}, c3: {c3.shape}, c4: {c4.shape}, c5: {c5.shape}")
        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        out = self.dropout(self.attention(s5, s4, s3, s2))
        out = self.final_conv(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        return out
if __name__ == "__main__":
    model = A2FPN(class_num=4).cuda()
    # torch.save(model, 'model.pth')
    torch.save(model.state_dict(), 'model.pth')
    # input = torch.rand(2, 3, 512, 512).cuda()
    input = torch.rand(2, 3, 256, 256).cuda()
    output = model(input)
    print(output.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    # 转换参数量为 MB
    total_params_mb = (total_params * 4) / (1024 * 1024)  # 4 字节每个浮点参数
    print(f'Total parameters in MB: {total_params_mb:.2f} MB')
