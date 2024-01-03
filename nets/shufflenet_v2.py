from typing import List, Callable
import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    '1x': "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
}


# 通道重排
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()  # 分离通道信息
    channels_per_group = num_channels // groups  # 划分组
    x = x.view(batch_size, groups, channels_per_group, height, width)  # 变为矩阵的形式
    x = torch.transpose(x, 1, 2).contiguous()  # 矩阵转置
    x = x.view(batch_size, -1, height, width)  # 还原shuffle后通道信息，flatten

    return x


# bottleneck
class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:  # stride 只能为1或者2
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0  # channel split，这里是对半分
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:  # 下采样模块，左边branch
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),  # 维度要降一半，为了和右边concat
                nn.ReLU(inplace=True)
            )
        else:  # 正常模块，左边channel split直接过去
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(  # 无论哪个模块，右边的branch是类似的
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod  # depthwise 卷积，不是组卷积！！，MobileNet V1深度可分离卷积的dw卷积
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)  # channel split
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


# shufflenet v2
class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3  # 图像维度是3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(  # 最初的Conv1
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # MaxPool

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


    def freeze_backbone(self):
        for name, para in self.named_parameters():
            if "fc" not in name:  # 冻结除了fc层的参数
                para.requires_grad = False

    def Unfreeze_backbone(self):
        for name, para in self.named_parameters():
            if "fc" not in name:  # 冻结除了fc层的参数
                para.requires_grad = True

# 定义不同的shuffleNet网络，x1_0 是宽度因子 α 扩大1倍
def shufflenet_v2_x0_5(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_0(num_classes=1000,pretrained=False,progress=True):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         #num_classes=num_classes
                         )

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['1x'], model_dir='./model_data',
                                            progress=progress)
        model.load_state_dict(state_dict)

    if num_classes!=1000:
        model.fc = nn.Linear(1024, num_classes)

    return model


def shufflenet_v2_x1_5(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024],
                         num_classes=num_classes
                        )

    return model


def shufflenet_v2_x2_0(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 244, 488, 976, 2048],
                         num_classes=num_classes)

    return model

if __name__ == '__main__':
    inputs = torch.randn(10, 3, 224, 224)
    model = shufflenet_v2_x1_0(num_classes=2,pretrained=True)  # k为分类数
    print(model)
    print("输入维度:", inputs.shape)
    outputs = model(inputs)
    print("输出维度:", outputs.shape)