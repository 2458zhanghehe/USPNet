# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, trunc_normal_
from functools import partial
import torch.nn.functional as F

class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, num_groups=1):
        """ We use GroupNorm (group = 1) to approximate LayerNorm
        for [N, C, H, W] layout"""
        super(GroupNorm, self).__init__(num_groups, num_channels)

class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2 * dim, (2, 2), stride=2, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int, tuple): Image size.
        patch_size (int, tuple): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
    

class SCMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, norm_layer=nn.BatchNorm2d):
        super(SCMLP, self).__init__()
        if hidden_channels is None:
            hidden_channels = in_channels * 4
        if out_channels is None:
            out_channels = in_channels

        self.channel_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),  # 逐点卷积
            nn.GELU()
        )

        self.spatial_processing = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),  # 深度可分离卷积
            norm_layer(hidden_channels),  # 批量归一化
            nn.GELU()
        )

        self.channel_projection_out = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)  # 逐点卷积
        )


    def forward(self, x):
        residual = x
        x = self.channel_projection(x)
        x = self.spatial_processing(x)
        x = self.channel_projection_out(x)
        return x



class ORIConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ORIConv, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 定义最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # 定义激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一层卷积
        conv1_out = self.conv1(x)

        maxpool_out = self.maxpool(x)
        conv2_out = self.conv2(maxpool_out)
        upsample_out = self.upsample(conv2_out)

        # 确保上采样后的特征图尺寸与原始输入一致
        if upsample_out.shape[2:] != x.shape[2:]:
            upsample_out = F.interpolate(upsample_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        # 逐元素相加
        add_out = torch.add(upsample_out, x)
        sigmod_out = self.sigmoid(add_out)

        # 逐元素相乘
        mul_out = torch.mul(conv1_out, sigmod_out)
        output = self.conv3(mul_out)

        return output


class SORS(nn.Module):
    def __init__(self, in_channels, out_channels, n_div=12, mlp_ratio = 4.0, input_resolution=None):
        super(SORS, self).__init__()

        self.dim = in_channels
        self.n_div = n_div
        self.hidden_dim = int(in_channels * mlp_ratio)
        self.input_resolution = input_resolution

        #self.ln = nn.LayerNorm(in_channels)
        self.ln = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.scMLP = SCMLP(in_channels,self.hidden_dim, in_channels)
        self.oriconv = ORIConv(in_channels, in_channels)
        self.conv = nn.Conv2d(2 * in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.shift_feat(x, self.n_div)
        ln = self.ln(x)
        x1 = self.scMLP(ln)
        x2 = self.oriconv(ln)
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        outPut = torch.add(x, out)
        return outPut

    def extra_repr(self) -> str:
        return f"dim={self.dim}," \
               f"input_resolution={self.input_resolution}," \
               f"shift percentage={4.0 / self.n_div * 100}%."

    @staticmethod
    def shift_feat(x, n_div):
        B, C, H, W = x.shape
        g = C // n_div
        out = torch.zeros_like(x)

        out[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # shift left
        out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
        out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
        out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down

        out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
        return out



class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 n_div=12,
                 mlp_ratio=4.,
                 norm_layer=None,
                 downsample=None,
                 use_checkpoint=False):

        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SORS(in_channels=dim,
                 out_channels=dim,
                 n_div=n_div,
                 mlp_ratio=mlp_ratio,
                 input_resolution=input_resolution)
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim,
                                         norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}," \
               f"input_resolution={self.input_resolution}," \
               f"depth={self.depth}"



class USPNet(nn.Module):

    def __init__(self,
                 n_div=12,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 # #small
                 # embed_dim=64,
                 # depths=(1, 1, 3, 1),
                 #base
                 # embed_dim=96,
                 # depths=(2, 2, 6, 2),
                 #large
                 embed_dim=128,
                 depths=(2, 2, 6, 2),
                 mlp_ratio=4.,
                 drop_rate=0.,
                 norm_layer='GN1',
                 patch_norm=True,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        assert norm_layer in ('GN1', 'BN')
        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'GN1':
            norm_layer = partial(GroupNorm, num_groups=1)
        else:
            raise NotImplementedError


        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)


        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               n_div=n_div,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) \
            if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = USPNet(num_classes=4)
    # print(model)
    x = torch.randn(10, 3, 224, 224)
    y = model(x)
    print(y.shape)