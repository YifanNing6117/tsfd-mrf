"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F

# Conv Block    2 x (Conv + In + LeakyReLU + Dropout)
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),

            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob)
            # nn.Conv2d(in_channels = in_chans,
            #                         out_channels = out_chans,
            #                         kernel_size = 3,
            #                         stride = 1,
            #                         padding = 1
            #                         ),
            # nn.BatchNorm2d(out_chans),
            # nn.ReLU())
        )
    def forward(self, input_):
        """
        Args:
            input_ (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input_)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


# DeConv Block (ConvTranspose2d + IN + LeakyReLU) to enlarge the size of feature map
class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_):
        """
        Args:
            input_ (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input_)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


# U-Net baseline model
class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans=200, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans     # in_channels = 1
        self.out_chans = out_chans  # out_channels = 1
        self.chans = chans                      # default value = 32
        self.num_pool_layers = num_pool_layers  # default value = 4
        # print(num_pool_layers)
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2  # double the channels
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        # ch=256
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2  # half the channels
        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )]

    def forward(self, input_):
        """
        Args:
            input_ (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []  # record down-sampling feature maps for skip connection in the same resolution when up-sampling
        output = input_  # input tensor, initial shape = (n, 1, 320, 320) for knee

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)  # out_channels: 1 -> 32 -> 64 -> 128 -> 256
            # print(output.shape)
            stack.append(output)    # record down-sampling feature maps
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            # print(i,output.shape)
        output = self.conv(output)  # num_channels: 256 -> 512

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()   # channels: 256、128、64、32
            # print(downsample_layer.shape)
            output = transpose_conv(output)  # out_channels: 512 -> 256 -> 128 -> 64 -> 32
            # print(output.shape)
            # Reflect pad on the right/bottom if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            # print(output.shape)
            output = torch.cat([output, downsample_layer], dim=1)  # skip connection - concat along channel dimension
            # print(output.shape)
            output = conv(output)  # out_channels: 512 -> 256 -> 128 -> 64 -> 32 -> 1
            # print(output.shape)

        return output

    '''
    Function:
        torch.nn.functional.pad(input, pad, mode, value) 
    Args:
        input： 4D or 5D Tensor
        pad： 不同 Tensor 的填充方式
            1. 4D-Tensor： 传入四元素 tuple(pad_l, pad_r, pad_t, pad_b)，
                          指（左填充，右填充，上填充，下填充），其数值代表填充次数
            2. 6D-Tensor： 传入六元素 tuple(pleft, pright, ptop, pbottom, pfront, pback)，
                          指（左填充，右填充，上填充，下填充，前填充，后填充），其数值代表填充次数
        mode： ’constant‘、‘reflect’、‘replicate’ 三种模式，指常量、反射、复制三种模式
        value： 填充的数值，在 "contant" 模式下默认填充 0，mode="reflect" or "replicate" 时没有value参数			
    '''
# class STQ(nn.Module):
#     """
#     PyTorch implementation of a U-Net model.

#     This is based on:
#         Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
#         for biomedical image segmentation. In International Conference on Medical image
#         computing and computer-assisted intervention, pages 234–241. Springer, 2015.
#     """

#     def __init__(self, in_chans=288, out_chans=1,num_pool_layers =3,chans  ):
#         """
#         Args:
#             in_chans (int): Number of channels in the input to the U-Net model.
#             out_chans (int): Number of channels in the output to the U-Net model.
#             chans (int): Number of output channels of the first convolution layer.
#             num_pool_layers (int): Number of down-sampling and up-sampling layers.
#             drop_prob (float): Dropout probability.
#         """
#         super().__init__()
#         self.in_chans = in_chans     # in_channels = 1
#         self.out_chans = out_chans  # out_channels = 1
#         self.chans = chans                      # default value = 32
#         self.num_pool_layers = num_pool_layers  # default value = 4
#         # # print(num_pool_layers)
#         # self.drop_prob = drop_prob

#         self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
#         ch = chans
#         for i in range(num_pool_layers - 1):
#             self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
#             ch *= 2  # double the channels
#         self.conv = ConvBlock(128, 256, drop_prob)
#         # ch=256
#         self.up_conv = nn.ModuleList()
#         self.up_transpose_conv = nn.ModuleList()
#         for i in range(num_pool_layers - 1):
#             self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
#             self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
#             ch //= 2  # half the channels
#         self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
#         self.up_conv += [
#             nn.Sequential(
#                 ConvBlock(ch * 2, ch, drop_prob),
#                 nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
#             )]

#     def forward(self, input_,model_2):
      
#         stack = []  # record down-sampling feature maps for skip connection in the same resolution when up-sampling
#         output = input_  # input tensor, initial shape = (n, 1, 320, 320) for knee

#         # Apply down-sampling layers
#         for i, layer in enumerate(self.down_sample_layers):
#             output = layer(output)  # out_channels: 1 -> 32 -> 64 -> 128 -> 256
#             # print(output.shape)
#             stack.append(output)    # record down-sampling feature maps
#             output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
#             # print(i,output.shape)
#         output = self.conv(output)  # num_channels: 256 -> 512

#         # Apply up-sampling layers
#         for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
#             downsample_layer = stack.pop()   # channels: 256、128、64、32
#             # print(downsample_layer.shape)
#             output = transpose_conv(output)  # out_channels: 512 -> 256 -> 128 -> 64 -> 32
#             # print(output.shape)
#             # Reflect pad on the right/bottom if needed to handle odd input dimensions.
#             padding = [0, 0, 0, 0]
#             if output.shape[-1] != downsample_layer.shape[-1]:
#                 padding[1] = 1  # Padding right
#             if output.shape[-2] != downsample_layer.shape[-2]:
#                 padding[3] = 1  # Padding bottom
#             if sum(padding) != 0:
#                 output = F.pad(output, padding, "reflect")

#             # print(output.shape)
#             output = torch.cat([output, downsample_layer], dim=1)  # skip connection - concat along channel dimension
#             # print(output.shape)
#             output = conv(output)  # out_channels: 512 -> 256 -> 128 -> 64 -> 32 -> 1
#             # print(output.shape)

#         return output