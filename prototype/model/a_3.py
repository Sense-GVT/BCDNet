import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from prototype.utils.misc import get_logger, get_bn

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 4 + [1024] * 1 # 6 2

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def binaryconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = BN(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out
##########################################################################################

class Shift(nn.Module):
    def __init__(self):
        super(Shift, self).__init__()
        self.pad1 = nn.ZeroPad2d(padding=(0, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        self.pad3 = nn.ZeroPad2d(padding=(1, 0, 0, 0))
        self.pad4 = nn.ZeroPad2d(padding=(0, 1, 0, 0))

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim = 1)

        x1 = torch.roll(x1, 1, dims=2)#[:,:,1:,:]
        #x1 = self.pad1(x1)
        x2 = torch.roll(x2, -1, dims=2)#[:,:,:-1,:]
        #x2 = self.pad2(x2)
        x3 = torch.roll(x3, 1, dims=3)#[:,:,:,1:]
        #x3 = self.pad3(x3)
        x4 = torch.roll(x4, -1, dims=3)#[:,:,:,:-1]
        #x4 = self.pad4(x4)
        
        x = torch.cat([x1, x2, x3, x4], 1)

        return x

class GlobalShift(nn.Module):
    def __init__(self):
        super(GlobalShift, self).__init__()
        self.pad1 = nn.ZeroPad2d(padding=(0, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        self.pad3 = nn.ZeroPad2d(padding=(1, 0, 0, 0))
        self.pad4 = nn.ZeroPad2d(padding=(0, 1, 0, 0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2, x3, x4 = x.chunk(4, dim = 1)

        x1 = torch.roll(x1, 3, dims=2)#[:,:,1:,:]
        #x1 = self.pad1(x1)
        x2 = torch.roll(x2, -3, dims=2)#[:,:,:-1,:]
        #x2 = self.pad2(x2)
        x3 = torch.roll(x3, 3, dims=3)#[:,:,:,1:]
        #x3 = self.pad3(x3)
        x4 = torch.roll(x4, -3, dims=3)#[:,:,:,:-1]
        #x4 = self.pad4(x4)
        
        x = torch.cat([x1, x2, x3, x4], 1)

        return x

class ShiftBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1, padding=0):
        super(ShiftBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.shift1 = Shift()
        self.shift2 = GlobalShift()
        self.binary_activation = BinaryActivation()

        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride, kernel_size=kernel_size, padding=padding)
        self.binary_conv1 = HardBinaryConv(inplanes, planes, stride=stride, kernel_size=kernel_size, padding=padding)
        self.binary_conv2 = HardBinaryConv(inplanes, planes, stride=stride, kernel_size=kernel_size, padding=padding)

        self.bn1 = BN(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)

        out1 = self.binary_conv(out)

        out2 = self.shift1(out)
        out2 = self.binary_conv1(out2)

        out3 = self.shift2(out)
        out3 = self.binary_conv2(out3)

        out = self.bn1(out1 + out2 + out3)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out

class PointBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1, padding=0):
        super(PointBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride, kernel_size=kernel_size, padding=padding)
        self.bn1 = BN(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out

class Block(nn.Module):
    expansion = 1

    def __init__(self, channel):
        super(Block, self).__init__()
        mlp_layer = []

        mlp_layer.append(ShiftBlock(channel, channel))
        mlp_layer.append(ShiftBlock(channel, channel))
        mlp_layer.append(ShiftBlock(channel, channel))
        mlp_layer.append(PointBlock(channel, channel))

        self.mlp_layer = nn.Sequential(*mlp_layer)

    def forward(self, x):
        x = self.mlp_layer(x)
        return x
###################################################################

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        #real_weights = self.weights.view(self.shape)
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = BN

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3= binaryconv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = binaryconv1x1(inplanes, planes)
            self.bn2 = norm_layer(planes)
        else:
            self.binary_pw_down1 = binaryconv1x1(inplanes, inplanes)
            self.binary_pw_down2 = binaryconv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2,2)

    def forward(self, x):

        out1 = self.move11(x)

        out1 = self.binary_activation(out1)
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        out2 = self.binary_activation(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1

        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2


class Reactnet(nn.Module):
    def __init__(self, num_classes=1000, bn=None):
        super(Reactnet, self).__init__()
        global BN
        BN = get_bn(bn)

        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1))

        self.mlp_layer = self.build(3, 1024)

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        self.fc_dist = nn.Linear(1024, num_classes)
    
    def build(self, num, channel):
        mlp_layer = []
        for i in range(num):
            mlp_layer.append(Block(channel))
        return nn.Sequential(*mlp_layer)

    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)

        ########################
        x = self.mlp_layer(x)
        ########################

        x = self.pool1(x)
        x = x.view(x.size(0), -1)

        ########################
        x_res = self.fc(x)
        x_dist = self.fc_dist(x)

        if self.training:
            return x_res, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x_res + x_dist) / 2


def a_3(**kwargs):
    
    model = Reactnet(**kwargs)
    return model

if __name__ == "__main__":
    print("################################")
    net = a_3()
    x = torch.rand((3, 3, 224, 224))
    y = net(x)
    print(y.size())

