import torch
from torchvision import models
from .resnet import *
from .basic_blocks import *

import torch.nn.functional as F
import torch.nn as nn
from .DeformableDecoder import DeformableRoadDecoder
class Curfres(nn.Module):
    def __init__(self,config, num_classes=1, num_channels=3, encoder_1dconv=0,  decoder_1dconv=0):
        super(Curfres, self).__init__()
        self.config=config
        filters = [64, 128, 256, 512]
        self.num_channels = num_channels
        resnet = models.resnet34(pretrained=True)
        if num_channels < 3:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        else:
            self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        if encoder_1dconv == 0:
            self.encoder1 = resnet.layer1   #3resnetblock
            self.encoder2 = resnet.layer2   #4resnetblock
            self.encoder3 = resnet.layer3   #6resnetblock
            self.encoder4 = resnet.layer4   #3resnetblock
        else:
            myresnet = ResnetBlock()
            layers = [3, 4, 6, 3]
            basicBlock = BasicBlock1DConv
            self.encoder1 = myresnet._make_layer(basicBlock, 64, layers[0])
            self.encoder2 = myresnet._make_layer(
                basicBlock, 128, layers[1], stride=2)
            self.encoder3 = myresnet._make_layer(
                basicBlock, 256, layers[2], stride=2)
            self.encoder4 = myresnet._make_layer(
                basicBlock, 512, layers[3], stride=2)
        #与vitb结合时通道调整
        self.skip_conv1 = nn.Conv2d(768, 64, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(768, 128, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(768, 256, kernel_size=1)
        self.skip_conv4 = nn.Conv2d(768, 512, kernel_size=1)
        self.avg_pool=nn.AvgPool2d(kernel_size=2, stride=2)

        self.dblock = DBlock(512)

        if decoder_1dconv == 0:
            self.decoder = DecoderBlock
        elif decoder_1dconv == 2:
            self.decoder = DecoderBlock1DConv2
        elif decoder_1dconv == 4:#
            self.decoder = DecoderBlock1DConv4   #
        if self.config.USE_NEW_DLINK_DECODER:
            # decoder4:  (10, 512, 16, 16) -> (10, 256, 32, 32)
            self.decoder4 = DeformableRoadDecoder(in_channels=512, out_channels=[128, 64, 64])
            # decoder3: (10, 256, 32, 32) -> (10, 128, 64, 64)
            self.decoder3= DeformableRoadDecoder(in_channels=256, out_channels=[64, 32, 32])
            # decoder2:  (10, 128, 64, 64)->  (10, 64, 128, 128)
            self.decoder2= DeformableRoadDecoder(in_channels=128, out_channels=[32, 16, 16])
            # decoder1:  (10, 64, 128, 128)->  (10, 64, 256, 256)
            self.decoder1= DeformableRoadDecoder(in_channels=64, out_channels=[32, 16, 16])
        else:
            self.decoder4 = self.decoder(filters[3], filters[2])#filters = [64, 128, 256, 512]
            self.decoder3 = self.decoder(filters[2], filters[1])
            self.decoder2 = self.decoder(filters[1], filters[0])
            self.decoder1 = self.decoder(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)#in=64 out=32
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)#in=32 out=32
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)#in=32 out=1

        if self.num_channels > 3:
            self.addconv = nn.Conv2d(
                self.num_channels - 3, 64, kernel_size=7, stride=2, padding=3)

    #
    def forward(self, x,encoder_features):
        # Encoder
        if self.num_channels > 3:
            #
            #
            a1=x.narrow(1, 3,  self.num_channels - 3)
            add_gps_osm = self.addconv(x.narrow(1, 3,  self.num_channels - 3))
            x = self.firstconv(x.narrow(1, 0, 3))
            x = x + add_gps_osm
        else:#I,T
            x = self.firstconv(x)#(2,64,256,256)
        x = self.firstbn(x)#(10,64,256,256)
        x = self.firstrelu(x)#(10,64,256,256)
        x = self.firstmaxpool(x)#(10,64,256,256)
        #dlinknet encoder
        e1 = self.encoder1(x)#(10,64,128,128)    #3resnetblock
        e2 = self.encoder2(e1)#(10,128,64,64)    #4resnetblock
        e3 = self.encoder3(e2)#(10,256,32,32)    #6resnetblock
        e4 = self.encoder4(e3)#(10,512,16,16)    #3resnetblock


        if self.config.DLINKNET_ENCODER_AND_VIT==True:
            f1 = encoder_features[1].transpose(1, 3)  #10,32,32,768(c)->10,768(c),32,32
            f1 = self.skip_conv1(f1)  # 10,768(c),32,32 -> 10,64(c),32,32
            f1=F.interpolate(f1, size=e1.shape[2:], mode='bilinear', align_corners=False)  # 上采样10,64(c),32,32->10,64(c),128,128


            f2 = encoder_features[5].transpose(1, 3)  #10,32,32,768(c)->10,768(c),32,32
            f2 = self.skip_conv2(f2)  # 10,768(c),32,32 -> 10,128(c),32,32
            f2=F.interpolate(f2, size=e2.shape[2:], mode='bilinear', align_corners=False)  # 上采样10,128(c),32,32->10,128(c),64,64


            f3 = encoder_features[9].transpose(1, 3)  #10,32,32,768(c)->10,768(c),32,32
            f3 = self.skip_conv3(f3)  # 10,768(c),32,32 -> 10,256(c),32,32


            f4 = encoder_features[11].transpose(1, 3)  #10,32,32,768(c)->10,768(c),32,32
            f4 = self.skip_conv4(f4)  # 10,768(c),32,32 -> 10,512(c),32,32
            f4=self.avg_pool(f4)  # 下采样10,512(c),32,32->10,512(c),16,16
            e4=e4+f4
        # Center
        e4 = self.dblock(e4)##(10,512,16,16) dilation cooperate

        # Decoder

        d4 = self.decoder4(e4) + e3#(10,256,32,32)
        d3 = self.decoder3(d4) + e2#(10,128,64,64)
        d2 = self.decoder2(d3) + e1#(10,64,128,128)
        d1 = self.decoder1(d2)#(10,64,256,256)

        out = self.finaldeconv1(d1)#(10,32,512,512)
        out = self.finalrelu1(out)#(10,32,512,512)
        out = self.finalconv2(out)#(10,32,512,512)
        out = self.finalrelu2(out)#(10,32,512,512)
        out = self.finalconv3(out)#(10,1,512,512)


        return out



class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x




