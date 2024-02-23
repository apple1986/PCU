######################################################################################
#U-Net: Convolutional Networks for BiomedicalImage Segmentation
#Paper-Link: https://arxiv.org/pdf/1505.04597.pdf
######################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



__all__ = ["UNet_SRFFA"]


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class PCE(nn.Module):
    def __init__(self, C, k, s, classes):
         super().__init__()
         self.gps = 2
         self.dim = 64
         #需要修改num_class的值
         self.num_class = classes
         self.pixel_shuffle = nn.PixelShuffle(s)
         self.softmax = nn.Softmax(dim=1)
         self.filter = nn.Conv2d(classes,2*s*s*classes,k,1,(k-1)//2,bias=False) # 2*s*s*C
         self.conv1 = nn.Conv2d(classes+1, classes, 1)
         self.conv2 = nn.Conv2d(1, classes, 1)
         self.ca = nn.Sequential(*[
             nn.AdaptiveAvgPool2d(1),
             nn.Conv2d(classes * self.gps, self.dim // 16, 1, padding=0),
             nn.ReLU(inplace=True),
             nn.Conv2d(self.dim // 16, classes * self.gps, 1, padding=0, bias=True),
             nn.Sigmoid()
         ])
         self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(classes, classes, 1)
        )
    def forward(self, x):
        filtered = self.pixel_shuffle(self.filter(x))
        B, C, H, W = filtered.shape
        filtered = filtered.view(B, 2, C // 2, H, W)
        upscaling = filtered[:, 0]
        matching = filtered[:, 1]
        x = self.ca(torch.cat([upscaling, matching], dim=1))
        x = x.view(-1, self.gps, self.num_class)[:, :, :, None, None]
        final_class = x[:, 0, ::] * upscaling + x[:, 1, ::] * matching
        final_class = self.conv(final_class)
        return final_class #torch.sum(upscaling * self.softmax(matching), dim=1, keepdim=True)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.bilinear = bilinear

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.SR = PCE(C=512, k=3, s=2, classes=64)

        self.conv = double_conv(in_ch, out_ch)

    # def forward(self, x1, x2):
    def forward(self, x1, x2):
        if self.bilinear:
            # x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
            if x1.size()[1] == 64:
                # x1 = self.SR(x1)
                x1 = self.SR(x1)
            else:
                x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x



class UNet_SRFFA(nn.Module):
    def __init__(self, classes, channels):
        super(UNet_SRFFA, self).__init__()
        self.inc = inconv(channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, classes)

    # def forward(self, x):
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        # x = self.up4(x, x1)
        x = self.up4(x, x1)
        x = self.outc(x)
        #return F.sigmoid(x)

        return x




"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_SRFFA(classes=19).to(device)
    summary(model,(3,512,1024))