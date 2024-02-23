import torch
import torch.nn as nn

class PCE(nn.Module):
    def __init__(self, C, k, s, classes):
         super().__init__()
         self.gps = 2
         self.dim = 64
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

