import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UpBlock(nn.Module):
    """Upsampling ตามด้วยการ Concat และ DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ลด channel ลงครึ่งหนึ่งด้วย ConvTranspose
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        
        # จัดการเรื่อง Size Mismatch แบบ Dynamic
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 conv
        self.aspp1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        # 3x3 atrous conv ด้วย dilation rates ต่างๆ
        self.aspp2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.aspp4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        # Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        res = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(res)

class UNetASPP(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model_name = "U-Net_ASPP"
        self.model_info = {
            'model_name': self.model_name,
            'in_channel': in_channels, 
            'out_channel(class)': num_classes,
            'description': "U-Net with ASPP block in bottleneck"
            }

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # เปลี่ยน Bottleneck เป็น ASPP
        self.aspp_bottleneck = nn.Sequential(nn.MaxPool2d(2), ASPP(512, 1024))

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        b = self.aspp_bottleneck(x4)
        x = self.up1(b, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# ทดสอบขนาด Output
if __name__ == "__main__":
    model = UNetASPP(in_channels=4, num_classes=4)
    test_data = torch.randn(1, 4, 182, 218)
    output = model(test_data)
    print(f"UNetASPP Output Shape: {output.shape}")