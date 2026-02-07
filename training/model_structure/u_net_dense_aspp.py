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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, dilation, padding):
        super().__init__()
        self.aspp_unit = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.aspp_unit(x)
        return torch.cat([x, out], dim=1)

class DenseASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # กำหนดช่องสัญญาณภายใน (เราจะแบ่งจาก out_channels เพื่อไม่ให้โมเดลใหญ่เกินไป)
        inter_ch = out_channels // 2 
        
        # Layer 1: Standard 1x1 Conv
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True)
        )
        
        # Dense Layers: แต่ละชั้นจะรับ Input จากทุกชั้นก่อนหน้ามาต่อกัน (Concatenation)
        self.aspp2 = DenseASPPBlock(inter_ch, inter_ch, dilation=2, padding=2)
        self.aspp3 = DenseASPPBlock(inter_ch * 2, inter_ch, dilation=4, padding=4)
        self.aspp4 = DenseASPPBlock(inter_ch * 3, inter_ch, dilation=6, padding=6)
        
        # Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, inter_ch, 1),
            nn.ReLU(inplace=True)
        )
        
        # ผลรวมของ Channel หลัง concat: inter_ch (aspp1) + inter_ch*4 (aspp4 result) + inter_ch (gap)
        self.project = nn.Sequential(
            nn.Conv2d(inter_ch * 6, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)
        
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # รวม feature จากชั้น aspp1 และ x4 (ซึ่งเก็บ x2, x3 มาแล้ว) และ x5
        res = torch.cat([x1, x4, x5], dim=1)
        return self.project(res)

class UNetDenseASPP(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model_name = "U-Net_DenseASPP"
        self.model_info = {
            'model_name': self.model_name,
            'in_channel': in_channels, 
            'out_channel': num_classes,
            'description': "U-Net with DenseASPP block in bottleneck"
        }

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Bottleneck: DenseASPP
        self.aspp_bottleneck = nn.Sequential(nn.MaxPool2d(2), DenseASPP(512, 1024))

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

if __name__ == "__main__":
    model = UNetDenseASPP(in_channels=4, num_classes=4)
    test_data = torch.randn(1, 4, 182, 218)
    output = model(test_data)
    print(f"UNetDenseASPP Output Shape: {output.shape}")