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

class AttentionGate(nn.Module):
    """Attention Gate สำหรับกรองฟีเจอร์จาก Skip Connection"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: feature จากชั้นที่ต่ำกว่า (gate), x: feature จาก skip connection
        g1 = self.W_g(g)
        # ปรับขนาด g1 ให้เท่ากับ x กรณี mismatch
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.att = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=out_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # ใส่ Attention ก่อน Concat
        skip = self.att(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetAG(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model_name = "U-Net_AG"
        self.model_info = {
            'model_name': self.model_name,
            'in_channel': in_channels, 
            'out_channel': num_classes,
            'description': "U-Net with attention gate in skip connection."
            }

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = AttentionUpBlock(1024, 512)
        self.up2 = AttentionUpBlock(512, 256)
        self.up3 = AttentionUpBlock(256, 128)
        self.up4 = AttentionUpBlock(128, 64)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        b = self.bottleneck(x4)
        x = self.up1(b, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# ทดสอบขนาด Output
if __name__ == "__main__":
    model = UNetAG(in_channels=4, num_classes=4)
    test_data = torch.randn(1, 4, 182, 218)
    output = model(test_data)
    print(f"UNetAG Output Shape: {output.shape}")