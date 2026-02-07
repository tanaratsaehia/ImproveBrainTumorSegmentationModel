import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""
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

    def forward(self, x):
        return self.double_conv(x)

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

class UNet4Layer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model_name = "U-Net_4Layer"
        self.model_info = {
            'model_name': self.model_name,
            'in_channel': in_channels, 
            'out_channel': num_classes,
            'description': "Original U-Net that reduce to 4 layers (include bottle neck)"
            }
        # --- Encoder ---
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        # self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        # self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # --- Decoder ---
        # UpBlock(ช่องสัญญาณเข้า, ช่องสัญญาณออก)
        # self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        # --- Output ---
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # ขาลง (Encoder)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        
        # ฐาน (Bottleneck)
        b  = self.bottleneck(x3)
        # b  = self.bottleneck(x4)

        # ขาขึ้น (Decoder + Skip Connections)
        # x = self.up1(b, x4)
        x = self.up2(b, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

# --- ทดสอบการใช้งาน ---
if __name__ == "__main__":
    model = UNet4Layer(in_channels=4, num_classes=4)
    test_data = torch.randn(1, 4, 182, 218)
    output = model(test_data)
    print(f"UNet Output Shape: {output.shape}")