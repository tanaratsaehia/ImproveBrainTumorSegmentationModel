import torch
import torch.nn as nn
import torch.nn.functional as F

class scSE(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Block
    ช่วยเน้นทั้ง 'Channel' และ 'Spatial' (ตำแหน่งพิกเซล) ที่สำคัญ
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel Squeeze and Excitation (cSE)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        # Spatial Squeeze and Excitation (sSE)
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x * cSE(x) + x * sSE(x)
        return x * self.cSE(x) + x * self.sSE(x)

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

    def forward(self, x):
        return self.double_conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_scse=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
        # เพิ่ม scSE เข้าไปหลัง DoubleConv ใน Decoder
        self.scse = scSE(out_channels) if use_scse else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.scse(x) # ผ่าน scSE เพื่อทำ Attention

class UNet_scSE(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model_name = "U-Net_scSE"
        self.model_info = {
            'model_name': self.model_name,
            'in_channel': in_channels, 
            'out_channel': num_classes,
            'description': "U-Net with scSE Attention"
        }

        # --- Encoder ---
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # --- Decoder (เพิ่ม scSE ทุก Block) ---
        self.up1 = UpBlock(1024, 512, use_scse=True)
        self.up2 = UpBlock(512, 256, use_scse=True)
        self.up3 = UpBlock(256, 128, use_scse=True)
        self.up4 = UpBlock(128, 64, use_scse=True)

        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b  = self.bottleneck(x4)

        x = self.up1(b, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

# --- ทดสอบการใช้งาน ---
if __name__ == "__main__":
    model = UNet_scSE(in_channels=4, num_classes=4)
    test_data = torch.randn(1, 4, 182, 218)
    output = model(test_data)
    print(f"UNet_scSE Output Shape: {output.shape}")