import torch
import torch.nn as nn
import torch.nn.functional as F

# --- [1] Encoder: Parallel Residual Block (ป้องกัน Gridding & รักษา Pixel) ---
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_rate=2):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        # แยกเป็น 2 Path ขนานกัน (Standard + Dilated)
        self.path_std = nn.Conv2d(in_channels, out_channels // 2, 3, padding=1, bias=False)
        self.path_dil = nn.Conv2d(in_channels, out_channels // 2, 3, padding=d_rate, dilation=d_rate, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.path_std(x), self.path_dil(x)], dim=1)
        out = self.bn(out)
        return self.relu(out + self.shortcut(x))

# --- [2] Bottleneck: Residual ASPP (Rates 2, 4, 6 จากโค้ดต้นฉบับของคุณ) ---
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
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.aspp1(x)
        feat2 = self.aspp2(x)
        feat3 = self.aspp3(x)
        feat4 = self.aspp4(x)
        feat5 = F.interpolate(self.global_avg_pool(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1))

class ResidualASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp = ASPP(in_channels, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.aspp(x) + self.shortcut(x))

# --- [3] Skip Connection: Attention Gate (กรองฟีเจอร์จาก Skip Connection) ---
class AttentionGate(nn.Module):
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
        g1 = self.W_g(g)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# --- [4] Decoder Block: Residual Reconstruction ---
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

# --- [5] Main Architecture: BraTSSmallNet ---
class UNetRes(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        self.model_name = "ResUNet_AG_ASPP_DS"
        self.model_info = {
            'model_name': self.model_name,
            'in_channel': in_channels, 
            'out_channel': num_classes,
            'description': "U-Net with attention gate, parallel-residual encoder, ASPP bottleneck, and Deep Supervision."
        }

        # Encoder Stages (Hierarchical Sequential-Parallel)
        self.enc1 = EncoderBlock(in_channels, 64, d_rate=1)
        self.enc2 = EncoderBlock(64, 128, d_rate=2)
        self.enc3 = EncoderBlock(128, 256, d_rate=2)
        self.enc4 = EncoderBlock(256, 512, d_rate=3)

        # Bottleneck
        self.bottleneck = ResidualASPP(512, 1024)

        # Attention Gates
        self.ag4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.ag3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.ag2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.ag1 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # Decoder & Up-sampling
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DecoderBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DecoderBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DecoderBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DecoderBlock(128, 64)

        # Deep Supervision Heads
        self.ds4 = nn.Conv2d(512, num_classes, 1)
        self.ds3 = nn.Conv2d(256, num_classes, 1)
        self.ds2 = nn.Conv2d(128, num_classes, 1)
        
        # Final Output Layer
        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # --- Encoder Path ---
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        s4 = self.enc4(F.max_pool2d(s3, 2))

        # --- Bottleneck ---
        b = self.bottleneck(F.max_pool2d(s4, 2))

        # --- Decoder Path with Spatial Alignment ---
        
        # Level 4
        d4 = self.up4(b)
        if d4.shape[2:] != s4.shape[2:]:
            d4 = F.interpolate(d4, size=s4.shape[2:], mode='bilinear', align_corners=False)
        a4 = self.ag4(g=d4, x=s4)
        d4 = self.dec4(torch.cat([d4, a4], dim=1))
        
        # Level 3
        d3 = self.up3(d4)
        if d3.shape[2:] != s3.shape[2:]:
            d3 = F.interpolate(d3, size=s3.shape[2:], mode='bilinear', align_corners=False)
        a3 = self.ag3(g=d3, x=s3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))
        
        # Level 2
        d2 = self.up2(d3)
        if d2.shape[2:] != s2.shape[2:]:
            d2 = F.interpolate(d2, size=s2.shape[2:], mode='bilinear', align_corners=False)
        a2 = self.ag2(g=d2, x=s2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))
        
        # Level 1
        d1 = self.up1(d2)
        if d1.shape[2:] != s1.shape[2:]:
            d1 = F.interpolate(d1, size=s1.shape[2:], mode='bilinear', align_corners=False)
        a1 = self.ag1(g=d1, x=s1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        # Final Mask
        final_mask = self.outc(d1)

        # --- Deep Supervision Path ---
        if self.training:
            out4 = F.interpolate(self.ds4(d4), size=x.shape[2:], mode='bilinear', align_corners=False)
            out3 = F.interpolate(self.ds3(d3), size=x.shape[2:], mode='bilinear', align_corners=False)
            out2 = F.interpolate(self.ds2(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
            return final_mask, out4, out3, out2
        
        return final_mask

if __name__ == "__main__":
    model = UNetRes(in_channels=4, num_classes=4)
    test_input = torch.randn(1, 4, 182, 218)
    output = model(test_input)
    if isinstance(output, tuple):
        print(f"Training mode: Main output shape {output[0].shape}, DS outputs count: {len(output)-1}")
    else:
        print(f"Eval mode: Output shape {output.shape}")