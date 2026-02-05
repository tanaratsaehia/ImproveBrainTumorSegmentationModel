import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ส่วนประกอบพื้นฐาน ---
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4): # เพิ่ม reduction เพื่อประหยัด mem
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ShadowASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        inter_ch = out_ch // 4
        self.b1 = nn.Conv2d(in_ch, inter_ch, 1)
        self.b2 = nn.Conv2d(in_ch, inter_ch, 3, padding=2, dilation=2)
        self.b3 = nn.Conv2d(in_ch, inter_ch, 3, padding=4, dilation=4)
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, inter_ch, 1))
        self.bottleneck = nn.Conv2d(inter_ch * 4, out_ch, 1)
    def forward(self, x):
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = F.interpolate(self.b4(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        return self.bottleneck(torch.cat([feat1, feat2, feat3, feat4], dim=1))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ParallelShadowUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        self.model_name = "ParallelShadowUNet_ASPP_SE_64"
        self.model_info = {
            'model_name': self.model_name,
            'in_channel': in_channels, 
            'out_channel': num_classes,
            'description': "U-Net with Shadow U-net Parallel and ASPP and SE in Base64"
        }

        # --- PRIMARY ENCODER (Base 64) ---
        self.p_enc1 = DoubleConv(in_channels, 64)
        self.p_enc2 = DoubleConv(64, 128)
        self.p_enc3 = DoubleConv(128, 256)
        self.p_enc4 = DoubleConv(256, 512)
        self.p_btlnk = DoubleConv(512, 1024)

        # --- SHADOW ENCODER (Calculated Channels) ---
        self.s_enc1 = DoubleConv(in_channels, 64)
        self.s_enc2 = DoubleConv(64 + 128, 128)   # s1_p(64) + p2(128)
        self.s_enc3 = DoubleConv(128 + 256, 256)  # s2_p(128) + p3(256)
        self.s_enc4 = DoubleConv(256 + 512, 512)  # s3_p(256) + p4(512)
        self.s_btlnk = ShadowASPP(512 + 1024, 1024) # s4_p(512) + p_btlnk(1024)

        # --- PRIMARY DECODER ---
        self.p_dec4 = DoubleConv(1024 + 512, 512)
        self.p_dec3 = DoubleConv(512 + 256, 256)
        self.p_dec2 = DoubleConv(256 + 128, 128)
        self.p_dec1 = DoubleConv(128 + 64, 64)

        # --- SHADOW DECODER (Triple Fusion) ---
        self.s_dec4 = DoubleConv(1024 + 512 + 512, 512) # s_up(1024) + s_skip(512) + p_dec(512)
        self.s_dec3 = DoubleConv(512 + 256 + 256, 256)
        self.s_dec2 = DoubleConv(256 + 128 + 128, 128)
        self.s_dec1 = DoubleConv(128 + 64 + 64, 64)

        # Final Decision
        self.p_out = nn.Conv2d(64, num_classes, 1)
        self.s_out = nn.Conv2d(64, num_classes, 1)
        self.final_se = SEBlock(num_classes * 2)
        self.final_conv = nn.Conv2d(num_classes * 2, num_classes, 1)

    def forward(self, x):
        # Encoder Path
        p1 = self.p_enc1(x)
        s1 = self.s_enc1(x)

        p2 = self.p_enc2(F.max_pool2d(p1, 2))
        s2 = self.s_enc2(torch.cat([F.max_pool2d(s1, 2), p2], dim=1))

        p3 = self.p_enc3(F.max_pool2d(p2, 2))
        s3 = self.s_enc3(torch.cat([F.max_pool2d(s2, 2), p3], dim=1))

        p4 = self.p_enc4(F.max_pool2d(p3, 2))
        s4 = self.s_enc4(torch.cat([F.max_pool2d(s3, 2), p4], dim=1))

        p_bn = self.p_btlnk(F.max_pool2d(p4, 2))
        s_bn = self.s_btlnk(torch.cat([F.max_pool2d(s4, 2), p_bn], dim=1))

        # Decoder Path (Interpolate to handle 182x218)
        p_d4 = self.p_dec4(torch.cat([F.interpolate(p_bn, size=p4.shape[2:], mode='bilinear'), p4], dim=1))
        s_d4 = self.s_dec4(torch.cat([F.interpolate(s_bn, size=s4.shape[2:], mode='bilinear'), s4, p_d4], dim=1))

        p_d3 = self.p_dec3(torch.cat([F.interpolate(p_d4, size=p3.shape[2:], mode='bilinear'), p3], dim=1))
        s_d3 = self.s_dec3(torch.cat([F.interpolate(s_d4, size=s3.shape[2:], mode='bilinear'), s3, p_d3], dim=1))

        p_d2 = self.p_dec2(torch.cat([F.interpolate(p_d3, size=p2.shape[2:], mode='bilinear'), p2], dim=1))
        s_d2 = self.s_dec2(torch.cat([F.interpolate(s_d3, size=s2.shape[2:], mode='bilinear'), s2, p_d2], dim=1))

        p_d1 = self.p_dec1(torch.cat([F.interpolate(p_d2, size=p1.shape[2:], mode='bilinear'), p1], dim=1))
        s_d1 = self.s_dec1(torch.cat([F.interpolate(s_d2, size=s1.shape[2:], mode='bilinear'), s1, p_d1], dim=1))

        # Output
        out_p, out_s = self.p_out(p_d1), self.s_out(s_d1)
        refined = self.final_se(torch.cat([out_p, out_s], dim=1))
        return self.final_conv(refined)

# --- Test กับ Input ที่ระบุ ---
if __name__ == "__main__":
    model = ParallelShadowUNet(n_channels=4, n_classes=4)
    test_input = torch.randn(1, 4, 182, 218)
    output = model(test_input)
    print(f"Input Shape: {test_input.shape}")
    print(f"Output Shape: {output.shape}")