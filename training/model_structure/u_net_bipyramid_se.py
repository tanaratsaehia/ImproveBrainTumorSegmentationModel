import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class UNetBiPyramidSE(nn.Module):
    def __init__(self, in_channels, num_classes, reduction=16):
        super(UNetBiPyramidSE, self).__init__()
        
        # --- 1. Encoder (ซ้ายสุด - 5 Levels) ---
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.enc5 = double_conv(512, 1024) # Bottleneck (ชั้นที่ 5)
        self.pool = nn.MaxPool2d(2)

        # --- 2. Center Pyramid (Out 1 - ทางด่วนตรงกลาง) ---
        # 1x1 conv (ลูกศรม่วง) เพื่อปรับ channel ให้เท่ากัน (สมมติใช้ 256)
        self.lat_e4 = nn.Conv2d(512, 256, 1)
        self.lat_e3 = nn.Conv2d(256, 256, 1)
        self.lat_e2 = nn.Conv2d(128, 256, 1)
        self.lat_e1 = nn.Conv2d(64, 256, 1)
        
        # Up-conv 2x2 (ลูกศรเขียวตรงกลาง)
        self.up_p5 = nn.ConvTranspose2d(1024, 256, 2, 2)
        self.up_p4 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.up_p3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.up_p2 = nn.ConvTranspose2d(256, 256, 2, 2)

        # --- 3. Standard Decoder (Out 2 - ขยับมาทางขวา) ---
        # รับข้อมูลจาก Bottleneck และ Skip Connection จาก Encoder
        self.up_d5 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = double_conv(1024, 512) # 512(skip) + 512(up)
        self.se_dec4 = SEBlock(512, reduction)
        
        self.up_d4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = double_conv(512, 256)
        self.se_dec3 = SEBlock(256, reduction)
        
        self.up_d3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = double_conv(256, 128)
        self.se_dec2 = SEBlock(128, reduction)
        
        self.up_d2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = double_conv(128, 64)
        self.se_dec1 = SEBlock(64, reduction)

        # --- 4. Right Pyramid (Out 3 - ขวาสุด) ---
        # สกัดฟีเจอร์จาก Decoder แต่ละชั้นมาทำ Pyramid อีกรอบ
        self.lat_d4 = nn.Conv2d(512, 64, 1)
        self.lat_d3 = nn.Conv2d(256, 64, 1)
        self.lat_d2 = nn.Conv2d(128, 64, 1)
        self.lat_d1 = nn.Conv2d(64, 64, 1)
        
        self.up_r4 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.up_r3 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.up_r2 = nn.ConvTranspose2d(64, 64, 2, 2)

        # --- 5. Final Output Fusion ---
        # รวม Out1 (256ch), Out2 (64ch), Out3 (64ch)
        self.final_head = nn.Sequential(
            nn.Conv2d(256 + 64 + 64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        # --- ENCODER PATH ---
        e1 = self.enc1(x)              # 1/1
        e2 = self.enc2(self.pool(e1))   # 1/2
        e3 = self.enc3(self.pool(e2))   # 1/4
        e4 = self.enc4(self.pool(e3))   # 1/8
        e5 = self.enc5(self.pool(e4))   # 1/16 (Bottleneck)

        # --- PATH 1: CENTER PYRAMID (Out 1) ---
        # ไหลจากล่างขึ้นบน: Up-conv จากชั้นล่างมา Add กับ 1x1 ของชั้นตัวเอง
        p4 = self.lat_e4(e4) + self.up_p5(e5)
        p3 = self.lat_e3(e3) + self.up_p4(p4)
        p2 = self.lat_e2(e2) + self.up_p3(p3)
        p1 = self.lat_e1(e1) + self.up_p2(p2)
        out1 = p1 

        # --- PATH 2: STANDARD DECODER (Out 2) With Squeeze and Excitation---
        # U-Net: Concat กับ Skip Connection ตรงๆ จาก Encoder
        d4 = self.dec4(torch.cat([self.up_d5(e5), e4], dim=1))
        d4 = self.se_dec4(d4)
        d3 = self.dec3(torch.cat([self.up_d4(d4), e3], dim=1))
        d3 = self.se_dec3(d3)
        d2 = self.dec2(torch.cat([self.up_d3(d3), e2], dim=1))
        d2 = self.se_dec2(d2)
        d1 = self.dec1(torch.cat([self.up_d2(d2), e1], dim=1))
        d1 = self.se_dec1(d1)
        out2 = d1

        # --- PATH 3: RIGHT PYRAMID (Out 3) ---
        # สร้าง Pyramid จากผลลัพธ์ของ Decoder (d4 -> d1)
        # เริ่มจากชั้น d4 ส่งขึ้นไปหา d1
        r3 = self.lat_d3(d3) + self.up_r4(self.lat_d4(d4))
        r2 = self.lat_d2(d2) + self.up_r3(r3)
        r1 = self.lat_d1(d1) + self.up_r2(r2)
        out3 = r1

        # --- FINAL FUSION ---
        # รวมข้อมูลจาก 3 แหล่งที่มีความละเอียดเท่าภาพ Input
        combined = torch.cat([out1, out2, out3], dim=1)
        return self.final_head(combined)


# ทดสอบขนาด Output
if __name__ == "__main__":
    model = UNetBiPyramidSE(in_channels=4, num_classes=2)
    test_input = torch.randn(1, 4, 256, 256)
    output = model(test_input)
    print(f"Final output shape: {output.shape}") # ควรต้องเป็น [1, 2, 256, 256]