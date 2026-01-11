import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

class UNetBiPyramid(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision=False):
        super(UNetBiPyramid, self).__init__()
        self.model_name = "U-Net_BiPyramid"
        self.deep_supervision = deep_supervision
        self.model_info = {
            'model_name': self.model_name,
            'in_channel': in_channels, 
            'out_channel(class)': num_classes
            }
        if self.deep_supervision:
            self.model_info['description'] = "Add deep supervision to help tuning loss at standard decoder(out 2) and right pyramid(out 3)"

        # --- 1. Encoder (Downsampling) ---
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.enc5 = double_conv(512, 1024) # Bottleneck
        self.pool = nn.MaxPool2d(2)

        # --- 2. Center Pyramid (Out 1) ---
        self.lat_e4 = nn.Conv2d(512, 256, 1)
        self.lat_e3 = nn.Conv2d(256, 256, 1)
        self.lat_e2 = nn.Conv2d(128, 256, 1)
        self.lat_e1 = nn.Conv2d(64, 256, 1)
        
        self.up_p5 = nn.ConvTranspose2d(1024, 256, 2, 2)
        self.up_p4 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.up_p3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.up_p2 = nn.ConvTranspose2d(256, 256, 2, 2)

        # --- 3. Standard Decoder (Out 2) ---
        self.up_d5 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = double_conv(1024, 512)
        self.up_d4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = double_conv(512, 256)
        self.up_d3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = double_conv(256, 128)
        self.up_d2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = double_conv(128, 64)

        # --- 4. Right Pyramid (Out 3) ---
        self.lat_d4 = nn.Conv2d(512, 64, 1)
        self.lat_d3 = nn.Conv2d(256, 64, 1)
        self.lat_d2 = nn.Conv2d(128, 64, 1)
        self.lat_d1 = nn.Conv2d(64, 64, 1)
        
        self.up_r4 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.up_r3 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.up_r2 = nn.ConvTranspose2d(64, 64, 2, 2)

        # --- 5. Final Output Fusion ---
        self.final_head = nn.Sequential(
            nn.Conv2d(256 + 64 + 64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

        # --- 6. Deep Supervision Heads (Auxiliary Classifiers) ---
        if self.deep_supervision:
            # Out 2 (Standard Decoder) outputs 64 channels -> map to num_classes
            self.ds_head_out2 = nn.Conv2d(64, num_classes, 1)
            # Out 3 (Right Pyramid) outputs 64 channels -> map to num_classes
            self.ds_head_out3 = nn.Conv2d(64, num_classes, 1)

    def _match_and_add(self, x, target):
        """Helper function: ปรับขนาด x ให้เท่ากับ target แล้วนำมาบวกกัน"""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x + target

    def _match_and_concat(self, x, skip):
        """Helper function: ปรับขนาด x ให้เท่ากับ skip แล้วนำมาต่อกัน (Concat)"""
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        # --- ENCODER PATH ---
        e1 = self.enc1(x)                # 1/1
        e2 = self.enc2(self.pool(e1))     # 1/2
        e3 = self.enc3(self.pool(e2))     # 1/4
        e4 = self.enc4(self.pool(e3))     # 1/8
        e5 = self.enc5(self.pool(e4))     # 1/16

        # --- PATH 1: CENTER PYRAMID (Out 1) ---
        p4 = self._match_and_add(self.up_p5(e5), self.lat_e4(e4))
        p3 = self._match_and_add(self.up_p4(p4), self.lat_e3(e3))
        p2 = self._match_and_add(self.up_p3(p3), self.lat_e2(e2))
        p1 = self._match_and_add(self.up_p2(p2), self.lat_e1(e1))
        out1 = p1 

        # --- PATH 2: STANDARD DECODER (Out 2) ---
        d4 = self.dec4(self._match_and_concat(self.up_d5(e5), e4))
        d3 = self.dec3(self._match_and_concat(self.up_d4(d4), e3))
        d2 = self.dec2(self._match_and_concat(self.up_d3(d3), e2))
        d1 = self.dec1(self._match_and_concat(self.up_d2(d2), e1))
        out2 = d1

        # --- PATH 3: RIGHT PYRAMID (Out 3) ---
        r3 = self._match_and_add(self.up_r4(self.lat_d4(d4)), self.lat_d3(d3))
        r2 = self._match_and_add(self.up_r3(r3), self.lat_d2(d2))
        r1 = self._match_and_add(self.up_r2(r2), self.lat_d1(d1))
        out3 = r1

        # --- FINAL FUSION ---
        combined = torch.cat([out1, out2, out3], dim=1)
        final_out = self.final_head(combined)

        # Return Logic for Deep Supervision
        if self.deep_supervision and self.training:
            # Apply 1x1 conv to auxiliary outputs to get class logits
            aux2 = self.ds_head_out2(out2)
            aux3 = self.ds_head_out3(out3)
            # Return list: [Main, Aux1, Aux2]
            return [final_out, aux2, aux3]
        else:
            return final_out


# ทดสอบขนาด Output
if __name__ == "__main__":
    model = UNetBiPyramid(in_channels=4, num_classes=4)
    test_input = torch.randn(1, 4, 182, 218)
    output = model(test_input)
    print(f"Final output shape: {output.shape}") 