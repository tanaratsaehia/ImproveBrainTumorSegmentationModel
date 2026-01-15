import torch
import torch.nn as nn
import torch.nn.functional as F

# --- [Encoder Module] Multi-Scale Dilated Ghost-Residual Block ---
class MDGRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        
        # 1. Multi-Scale Dilated Paths (Context Extraction)
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Feature Fusion
        self.fuse = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        
        # 3. Coordinate Attention (Spatial Awareness)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.attn_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU6(inplace=True)
        )
        self.conv_h = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)

        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.identity(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = self.fuse(torch.cat([x1, x2], dim=1))
        
        b, c, h, w = out.size()
        x_h = self.pool_h(out)
        x_w = self.pool_w(out).permute(0, 1, 3, 2)
        y = self.attn_conv(torch.cat([x_h, x_w], dim=2))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        attn = self.conv_h(y_h).sigmoid() * self.conv_w(y_w.permute(0, 1, 3, 2)).sigmoid()
        
        return self.final_relu(out * attn + res)


# --- [Decoder Module] Feature Refinement Decoder Block ---
class FRDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. Depthwise Separable Convolution (Refinement)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Gated Feature Selection (Filtering Noise from Skip Connection)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_refine(x)
        g = self.gate(out)
        return out * g


# --- [Full Hybrid UNet Architecture] ---
class HybridUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model_name = "Hybrid_MDGR_FRD_UNet"
        self.model_info = {
            'model_name': self.model_name,
            'in_channel': in_channels, 
            'out_channel': num_classes
            }
        
        # --- Encoder (Using MDGRBlock) ---
        self.inc   = MDGRBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), MDGRBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), MDGRBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), MDGRBlock(256, 512))
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), MDGRBlock(512, 1024))

        # --- Decoder (Using FRDBlock) ---
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.frd1     = FRDBlock(1024, 512) # 512(up) + 512(skip) = 1024 input
        
        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.frd2     = FRDBlock(512, 256)
        
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.frd3     = FRDBlock(256, 128)
        
        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.frd4     = FRDBlock(128, 64)

        # --- Output Layer ---
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        
        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder Path with Skip Connections
        def up_step(x, skip, up_conv, frd_block):
            x = up_conv(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return frd_block(x)

        d1 = up_step(b, s4, self.up_conv1, self.frd1)
        d2 = up_step(d1, s3, self.up_conv2, self.frd2)
        d3 = up_step(d2, s2, self.up_conv3, self.frd3)
        d4 = up_step(d3, s1, self.up_conv4, self.frd4)

        return self.outc(d4)

if __name__ == "__main__":
    model = HybridUNet(in_channels=4, num_classes=4)
    test_data = torch.randn(1, 4, 182, 218)
    output = model(test_data)
    print(f"HybridUNet Output Shape: {output.shape}")