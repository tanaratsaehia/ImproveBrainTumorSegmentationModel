import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Utility Blocks (Your original functions) ---

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

def dilated_block(in_ch, out_ch, dilation=2):
    pad = dilation
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
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

# --- New Model Definition: FPU-Net with Dilation and SE ---

class UNetBiPyramidSE(nn.Module):
    """
    Feature Pyramid U-Net (FPU-Net) enhanced with Dilated Convolutions (Encoder)
    and Squeeze-and-Excitation Blocks (Decoder - on P features).
    """
    def __init__(self, in_channels=4, out_channels=2, num_classes=None, features=[64, 128, 256, 512],
                    fpn_channels=128, reduction=16):
        super().__init__()
        self.model_name = "FPU-Net_SEDilation_"
        if num_classes is not None:
            out_channels = num_classes
        self.features = features
        self.fpn_channels = fpn_channels
        
        # Dilations for C1, C2, C3, C4
        self.dilations = [1, 1, 2, 3] # -> 1 1 2 3 | 1 2 1 2 
        self.model_name = self.model_name + ''.join(str(n) for n in self.dilations)
        
        # C-channels: C1..C4, C5 (bottleneck)
        C_channels = features + [features[-1] * 2]
        
        # ---------------- 1. ENCODER (Bottom-Up - Dilation) ----------------
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for i, f in enumerate(features):
            # 💡 APPLY REQUESTED DILATION RATE
            self.downs.append(dilated_block(prev_ch, f, dilation=self.dilations[i]))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f
        
        # Bottleneck (C5)
        self.bottleneck = conv_block(features[-1], features[-1] * 2)

        # ---------------- 2. FPN (Top-Down Pathway - ACTS AS DECODER) ----------------
        
        # 2a. Lateral 1x1 convs (L1..L5)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, kernel_size=1) for c in C_channels
        ])
        
        # 2b. Optional 3x3 "smooth" convs (P1..P5)
        # 2c. Squeeze-and-Excitation (SE) Blocks
        self.smooth_blocks = nn.ModuleList()
        for _ in C_channels:
            # Combine smoothing and SE into one sequential module
            self.smooth_blocks.append(nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
                SEBlock(fpn_channels, reduction=reduction) # SE Block after smoothing
            ))

        # ---------------- 3. OUTPUT HEAD ----------------
        self.final_upsample = nn.ConvTranspose2d(fpn_channels, fpn_channels, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(fpn_channels, out_channels, kernel_size=1)

    def _fpn_topdown(self, C_feats):
        """
        Build P5..P1 from C5..C1 (Top-Down Feature Pyramid).
        C_feats: [C1, C2, C3, C4, C5]
        Returns P_feats in order [P1, P2, P3, P4, P5].
        """
        lat = [l(c) for l, c in zip(self.lateral_convs, C_feats)]

        # Top-down Aggregation, list is currently [L1, L2, L3, L4, L5]
        # Start at P5
        P_list = [lat[-1]]  # P5

        # Iterate backward from L4 to L1 (i = 3 down to 0)
        for i in range(len(lat) - 2, -1, -1):
            L_i = lat[i]
            P_i_plus_1 = P_list[-1]
            
            P_up = F.interpolate(P_i_plus_1, size=L_i.shape[2:], mode='bilinear', align_corners=False)
            P_i = L_i + P_up
            P_list.append(P_i)
        
        # P_list is currently [P5, P4, P3, P2, P1]. Reverse it to [P1, P2, P3, P4, P5]
        P_list = P_list[::-1]
        
        # 3. Smooth & SE Blocks
        P = [smooth_se(p) for p, smooth_se in zip(P_list, self.smooth_blocks)]
        return P 

    def forward(self, x):
        # -------- 1. Encoder (C1..C5) --------
        skips = []
        out = x
        for down, pool in zip(self.downs, self.pools):
            out = down(out)
            skips.append(out)   # C1..C4
            out = pool(out)
        C5 = self.bottleneck(out)

        C_feats = skips + [C5]

        # -------- 2. FPN (P1..P5) - The Decoder Path --------
        P_feats = self._fpn_topdown(C_feats)
        P1 = P_feats[0]  # P1 is the final highest resolution feature map

        # -------- 3. Final Output Head --------
        
        # Upsample P1 to the original input resolution
        # D_final = self.final_upsample(P1)
        
        return self.final_conv(P1)