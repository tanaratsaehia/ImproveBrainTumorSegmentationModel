import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Utility Blocks (Copied from original) ---

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
        # Squeeze (Global Information Embedding)
        y = self.avg_pool(x).view(b, c)
        # Excitation (Activation and Weighting)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale (Channel Recalibration)
        return x * y.expand_as(x)

# --- New Model Definition ---

class UNetBiPyramidSE(nn.Module):
    """
    U-Net with Bi-Directional Pyramid (FPN+Path Aggregation) and Squeeze-and-Excitation (SE) 
    Blocks on the Decoder path for channel refinement.
    """
    def __init__(self, in_channels=4, out_channels=2, num_classes=None, features=[64, 128, 256, 512],
                    pyramid_channels=128, reduction=16):
        super().__init__()
        self.model_name = "UNet_BiPyramid_SE"
        if num_classes is not None:
            out_channels = num_classes
        self.features = features
        self.pyramid_channels = pyramid_channels
        
        # Channels for C1..C4, C5 (bottleneck)
        C_channels = [features[0], features[1], features[2], features[3], features[-1]*2]
        
        # ---------------- 1. ENCODER (Bottom-Up - Dilation) ----------------
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for i, f in enumerate(features):
            # Using Dilated block
            self.downs.append(dilated_block(prev_ch, f, dilation=2**(i+1))) # 2 4 8 16
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f
        self.bottleneck = conv_block(features[-1], features[-1] * 2)

        # ---------------- 2. FPN (Top-Down Aggregation for Encoder) ----------------
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, pyramid_channels, kernel_size=1) for c in C_channels
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1) for _ in C_channels
        ])

        # ---------------- 3. DECODER (Bottom-Up Path Aggregation + SE) ----------------
        
        # 3a. Initial Conv (D5 from P5) - start the decoder path
        self.dec_conv_p5 = conv_block(pyramid_channels, pyramid_channels)
        self.se_p5 = SEBlock(pyramid_channels, reduction=reduction) # SE for D5
        
        # 3b. Bottom-up aggregation blocks (D4 from P4+D5_up, D3 from P3+D4_up, etc.)
        self.dec_convs = nn.ModuleList()
        self.se_modules = nn.ModuleList() # NEW: SE modules for D4, D3, D2, D1
        
        # Aggregation is P4 -> P1 (4 steps)
        for _ in range(4):
            # Conv block to process the aggregated feature (D_i)
            self.dec_convs.append(conv_block(pyramid_channels, pyramid_channels))
            # SE block to refine the aggregated feature D_i
            self.se_modules.append(SEBlock(pyramid_channels, reduction=reduction))

        # ---------------- 4. OUTPUT HEAD ----------------
        self.final_upsample = nn.ConvTranspose2d(pyramid_channels, pyramid_channels, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(pyramid_channels, out_channels, kernel_size=1)


    def _fpn_topdown(self, C_feats):
        """Build P5..P1 from C5..C1 (Top-Down Feature Pyramid)."""
        lat = [l(c) for l, c in zip(self.lateral_convs, C_feats)]
        P5 = lat[4]
        P4 = lat[3] + F.interpolate(P5, size=lat[3].shape[2:], mode='bilinear', align_corners=False)
        P3 = lat[2] + F.interpolate(P4, size=lat[2].shape[2:], mode='bilinear', align_corners=False)
        P2 = lat[1] + F.interpolate(P3, size=lat[1].shape[2:], mode='bilinear', align_corners=False)
        P1 = lat[0] + F.interpolate(P2, size=lat[0].shape[2:], mode='bilinear', align_corners=False)
        P = [P1, P2, P3, P4, P5]
        P = [smooth(p) for p, smooth in zip(P, self.smooth_convs)]
        return P


    def forward(self, x):
        # 1. ENCODER (Bottom-Up)
        skips = []
        out = x
        for down, pool in zip(self.downs, self.pools):
            out = down(out)
            skips.append(out)   # C1..C4
            out = pool(out)
        C5 = self.bottleneck(out)
        C_feats = skips + [C5]

        # 2. FPN (Top-Down) -> Encoder side pyramid
        P_feats = self._fpn_topdown(C_feats)
        P1, P2, P3, P4, P5 = P_feats

        # 3. DECODER (Bottom-Up Path Aggregation + SE)
        
        # D5: Initial Decoder feature from P5
        D5 = self.dec_conv_p5(P5)
        D5 = self.se_p5(D5) # SE on D5
        
        # P_list is [P4, P3, P2, P1]
        P_list = [P4, P3, P2, P1]
        D_current = D5
        
        for i, Pk in enumerate(P_list):
            # Upsample the previous decoder feature
            D_up = F.interpolate(D_current, size=Pk.shape[2:], mode='bilinear', align_corners=False)
            
            # Aggregate: D_i = P_i + D_{i+1}_up
            D_fused = Pk + D_up
            
            # Process with Conv Block to create the new D_i
            D_current = self.dec_convs[i](D_fused)
            
            # Apply SE Block for channel-wise refinement
            D_current = self.se_modules[i](D_current)

        # D_current is now D1 (the highest resolution decoder feature)
        D1 = D_current
        
        # 4. Final Output Head
        if D1.shape[2:] != x.shape[2:]:
            D1 = self.final_upsample(D1)
        
        return self.final_conv(D1)