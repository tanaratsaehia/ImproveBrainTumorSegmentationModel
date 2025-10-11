import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNetBiPyramid(nn.Module):
    """
    U-Net with a Bi-Directional Feature Pyramid (FPN for Encoder skips + Path Aggregation for Decoder).
    FPN (Encoder): Generates P-features (P1..P5) from C-features (C1..C5).
    Decoder (Path Aggregation): Uses a bottom-up path to aggregate the P-features.
    """
    def __init__(self, in_channels=4, out_channels=2, num_classes=None, features=[64, 128, 256, 512],
                    pyramid_channels=128):
        super().__init__()
        self.model_name = "U-Net_BiPyramid"
        if num_classes is not None:
            out_channels = num_classes
        self.features = features
        self.pyramid_channels = pyramid_channels
        C_channels = [features[0], features[1], features[2], features[3], features[-1]*2]
        
        # ---------------- 1. ENCODER (Bottom-Up) ----------------
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for f in features:
            self.downs.append(conv_block(prev_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f
        self.bottleneck = conv_block(features[-1], features[-1] * 2)

        # ---------------- 2. FPN (Top-Down Aggregation for Encoder) ----------------
        # Lateral 1x1 convs (L1..L5)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, pyramid_channels, kernel_size=1) for c in C_channels
        ])
        # Smooth 3x3 convs (P1..P5)
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1) for _ in C_channels
        ])

        # ---------------- 3. DECODER (Bottom-Up Path Aggregation - the "second pyramid") ----------------
        
        # We start with P5 (the largest feature map, but smallest in spatial size) and aggregate upwards.
        # This replaces the standard UNet upsampling + concatenation decoder.
        
        # 3a. Initial Conv (D5 from P5) - start the decoder path
        self.dec_conv_p5 = conv_block(pyramid_channels, pyramid_channels)
        
        # 3b. Bottom-up aggregation blocks (D4 from P4+D5_up, D3 from P3+D4_up, etc.)
        self.dec_convs = nn.ModuleList()
        self.dec_pools = nn.ModuleList()
        # Aggregation is P4 -> P1 (4 steps)
        for _ in range(4):
            # 3x3 pool to downsample (PAG)
            self.dec_pools.append(nn.MaxPool2d(2)) 
            # Conv block to process the aggregated feature
            self.dec_convs.append(conv_block(pyramid_channels, pyramid_channels))

        # ---------------- 4. OUTPUT HEAD ----------------
        # Final upsampling block to match the largest feature map size (P1) and produce final segmentation map.
        self.final_upsample = nn.ConvTranspose2d(pyramid_channels, pyramid_channels, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(pyramid_channels, out_channels, kernel_size=1)


    def _fpn_topdown(self, C_feats):
        """Build P5..P1 from C5..C1 (Top-Down Feature Pyramid)."""
        lat = [l(c) for l, c in zip(self.lateral_convs, C_feats)]  # [L1..L5]

        # Top-down fusion
        P5 = lat[4]
        P4 = lat[3] + F.interpolate(P5, size=lat[3].shape[2:], mode='bilinear', align_corners=False)
        P3 = lat[2] + F.interpolate(P4, size=lat[2].shape[2:], mode='bilinear', align_corners=False)
        P2 = lat[1] + F.interpolate(P3, size=lat[1].shape[2:], mode='bilinear', align_corners=False)
        P1 = lat[0] + F.interpolate(P2, size=lat[0].shape[2:], mode='bilinear', align_corners=False)

        # Smooth (optional but usually helpful)
        P = [P1, P2, P3, P4, P5]
        P = [smooth(p) for p, smooth in zip(P, self.smooth_convs)]
        return P # [P1, P2, P3, P4, P5]


    def forward(self, x):
        # 1. ENCODER (Bottom-Up)
        skips = []
        out = x
        for down, pool in zip(self.downs, self.pools):
            out = down(out)
            skips.append(out)   # C1..C4
            out = pool(out)
        C5 = self.bottleneck(out)  # bottleneck
        C_feats = skips + [C5]

        # 2. FPN (Top-Down) -> Encoder side pyramid
        P_feats = self._fpn_topdown(C_feats)
        P1, P2, P3, P4, P5 = P_feats

        # 3. DECODER (Bottom-Up Path Aggregation) -> Decoder side pyramid
        # We start the decoder path D from the top FPN feature P5
        D5 = self.dec_conv_p5(P5)
        
        # D4 is aggregated from P4 and D5 (upsampled)
        D5_up = F.interpolate(D5, size=P4.shape[2:], mode='bilinear', align_corners=False)
        D4 = self.dec_convs[0](P4 + D5_up) # Fused D4 = P4 + D5_up
        
        # D3 is aggregated from P3 and D4 (upsampled)
        D4_up = F.interpolate(D4, size=P3.shape[2:], mode='bilinear', align_corners=False)
        D3 = self.dec_convs[1](P3 + D4_up) # Fused D3 = P3 + D4_up

        # D2 is aggregated from P2 and D3 (upsampled)
        D3_up = F.interpolate(D3, size=P2.shape[2:], mode='bilinear', align_corners=False)
        D2 = self.dec_convs[2](P2 + D3_up) # Fused D2 = P2 + D3_up
        
        # D1 is aggregated from P1 and D2 (upsampled)
        D2_up = F.interpolate(D2, size=P1.shape[2:], mode='bilinear', align_corners=False)
        D1 = self.dec_convs[3](P1 + D2_up) # Fused D1 = P1 + D2_up
        
        # 4. Final Output Head (D1 is the final, high-resolution feature map)
        
        # Final upsampling (if P1 is not same size as input x)
        # Assuming P1 is 1/2 size of original input x, we need one final upsample.
        # This part needs adjustment based on the input size/feature list: 
        # C1 (P1) is usually 1/2 of input size. 
        # The line below handles the final upscale to the input size.
        if D1.shape[2:] != x.shape[2:]:
            D1 = self.final_upsample(D1)
        
        return self.final_conv(D1)