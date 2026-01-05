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
                    fpn_channels=128):
        super().__init__()
        self.model_name = "FeaturePyramidU-Net"
        if num_classes is not None:
            out_channels = num_classes
        self.features = features
        self.fpn_channels = fpn_channels
        
        # Calculate C-channels: C1..C4, and C5 (bottleneck)
        C_channels = features + [features[-1] * 2]
        
        # ---------------- 1. ENCODER (Bottom-Up) ----------------
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for f in features:
            self.downs.append(conv_block(prev_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f
        
        # Bottleneck (C5) - same width as C4*2
        self.bottleneck = conv_block(features[-1], features[-1] * 2)

        # ---------------- 2. FPN (Top-Down Pathway - ACTS AS DECODER) ----------------
        
        # 2a. Lateral 1x1 convs (L1..L5) to unify channel count to fpn_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, kernel_size=1) for c in C_channels
        ])
        
        # 2b. Optional 3x3 "smooth" convs (P1..P5) after fusion
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1) for _ in C_channels
        ])

        # ---------------- 3. OUTPUT HEAD ----------------
        # The FPN produces P1, P2, P3, P4, P5. P1 is the highest resolution feature map (1/2 size of input).
        # We process P1 further and apply the final classification layer.

        # The paper suggests a final processing block and an upsampling/conv layer.
        # We will use a final ConvTranspose to upscale P1 back to the input resolution, 
        # followed by the classification conv.
        
        # We need to process the final highest resolution P1 feature map.
        # self.final_upsample = nn.ConvTranspose2d(fpn_channels, fpn_channels, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(fpn_channels, out_channels, kernel_size=1)

    def _fpn_topdown(self, C_feats):
        """
        Build P5..P1 from C5..C1 (Top-Down Feature Pyramid).
        C_feats: [C1, C2, C3, C4, C5]
        Returns P_feats in reverse order [P5, P4, P3, P2, P1] for easy iteration.
        """
        # 1. Lateral Convolutions: L = [L1, L2, L3, L4, L5]
        lat = [l(c) for l, c in zip(self.lateral_convs, C_feats)]

        # 2. Top-down Aggregation
        # Start at P5 = L5
        P_list = [lat[-1]]  # P5

        # Iterate backward from L4 to L1 (i = 3 down to 0)
        for i in range(len(lat) - 2, -1, -1):
            L_i = lat[i]
            P_i_plus_1 = P_list[-1]
            
            # Upsample P_{i+1} to match spatial size of L_i
            P_up = F.interpolate(P_i_plus_1, size=L_i.shape[2:], mode='bilinear', align_corners=False)
            
            # P_i = L_i + P_up (Summation is the core FPN operation)
            P_i = L_i + P_up
            P_list.append(P_i)
        
        # P_list is currently [P5, P4, P3, P2, P1]. Reverse it to [P1, P2, P3, P4, P5]
        P_list = P_list[::-1]
        
        # 3. Smooth Convolutions (P1..P5)
        P = [smooth(p) for p, smooth in zip(P_list, self.smooth_convs)]
        return P # [P1, P2, P3, P4, P5]

    def forward(self, x):
        # -------- Encoder: collect C1..C4, then C5 from bottleneck --------
        skips = []
        out = x
        for down, pool in zip(self.downs, self.pools):
            out = down(out)
            skips.append(out)   # C1..C4
            out = pool(out)
        C5 = self.bottleneck(out)  # bottleneck

        # Align shapes list: [C1, C2, C3, C4, C5]
        C_feats = skips + [C5]

        # -------- FPN: build pyramids P1..P5 (This is the DECODER) --------
        P_feats = self._fpn_topdown(C_feats)
        P1 = P_feats[0]  # P1 is the highest resolution map

        # -------- Final Output Head --------
        
        # Upsample P1 (which is 1/2 the input resolution) to the original input resolution
        # D_final = self.final_upsample(P1)
        
        # Apply the final classification 1x1 convolution
        return self.final_conv(P1)