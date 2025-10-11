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

class UNetFPN(nn.Module):
    """
    U-Net with a Feature Pyramid Network (FPN) on top of the encoder.
    The decoder uses FPN pyramid maps (Pk) as the skip features to improve small-object segmentation.

    Args:
        in_channels: input channels
        out_channels: number of classes
        features: encoder widths (like your original)
        fpn_channels: width of the FPN lateral/smooth channels (128 or 256 are common)
        use_concat: if True, concat [upsampled decoder, Pk] (more capacity).
                    if False, just feed Pk (leaner).
    """
    def __init__(self, in_channels=4, out_channels=2, num_classes=None, features=[64,128,256,512],
                    fpn_channels=128, use_concat=True):
        super().__init__()
        self.model_name = "U-Net_FeaturePyramid"
        if num_classes is not None:
            out_channels = num_classes
        self.use_concat = use_concat
        self.features = features
        # ---------------- Encoder ----------------
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for f in features:
            self.downs.append(conv_block(prev_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f

        # Bottleneck (C5)
        self.bottleneck = conv_block(features[-1], features[-1] * 2)  # 1024 if features[-1]==512

        # ---------------- FPN (laterals + top-down + smooth) ----------------
        # Encoder stage outputs: C1..C4 (after each down block), C5 (bottleneck)
        # Build lateral 1x1 to fpn_channels for each C1..C5
        C_channels = [features[0], features[1], features[2], features[3], features[-1]*2]
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, kernel_size=1) for c in C_channels
        ])
        # Optional 3x3 "smooth" convs to reduce aliasing after top-down sums
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1) for _ in C_channels
        ])

        # ---------------- Decoder ----------------
        # We will do 4 up steps. At each step k, we upsample decoder from level k+1 and fuse with Pk.
        # The transpose-conv only transforms channels (and doubles spatial size).
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev_dec_ch = features[-1] * 2  # start from bottleneck channels

        # We match the original decoder targets: 512 -> 256 -> 128 -> 64 (if features as given)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev_dec_ch, f, kernel_size=2, stride=2))
            if self.use_concat:
                in_ch = f + fpn_channels   # concat [decoder_up, Pk]
            else:
                in_ch = fpn_channels       # replace skip with Pk only
            self.up_convs.append(conv_block(in_ch, f))
            prev_dec_ch = f

        # Final head
        self.final_conv = nn.Conv2d(prev_dec_ch, out_channels, kernel_size=1)

    def _topdown_fpn(self, C_feats):
        """
        Build P5..P1 from C5..C1.
        C_feats: [C1, C2, C3, C4, C5]
        Returns P_feats in same order [P1, P2, P3, P4, P5]
        """
        # Laterals
        lat = [l(c) for l, c in zip(self.lateral_convs, C_feats)]  # [L1..L5]

        # Top-down: start at the top (P5 = L5), then add upsampled higher-level to lower-level lateral
        P5 = lat[4]
        P4 = lat[3] + F.interpolate(P5, size=lat[3].shape[2:], mode='bilinear', align_corners=False)
        P3 = lat[2] + F.interpolate(P4, size=lat[2].shape[2:], mode='bilinear', align_corners=False)
        P2 = lat[1] + F.interpolate(P3, size=lat[1].shape[2:], mode='bilinear', align_corners=False)
        P1 = lat[0] + F.interpolate(P2, size=lat[0].shape[2:], mode='bilinear', align_corners=False)

        # Smooth (optional but usually helpful)
        P = [P1, P2, P3, P4, P5]
        P = [smooth(p) for p, smooth in zip(P, self.smooth_convs)]
        return P  # [P1..P5]

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

        # -------- FPN: build pyramids P1..P5 --------
        P1, P2, P3, P4, P5 = self._topdown_fpn(C_feats)

        # -------- Decoder: fuse Pk at each scale --------
        dec = C5  # start decoding from bottleneck
        P_list = [P4, P3, P2, P1]  # match 4 up-steps (same spatial sizes as features[::-1])
        for up, up_conv, Pk in zip(self.ups, self.up_convs, P_list):
            dec = up(dec)
            # ensure spatial match (robustness)
            if dec.shape[2:] != Pk.shape[2:]:
                dec = F.interpolate(dec, size=Pk.shape[2:], mode='bilinear', align_corners=False)
            if self.use_concat:
                dec = torch.cat([dec, Pk], dim=1)
            else:
                dec = Pk
            dec = up_conv(dec)

        return self.final_conv(dec)