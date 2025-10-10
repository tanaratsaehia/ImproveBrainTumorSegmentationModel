import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, num_classes=None, features=[64,128,256,512]):
        super().__init__()
        if num_classes is not None:
            out_channels = num_classes
        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for f in features:
            self.downs.append(conv_block(prev_ch, f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_ch = f

        # Bottleneck
        self.bottleneck = conv_block(features[-1], features[-1] * 2)

        # Decoder
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev_ch = features[-1] * 2
        for f in reversed(features):
            # upsample
            self.ups.append(nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2))
            # conv after concat (2*f -> f)
            self.up_convs.append(conv_block(prev_ch, f))
            prev_ch = f

        # final 1x1 conv
        self.final_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # Encoder path
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for up, up_conv, skip in zip(self.ups, self.up_convs, skip_connections):
            x = up(x)
            # in case of size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            # concat along channel dim
            x = torch.cat((skip, x), dim=1)
            x = up_conv(x)

        return self.final_conv(x)