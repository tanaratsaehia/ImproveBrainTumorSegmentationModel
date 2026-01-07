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

class UNetSeDi(nn.Module):
    def __init__(self, in_channels=4, features=[64, 128, 256, 512],
                reduction=16, num_classes=None, dilations_rate= [1, 2, 1, 2]):
        super().__init__()

        if type(dilations_rate) is not list or len(dilations_rate) != 4:
            raise ValueError(f"Dilation '{dilations_rate}' rate not correct must be list and size 4 ex, [1,1,2,2]")
        
        self.model_name = "U-Net_SE_DI" + "".join(str(n) for n in dilations_rate)
        self.model_info = {
            'model_name': self.model_name,
            'dilation_rate': dilations_rate,
            'se_reduction': reduction,
            'description': "Squeeze and excitation add only decoder side of U-Net",
            'in_channel': in_channels, 
            'out_channel(class)': num_classes
            }
        
        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for f, di in zip(features, dilations_rate):
            self.downs.append(dilated_block(prev_ch, f, di))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_ch = f

        # Bottleneck
        self.bottleneck = conv_block(features[-1], features[-1] * 2)

        # Decoder
        self.ups = nn.ModuleList()
        self.se_modules = nn.ModuleList() # New: Squeeze-and-Excitation modules
        self.up_convs = nn.ModuleList()
        
        prev_ch = features[-1] * 2
        for f in reversed(features):
            # 1. upsample (prev_ch -> f)
            self.ups.append(nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2))
            
            # The channel dimension after concat is (f + f) = prev_ch
            # 2. SEBlock (applied after concatenation)
            # The input channels to SEBlock is (f + f)
            self.se_modules.append(SEBlock(channel=f * 2, reduction=reduction))
            
            # 3. conv after SE + concat (2*f -> f)
            self.up_convs.append(conv_block(f * 2, f))
            prev_ch = f # next input to upsample is the current output channel

        # final 1x1 conv
        self.final_conv = nn.Conv2d(prev_ch, num_classes, kernel_size=1)

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
        
        # SE modules are integrated into the decoding loop
        for up, se_mod, up_conv, skip in zip(self.ups, self.se_modules, self.up_convs, skip_connections):
            x = up(x)
            
            # in case of size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # concat along channel dim
            x = torch.cat((skip, x), dim=1)
            
            # Apply Squeeze-and-Excitation to the concatenated feature map
            x = se_mod(x)
            
            # Final convolution block for this stage
            x = up_conv(x)

        return self.final_conv(x)