import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=(2, 2)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(0.25)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = out + self.skip(x)  # residual skip connection
        return self.dropout(self.pool(out))


class MusicGenreCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, pool_size=(2, 2)),      # output: (B, 32, 64, 64)
            ConvBlock(32, 64, pool_size=(2, 2)),     # output: (B, 64, 32, 32)
            ConvBlock(64, 128, pool_size=(2, 2)),    # output: (B, 128, 16, 16)
            ConvBlock(128, 256, pool_size=(2, 2)),   # output: (B, 256, 8, 8)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)           # output: (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)
