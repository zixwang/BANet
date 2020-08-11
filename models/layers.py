import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            # nn.Dropout2d(0.2),
        )

    def forward(self, x):
        return self.layer(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i*growth_rate, growth_rate)
                                        for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            outs = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], dim=1)
                outs.append(out)
            return torch.cat(outs, dim=1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], dim=1)
            return x


class TransitionDown(nn.Module):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            # nn.Dropout2d(0.2),
        )
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.layer(x)
        x = self.max_pool(x)
        return x


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, 3, 2)

    def forward(self, x, skip):
        out = self.conv_trans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.bottleneck = DenseBlock(in_channels, growth_rate, n_layers, upsample=True)

    def forward(self, x):
        return self.bottleneck(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]
