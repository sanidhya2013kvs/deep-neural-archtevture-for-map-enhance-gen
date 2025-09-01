import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------
# Bottleneck Block
# -------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.shortcut = shortcut and (c1 == c2)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.shortcut:
            y = y + x
        return F.relu(y)


# -------------------
# Self Attention Block
# -------------------
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, 1)
        self.key   = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, -1)   # B, C, HW
        k = self.key(x).view(B, C, -1)     # B, C, HW
        v = self.value(x).view(B, C, -1)   # B, C, HW

        attn = torch.bmm(q.permute(0, 2, 1), k)  # B, HW, HW
        attn = self.softmax(attn / (C ** 0.5))

        out = torch.bmm(v, attn.permute(0, 2, 1))  # B, C, HW
        out = out.view(B, C, H, W)
        return out + x  # residual


# -------------------
# Encoder-Decoder with Attention
# -------------------
class PyramidNet(nn.Module):
    def __init__(self, channels=[8,16,32,64,128,256,512,1024]):
        super().__init__()
        
        # Encoder: progressively up channels
        enc_layers = []
        in_c = 3
        for c in channels:
            enc_layers.append(nn.Conv2d(in_c, c, kernel_size=3, stride=2, padding=1))
            enc_layers.append(Bottleneck(c, c))
            in_c = c
        self.encoder = nn.Sequential(*enc_layers)

        # Attention at top
        self.attention = SelfAttention(channels[-1])

        # Decoder: reverse channels with deconv + bottleneck
        dec_layers = []
        rev_channels = channels[::-1]
        for i in range(len(rev_channels)-1):
            c1, c2 = rev_channels[i], rev_channels[i+1]
            dec_layers.append(nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1))
            dec_layers.append(Bottleneck(c2, c2))
        self.decoder = nn.Sequential(*dec_layers)

        # Final conv to 3 channels
        self.conv_out = nn.Conv2d(channels[0], 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        x = self.conv_out(x)
        return x


# -------------------
# Test
# -------------------
if __name__ == "__main__":
    model = PyramidNet()
    img = torch.randn(1, 3, 256, 256)  # RGB input
    out = model(img)
    print("Input:", img.shape)
    print("Output:", out.shape)
