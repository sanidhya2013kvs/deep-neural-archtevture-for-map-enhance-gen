
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
# Feed Forward Network (FFN)
# -------------------
class FeedForward(nn.Module):
    def __init__(self, dim, linear=True):
        super().__init__()
        if linear:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim*4),
                nn.ReLU(inplace=True),
                nn.Linear(dim*4, dim)
            )
        else:
            self.ffn = nn.Sequential(
                nn.Conv2d(dim, dim*4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim*4, dim, 1)
            )

    def forward(self, x):
        return self.ffn(x)


# -------------------
# Full Attention Block
# -------------------
class FullAttentionBlock(nn.Module):
    def __init__(self, dim, mode="linear", heads=8):
        super().__init__()
        self.mode = mode
        self.heads = heads
        self.dim = dim

        if mode == "linear":  # Transformer-style attention
            self.norm1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, dim*3)
            self.proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ffn = FeedForward(dim, linear=True)

        else:  # Conv-style attention
            self.q = nn.Conv2d(dim, dim, 1)
            self.k = nn.Conv2d(dim, dim, 1)
            self.v = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)
            self.ffn = FeedForward(dim, linear=False)

    def forward(self, x):
        if self.mode == "linear":
            # Flatten to sequence
            B, C, H, W = x.shape
            x_in = x
            x = x.flatten(2).transpose(1, 2)  # B, HW, C

            # Attention
            h = self.norm1(x)
            qkv = self.qkv(h).chunk(3, dim=-1)
            q, k, v = qkv
            q = q.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)
            k = k.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)
            v = v.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, -1, C)

            out = self.proj(out)
            x = x + out  # residual

            # FFN
            x = x + self.ffn(self.norm2(x))

            # Reshape back to 2D
            x = x.transpose(1, 2).reshape(B, C, H, W)
            return x + x_in

        else:  # Conv attention
            B, C, H, W = x.shape
            q = self.q(x).view(B, C, -1)  # B,C,HW
            k = self.k(x).view(B, C, -1)
            v = self.v(x).view(B, C, -1)

            attn = torch.bmm(q.permute(0, 2, 1), k)  # B,HW,HW
            attn = attn.softmax(dim=-1)
            out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)

            out = self.proj(out)
            x = x + out  # residual
            x = x + self.ffn(x)
            return x


# -------------------
# Pyramid Encoder-Decoder with Attention
# -------------------
class PyramidNet(nn.Module):
    def __init__(self, channels=[8,16,32,64,128,256,512,1024], attn_mode="linear"):
        super().__init__()

        # Encoder
        enc_layers = []
        in_c = 3
        for c in channels:
            enc_layers.append(nn.Conv2d(in_c, c, kernel_size=3, stride=2, padding=1))
            enc_layers.append(Bottleneck(c, c))
            in_c = c
        self.encoder = nn.Sequential(*enc_layers)

        # Full attention block
        self.attention = FullAttentionBlock(channels[-1], mode=attn_mode)

        # Decoder
        dec_layers = []
        rev_channels = channels[::-1]
        for i in range(len(rev_channels)-1):
            c1, c2 = rev_channels[i], rev_channels[i+1]
            dec_layers.append(nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1))
            dec_layers.append(Bottleneck(c2, c2))
        self.decoder = nn.Sequential(*dec_layers)

        # Final output conv
        self.conv_out = nn.Conv2d(channels[0], 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        return self.conv_out(x)


# -------------------
# Test
# -------------------
if __name__ == "__main__":
    model = PyramidNet(attn_mode="linear")  # try "conv" too
    img = torch.randn(1, 3, 256, 256)
    out = model(img)
    print("Input:", img.shape)
    print("Output:", out.shape)

