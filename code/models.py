"""
Created on June 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pad3d


class AttentionGrid(nn.Module):
    """
    Attention Grid module for 3D convolutions.

    Args:
        x_c (int): Number of channels in the input tensor.
        g_c (int): Number of channels in the gating signal.
        i_c (int): Number of channels for the intermediate computations.
        stride (int, optional): Stride for the input filter convolution. Default is 2.
        mode (str, optional): Mode for interpolation. Default is 'trilinear'.
    """

    def __init__(self, x_c, g_c, i_c, stride=2, mode='trilinear'):
        super(AttentionGrid, self).__init__()
        self.input_filter = nn.Conv3d(
            in_channels=x_c, out_channels=i_c, kernel_size=1, stride=stride, bias=False)
        self.gate_filter = nn.Conv3d(
            in_channels=g_c, out_channels=i_c, kernel_size=1, stride=1, bias=True)
        self.psi = nn.Conv3d(in_channels=i_c, out_channels=1,
                             kernel_size=1, stride=1, bias=True)
        self.bnorm = nn.InstanceNorm3d(i_c)
        self.mode = mode

    def forward(self, x, g):
        x_shape = x.shape
        a = self.input_filter(x)
        b = self.gate_filter(g)

        # Padding to match shapes
        if a.shape[-1] < b.shape[-1]:
            a = pad3d(a, b)
        elif a.shape[-1] > b.shape[-1]:
            b = pad3d(b, a)

        w = torch.sigmoid(self.psi(F.relu(a + b)))
        w = F.interpolate(w, size=x_shape[2:], mode=self.mode)

        y = x * w
        y = self.bnorm(y)
        return y, w


class Block(nn.Module):
    """
    Block module for the U-Net architecture with optional dropout and InstanceNorm3D.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        hid_c (int, optional): Number of hidden channels. If not provided, uses a simpler block.
        final_layer (bool, optional): If true, disables pooling in the block. Default is False.
        dropout_rate (float, optional): Dropout rate. Default is 0.3.
    """

    def __init__(self, in_c, out_c, hid_c=None, final_layer=False, dropout_rate=0.3):
        super(Block, self).__init__()
        if hid_c is None:
            self.layer = nn.Sequential(
                nn.Conv3d(in_channels=in_c, out_channels=out_c,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(out_c)
            )

            self.out_block = nn.Sequential(
                nn.Conv3d(in_channels=out_c, out_channels=out_c,
                          kernel_size=2, padding=0),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(out_c)
            )

            self.pool = not final_layer
            if self.pool:
                self.pool_block = nn.Sequential(
                    nn.Conv3d(in_channels=out_c, out_channels=out_c,
                              kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Dropout3d(dropout_rate),
                    nn.InstanceNorm3d(out_c)
                )

        else:
            self.pool = False

            self.layer = nn.Sequential(
                nn.Conv3d(in_channels=in_c, out_channels=hid_c,
                          kernel_size=3, padding=0),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(hid_c)
            )

            self.out_block = nn.Sequential(
                nn.Conv3d(in_channels=hid_c, out_channels=hid_c,
                          kernel_size=2, padding=0),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(hid_c),
                nn.ConvTranspose3d(
                    in_channels=hid_c, out_channels=out_c, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(out_c)
            )

    def forward(self, x):
        y = self.layer(x)
        y = self.out_block(y)

        if self.pool:
            y_ = self.pool_block(y)
            return y_, y
        else:
            return y


class PatchEmbed3D(nn.Module):
    """
    Patch Embedding for 3D images.

    Args:
        img_size (int): Size of the input image.
        patch_size (int): Size of the patch.
        in_c (int, optional): Number of input channels. Default is 1.
        embed_dim (int, optional): Dimension of the embedding. Default is 512.
    """

    def __init__(self, img_size, patch_size, in_c=1, embed_dim=512):
        super(PatchEmbed3D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 3
        self.proj = nn.Conv3d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    Multi-head Self-Attention mechanism.

    Args:
        dim (int): Input dimension.
        n_heads (int, optional): Number of attention heads. Default is 8.
        qkv_bias (bool, optional): Whether to use bias for QKV linear layers. Default is True.
        attn_p (float, optional): Dropout probability for attention. Default is 0.
        proj_p (float, optional): Dropout probability for projection. Default is 0.
    """

    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_p=0., proj_p=0.):
        super(Attention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_toks, dim = x.shape
        if dim != self.dim:
            raise ValueError(
                f"Expected input dimension {self.dim}, but got {dim}")

        qkv = self.qkv(x).reshape(n_samples, n_toks, 3,
                                  self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        dp = (q @ k_t) * self.scale
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        w_av = attn @ v
        w_av = w_av.transpose(1, 2).flatten(2)
        x = self.proj(w_av)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with GELU activation.

    Args:
        in_features (int): Input dimension.
        hidden_features (int): Hidden layer dimension.
        out_features (int): Output dimension.
        p (float, optional): Dropout probability. Default is 0.
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.layers(x)


class Transformer(nn.Module):
    """
    Transformer block with LayerNorm, Multi-head Attention and MLP.

    Args:
        dim (int): Input dimension.
        n_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio for MLP hidden layer dimension. Default is 4.0.
        qkv_bias (bool, optional): Whether to use bias for QKV linear layers. Default is True.
        p (float, optional): Dropout probability. Default is 0.
        attn_p (float, optional): Dropout probability for attention. Default is 0.
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super(Transformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1E-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm2 = nn.LayerNorm(dim, eps=1E-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features, dim, p)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer3D(nn.Module):
    """
    Vision Transformer for 3D images.

    Args:
        img_size (int, optional): Size of the input image. Default is 128.
        patch_size (int, optional): Size of the patch. Default is 16.
        in_c (int, optional): Number of input channels. Default is 1.
        n_classes (int, optional): Number of output classes. Default is 1.
        embed_dim (int, optional): Dimension of the embedding. Default is 512.
        depth (int, optional): Number of Transformer layers. Default is 8.
        n_heads (int, optional): Number of attention heads. Default is 8.
        mlp_ratio (float, optional): Ratio for MLP hidden layer dimension. Default is 4.0.
        qkv_bias (bool, optional): Whether to use bias for QKV linear layers. Default is True.
        dropout (float, optional): Dropout probability. Default is 0.
    """

    def __init__(self, img_size=128, patch_size=16, in_c=1, n_classes=1, embed_dim=512, depth=8, n_heads=8, mlp_ratio=4., qkv_bias=True, dropout=0.):
        super(VisionTransformer3D, self).__init__()
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, in_c=in_c, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.transformers = nn.ModuleList([
            Transformer(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=dropout, attn_p=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1E-6)
        self.head = nn.Linear(embed_dim, n_classes)
        self.prob_dist = nn.Softmax(dim=-1)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for transformer in self.transformers:
            x = transformer(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        out = self.prob_dist(x)
        return out


class Attention_UNetT(nn.Module):
    """
    Attention U-Net with optional Vision Transformer in the latent layer.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        n (int, optional): Scale factor for the number of channels. Default is 1.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
        vision_transformer (bool, optional): Whether to use Vision Transformer in the latent layer. Default is False.
    """

    def __init__(self, in_c, out_c, n=1, dropout_rate=0.1, vision_transformer=False):
        super(Attention_UNetT, self).__init__()
        self.out_c = out_c
        self.vis_transform = vision_transformer

        self.encoder_layers = nn.ModuleList([
            Block(in_c=in_c, out_c=int(64/n), dropout_rate=dropout_rate),
            Block(in_c=int(64/n), out_c=int(128/n), dropout_rate=dropout_rate),
            Block(in_c=int(128/n), out_c=int(256/n),
                  dropout_rate=dropout_rate),
            Block(in_c=int(256/n), out_c=int(512/n), dropout_rate=dropout_rate)
        ])

        if self.vis_transform:
            self.latent_layer = nn.ModuleList([
                VisionTransformer3D(
                    img_size=14,
                    patch_size=7,
                    in_c=int(512/n),
                    n_classes=int(512/n) * 10648,
                    embed_dim=512,
                    depth=8,
                    n_heads=8,
                    mlp_ratio=4,
                    qkv_bias=False,
                    dropout=dropout_rate
                )
            ])
        else:
            self.latent_layer = nn.ModuleList([
                Block(in_c=int(512/n), out_c=int(512/n),
                      hid_c=int(1024/n), dropout_rate=dropout_rate)
            ])

        self.decoder_layers = nn.ModuleList([
            Block(in_c=int(1024/n), out_c=int(256/n),
                  hid_c=int(512/n), dropout_rate=dropout_rate),
            Block(in_c=int(512/n), out_c=int(128/n),
                  hid_c=int(256/n), dropout_rate=dropout_rate),
            Block(in_c=int(256/n), out_c=int(64/n),
                  hid_c=int(128/n), dropout_rate=dropout_rate),
            Block(in_c=int(128/n), out_c=int(64/n),
                  final_layer=True, dropout_rate=dropout_rate)
        ])

        self.skip_layers = nn.ModuleList([
            AttentionGrid(int(512/n), int(512/n), int(512/n)),
            AttentionGrid(int(256/n), int(256/n), int(256/n)),
            AttentionGrid(int(128/n), int(128/n), int(128/n)),
            AttentionGrid(int(64/n), int(64/n), int(64/n))
        ])

        self.out = nn.Conv3d(in_channels=int(
            64/n), out_channels=out_c, kernel_size=1)

    def forward(self, x, gans=False):
        target_shape = x.shape[2:]

        y = pad3d(x.float(), 240)

        encoder_outputs = []
        for layer in self.encoder_layers:
            y, y_out = layer(y)
            encoder_outputs.append(y_out)

        for layer in self.latent_layer:
            if self.vis_transform:
                b, c, _, _, _ = y.shape
                y = layer(y).view(b, c, 22, 22, 22)
            else:
                y = layer(y)

        if gans:
            y += torch.rand_like(y, device=y.device)

        for layer, skip, encoder_output in zip(self.decoder_layers, self.skip_layers, encoder_outputs[::-1]):
            y = layer(torch.cat((skip(encoder_output, y)[
                      0], pad3d(y, encoder_output)), dim=1))

        y = self.out(y)
        y = pad3d(y, target_shape)
        return y

    def freeze_encoder(self):
        for layer in self.encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in self.skip_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.latent_layer.parameters():
            param.requires_grad = False


class Critic(nn.Module):
    """
    Critic model for adversarial training.

    Args:
        in_c (int, optional): Number of input channels. Default is 1.
        fact (int, optional): Scaling factor for the number of channels. Default is 1.
    """

    def __init__(self, in_c=1, fact=1):
        super(Critic, self).__init__()
        self.c = fact

        self.E = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(in_c, self.c * 16, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv3d(self.c * 16, self.c * 32, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv3d(self.c * 32, self.c * 64, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(nn.Linear(self.c * 64, 1))

    def forward(self, x):
        target_shape = x.shape[2:]
        y = pad3d(x.float(), max(target_shape))
        y = self.E(y)
        y = self.fc(y.squeeze())
        return y


class Critic_VT(nn.Module):
    """
    Vision Transformer-based Critic model for adversarial training.

    Args:
        in_c (int, optional): Number of input channels. Default is 1.
        fact (int, optional): Scaling factor for the number of channels. Default is 1.
    """

    def __init__(self, in_c=1, fact=1):
        super(Critic_VT, self).__init__()
        self.c = fact

        self.E = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(in_c, self.c * 16, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = VisionTransformer3D(
            img_size=48,
            patch_size=12,
            in_c=self.c * 16,
            n_classes=1,
            embed_dim=512,
            depth=8,
            n_heads=8,
            mlp_ratio=4,
            qkv_bias=False,
            dropout=0.
        )

    def forward(self, x):
        target_shape = x.shape[2:]
        y = pad3d(x.float(), max(target_shape))
        y = self.E(y)
        y = self.fc(y)
        return y.view(x.shape[0])


def test_model(device='cpu', B=1, emb=1, ic=1, oc=1, n=64):
    """
    Test function to instantiate and test the models.

    Args:
        device (str, optional): Device to run the models on. Default is 'cpu'.
        B (int, optional): Batch size. Default is 1.
        emb (int, optional): Embedding dimension. Default is 1.
        ic (int, optional): Input channels. Default is 1.
        oc (int, optional): Output channels. Default is 1.
        n (int, optional): Scaling factor for channels. Default is 64.
    """
    a = torch.ones((B, ic, 240, 240, 155), device=device)

    model = Attention_UNetT(in_c=ic, out_c=oc, n=n,
                            vision_transformer=True).to(device)
    model.freeze_encoder()
    critic1 = Critic(in_c=ic, fact=1).to(device)
    critic2 = Critic_VT(in_c=ic, fact=1).to(device)

    b = model(a, gans=True)
    c = critic1(b)
    d = critic2(b)

    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)


if __name__ == '__main__':
    test_model('cpu', B=1, n=64)
