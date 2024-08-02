import torch
import torch.nn as nn
from tqdm import trange
from matplotlib import pyplot as plt

from dataloader import DataLoader
from utils import pad3d, get_fft, reconst_image, show_images


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
        return x


class Transformer_Model(nn.Module):
    def __init__(self, depth=64, dropout_rate=0.2, img_size=33):
        super(Transformer_Model, self).__init__()
        self.img_size = img_size
        self.layer = nn.ModuleList([
            VisionTransformer3D(
                img_size=img_size,
                patch_size=11,
                in_c=4,
                n_classes=2*240*240*121,  # 2*img_size*(img_size-1)**2,
                embed_dim=48,
                depth=depth,
                n_heads=16,
                mlp_ratio=8,
                qkv_bias=False,
                dropout=dropout_rate
            )
        ])

    def forward(self, x):
        y = pad3d(x, self.img_size)

        for layer in self.layer:
            y = layer(y)

        return y.view((x.shape[0], 2, 240, 240, 121))


if __name__ == '__main__':
    loader = DataLoader(augment=True, batch=1, workers=4,
                        path='/home/agam/Desktop/brats_2024_local_impainting/TrainingData/')
    model = Transformer_Model(dropout_rate=0.1, depth=64).cuda()
    criterion = [nn.MSELoss(), nn.L1Loss()]
    optimizer = torch.optim.AdamW(model.parameters(), 1E-4)
    errors = []
    accumulated_error = []
    iterations = loader.max_id
    epochs = 1000

    for eps in range(epochs):
        print(f'\nEpoch: {eps}\n')
        for i in trange(iterations):
            optimizer.zero_grad()
            x, xm = loader.load_batch()
            x = pad3d(x, 240).cuda()
            xm = pad3d(xm, 240).cuda()
            x_ = x * (xm < 0.5)
            mp = torch.cat(
                (get_fft(x_), get_fft(xm)), dim=1)

            y = model(mp)

            x_reconstruct = reconst_image(y)

            error = criterion[0](get_fft(x, cropped=240), y) + \
                criterion[1](x, reconst_image(y))

            error.backward()
            optimizer.step()

            accumulated_error.append(error.item())

            if i % 100 == 0:
                show_images(torch.cat([x, reconst_image(get_fft(
                    x)), x_reconstruct], dim=0).detach().cpu(), 3, dpi=250)

        errors.append(sum(accumulated_error) / len(accumulated_error))
        accumulated_error = []
        plt.plot(errors)
        plt.title("Average L1 Error")
        plt.show()
