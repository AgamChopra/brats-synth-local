"""
Created on July 2024
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
"""
import torch
import torch.nn as nn
import time

from utils import count_parameters, test_model_memory_usage


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, in_c=1, embed_dim=512):
        super(PatchEmbed3D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.n_patches = (img_size // patch_size) ** 3
        self.proj = nn.Conv3d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class LinearAttention(nn.Module):
    '''
    @Ref:
        https://arxiv.org/pdf/2305.07239
        https://github.com/dengyecode/T-former_image_inpainting/blob/master/model/network.py#L239
    '''

    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_p=0., proj_p=0.):
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkvg = nn.Linear(dim, dim * 4, bias=qkv_bias)

        self.gelu = nn.GELU()
        self.softplus = nn.Softplus()

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_toks, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected input dimension {
                             self.dim}, but got {dim}")

        qkvg = self.qkvg(x).reshape(n_samples, n_toks, 4,
                                    self.n_heads,
                                    self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v, g = self.softplus(qkvg[0]), self.softplus(
            qkvg[1]), self.gelu(qkvg[2]), self.gelu(qkvg[3])

        kv = torch.matmul(k, v.transpose(-2, -1))
        z = torch.einsum('bhcn,bhc -> bhn', q, k.sum(dim=-1)) * self.scale
        z = 1.0 / (z + self.dim)
        out = torch.einsum('bhcn, bhcd -> bhdn', q, kv) * self.scale
        out = ((out + v) * z.unsqueeze(2)).reshape(n_samples, n_toks, dim)
        out *= g.reshape(out.shape)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MLP(nn.Module):
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
    def __init__(self, dim, n_heads, mlp_ratio=4.0,
                 qkv_bias=True, p=0., attn_p=0.):
        super(Transformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1E-6)
        self.attn = LinearAttention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm2 = nn.LayerNorm(dim, eps=1E-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features, dim, p)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer3D(nn.Module):
    """
    Example Vision Transformer model for 3D images.

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

    def __init__(self, img_size=120, patch_size=16, in_c=1,
                 n_classes=1, embed_dim=512, depth=8, n_heads=8,
                 mlp_ratio=4., qkv_bias=True, dropout=0.,
                 dws_ratio=2):
        super(VisionTransformer3D, self).__init__()
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, in_c=in_c, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.transformers = nn.ModuleList([
            Transformer(dim=embed_dim, n_heads=n_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        p=dropout, attn_p=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1E-6)
        self.head = nn.Linear(embed_dim, n_classes)
        
        self.downsample = nn.Conv3d(in_c, in_c,
                                    kernel_size=dws_ratio, stride=dws_ratio)       
        self.upsample = nn.ConvTranspose3d(in_c, in_c,
                                    kernel_size=dws_ratio, stride=dws_ratio)

    def forward(self, x):
        x = self.downsample(x)
        x_shape = x.shape
        n_samples = x_shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for transformer in self.transformers:
            x = transformer(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final).reshape(x_shape)
        x = self.upsample(x)
        return x


def test_vision_transformer3d():
    # Define the model parameters
    img_size = 240
    patch_size = 8
    in_c = 1
    n_classes = 48*48*48
    embed_dim = 512
    depth = 8
    n_heads = 8
    mlp_ratio = 8.0
    qkv_bias = True
    dropout = 0.0
    dws_ratio = 5

    # Instantiate the VisionTransformer3D model
    model = VisionTransformer3D(
        img_size=int(img_size/dws_ratio),
        patch_size=patch_size,
        in_c=in_c,
        n_classes=n_classes,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        dropout=dropout,
        dws_ratio=dws_ratio
    )

    # Print the model architecture (optional)
    print(f'\nGated Linear 3DVT Model size: {
          int(count_parameters(model)/1000000)}M\n')
    # print(model)

    # Create a random input tensor with the shape (batch_size, channels, depth, height, width)
    batch_size = 2
    input_tensor = torch.randn(batch_size, in_c, img_size, img_size, img_size)

    # Pass the input tensor through the model
    output = model(input_tensor).view(input_tensor.shape)

    # Print the output shape
    print("Output shape:", output.shape)

    test_model_memory_usage(model, input_tensor)

    print("Test passed!")


# Run the test function
if __name__ == '__main__':
    test_vision_transformer3d()
