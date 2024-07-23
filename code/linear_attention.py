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
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim))

    def forward(self, x):
        pos_embed = torch.cat(
            [self.pos_embed for _ in range(x.shape[0])], dim=0)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) + pos_embed
        return x


class LinearGatedAttention(nn.Module):
    '''
    @Ref:
        https://arxiv.org/pdf/2305.07239
        https://github.com/dengyecode/T-former_image_inpainting/blob/master/model/network.py#L239
    '''

    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_p=0., proj_p=0.):
        super(LinearGatedAttention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                               nn.GELU(),
                               nn.Linear(dim, dim, bias=qkv_bias),
                               nn.Softplus())
        self.k = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                               nn.GELU(),
                               nn.Linear(dim, dim, bias=qkv_bias),
                               nn.Softplus())
        self.v = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                               nn.GELU())
        self.g = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                               nn.GELU())

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_toks, dim = x.shape
        if dim != self.dim:
            raise ValueError(
                f"Expected input dimension {self.dim}, but got {dim}"
            )

        q = self.q(x).reshape(n_samples, n_toks,
                              self.n_heads,
                              self.head_dim).permute(0, 2, 1, 3)

        k = self.k(x).reshape(n_samples, n_toks,
                              self.n_heads,
                              self.head_dim).permute(0, 2, 1, 3)

        v = self.v(x).reshape(n_samples, n_toks,
                              self.n_heads,
                              self.head_dim).permute(0, 2, 1, 3)

        g = self.g(x).reshape(n_samples, n_toks,
                              self.n_heads,
                              self.head_dim).permute(0, 2, 1, 3)

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


class LinearGatedTransformer(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0,
                 qkv_bias=True, p=0., attn_p=0.):
        super(LinearGatedTransformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1E-6)
        self.attn = LinearGatedAttention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm2 = nn.LayerNorm(dim, eps=1E-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features, dim, p)

    def forward(self, x):
        y = self.norm1(x)
        y = self.attn(y)
        y = y + self.mlp(self.norm2(y))
        return y


class VisionTransformerBlock(nn.Module):
    def __init__(self, img_size=120, patch_size=16, in_c=1, out_c=1,
                 n_classes=1, embed_dim=512, n_heads=8,
                 mlp_ratio=4., qkv_bias=True, dropout=0.):
        super(VisionTransformerBlock, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size//patch_size)**3
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, in_c=in_c, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(dropout)
        self.transformer = LinearGatedTransformer(dim=embed_dim, n_heads=n_heads,
                                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                  p=dropout, attn_p=dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=1E-6)
        self.head = nn.Sequential(nn.Linear(embed_dim, n_classes),
                                  nn.LayerNorm(n_classes),
                                  nn.GELU())

        self.conv_out = nn.Sequential(nn.Conv3d(
            self.num_patches, out_c, kernel_size=1, stride=1),
            nn.InstanceNorm3d(out_c),
            nn.GELU())

    def forward(self, x):
        y_shape = x.shape
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.transformer(y)
        y = self.norm(y)
        y = self.head(y)
        y = y.reshape(y_shape[0], self.num_patches,
                      y_shape[2], y_shape[3], y_shape[4])
        y = self.conv_out(y)
        return y


def test_vision_transformer3d():
    # Define the model parameters
    img_size = 48
    patch_size = 8
    in_c = 1
    out_c = 32
    n_classes = img_size**3
    embed_dim = 512
    n_heads = 8
    mlp_ratio = 8.0
    qkv_bias = True
    dropout = 0.0

    # Instantiate the VisionTransformer3D model
    model = VisionTransformerBlock(
        img_size=img_size,
        patch_size=patch_size,
        in_c=in_c,
        out_c=out_c,
        n_classes=n_classes,
        embed_dim=embed_dim,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        dropout=dropout
    )

    # Print the model architecture (optional)
    print(
        f'\nGated Linear 3DVT Model size: {int(count_parameters(model)/1000000)}M\n'
    )
    # print(model)

    # Create a random input tensor with the shape (batch_size, channels, depth, height, width)
    batch_size = 1
    input_tensor = torch.randn(batch_size, in_c, img_size, img_size, img_size)

    # Pass the input tensor through the model
    output = model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)

    test_model_memory_usage(model, input_tensor)

    print("Test passed!")


# Run the test function
if __name__ == '__main__':
    test_vision_transformer3d()
