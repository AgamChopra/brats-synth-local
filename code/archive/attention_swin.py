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

from utils import count_parameters


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, in_c=1, embed_dim=96):
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


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, n_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C //
                                  self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, attn_p=0., proj_p=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size,
                                    n_heads=num_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_p),
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(
                self.input_resolution, self.window_size, self.shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, input_resolution, window_size, shift_size):
        H, W, D = input_resolution
        img_mask = torch.zeros((1, H, W, D, 1))
        h_slices = (slice(0, -window_size), slice(-window_size, -
                    shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -
                    shift_size), slice(-shift_size, None))
        d_slices = (slice(0, -window_size), slice(-window_size, -
                    shift_size), slice(-shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for d in d_slices:
                    img_mask[:, h, w, d, :] = cnt
                    cnt += 1

        mask_windows = img_mask.view(1, H // window_size, window_size,
                                     W // window_size, window_size, D // window_size, window_size, 1)
        mask_windows = mask_windows.permute(
            0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, window_size * window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        H, W, D = self.input_resolution
        B, L, C = x.shape
        assert L == H * W * D, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, D, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        else:
            shifted_x = x

        x_windows = shifted_x.view(-1, self.window_size,
                                   self.window_size, self.window_size, C)
        x_windows = x_windows.view(-1, self.window_size *
                                   self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, self.window_size, C)

        if self.shift_size > 0:
            shifted_x = attn_windows.view(B, H, W, D, C)
            x = torch.roll(shifted_x, shifts=(self.shift_size,
                           self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = attn_windows.view(B, H, W, D, C)

        x = x.view(B, H * W * D, C)
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W, D = self.input_resolution
        B, L, C = x.shape
        assert L == H * W * D, "input feature has wrong size"

        x = x.view(B, H, W, D, C)
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinTransformer3D(nn.Module):
    """
    Example Swin Transformer model for 3D images.
    """
    def __init__(self, img_size=128, patch_size=4, in_c=1, n_classes=1, embed_dim=96, depths=[2, 2, 6, 2], n_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., dropout=0., norm_layer=nn.LayerNorm):
        super(SwinTransformer3D, self).__init__()
        self.num_classes = n_classes
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, in_c=in_c, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            for _ in range(depths[i_layer]):
                layer.append(
                    SwinTransformerBlock(
                        dim=int(embed_dim * 2**i_layer),
                        input_resolution=(
                            img_size // (2**i_layer), img_size // (2**i_layer), img_size // (2**i_layer)),
                        num_heads=n_heads[i_layer],
                        window_size=window_size,
                        shift_size=0 if (i_layer %
                                         2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        attn_p=dropout,
                        proj_p=drop_rate,
                    )
                )
            if i_layer < self.num_layers - 1:
                layer.append(PatchMerging((img_size // (2**i_layer), img_size // (
                    2**i_layer), img_size // (2**i_layer)), dim=int(embed_dim * 2**i_layer)))
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim * 2**(self.num_layers - 1))
        self.head = nn.Linear(
            embed_dim * 2**(self.num_layers - 1), n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            for blk in layer:
                x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


def test_swin_transformer3d():
    # Define the model parameters
    img_size = 128
    patch_size = 16
    in_c = 1
    n_classes = 1
    embed_dim = 512
    depth = [2, 2, 6, 2]
    n_heads = [3, 6, 12, 24]
    window_size=7
    mlp_ratio = 4.0
    qkv_bias = True
    dropout = 0.0

    # Instantiate the VisionTransformer3D model
    model = SwinTransformer3D(
        img_size=img_size,
        patch_size=patch_size,
        in_c=in_c,
        n_classes=n_classes,
        embed_dim=embed_dim,
        depths=depth,
        n_heads=n_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        dropout=dropout
    )

    # Print the model architecture (optional)
    print(f'Model size: {int(count_parameters(model)/1000000)}M')
    print(model)

    # Create a random input tensor with the shape (batch_size, channels, depth, height, width)
    batch_size = 2
    input_tensor = torch.randn(batch_size, in_c, img_size, img_size, img_size)

    # Pass the input tensor through the model
    start_time = time.time()
    output = model(input_tensor)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the output shape
    print("Output shape:", output.shape)

    # Check if the output shape is correct
    assert output.shape == (
        batch_size, n_classes), "Output shape is incorrect!"
    
    print("Elapsed time: {:.6f} seconds".format(elapsed_time))

    print("Test passed!")


# Run the test function
if __name__ == '__main__':
    test_swin_transformer3d()