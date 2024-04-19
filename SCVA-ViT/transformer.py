import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.registry import register_model


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=16, patch_size=4, in_chans=3, embed_dim=768):#img_size=224,patch_size=16
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):#head=8
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.att_drop = nn.Dropout(attn_drop)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)#分离维度
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        energy *= self.scale
        att = F.softmax(energy, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(
            ResidualAdd(nn.Sequential(
                norm_layer(dim),
                MultiHeadAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                   proj_drop=drop),
                nn.Dropout(drop)
            )),
            ResidualAdd(nn.Sequential(
                norm_layer(dim),
                Mlp(in_features=dim, hidden_features=int(mlp_ratio * dim), act_layer=act_layer, drop=drop),
                nn.Dropout(drop)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(*[TransformerEncoderBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                                   drop_path, act_layer, norm_layer) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, dim=768, num_classes=1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes))


class ViT(nn.Sequential):
    def __init__(self, img_size=16, patch_size=4, in_chans=3, num_classes=1000, embed_dim=768, # img_size=224, patch_size=16
                 num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depth=12):
        super().__init__(
            PatchEmbedding(img_size, patch_size, in_chans, embed_dim),
            TransformerEncoder(depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate,
                               drop_path=drop_path_rate),
            ClassificationHead(embed_dim, num_classes)
        )


@register_model
def vit_small(pretrained=False, **kwargs):
    model = ViT(img_size=16, patch_size=4, embed_dim=192, num_heads=3, depth=12, **kwargs)#img_size=16, patch_size=16

    return model


@register_model
def vit_base(pretrained=False, **kwargs):
    model = ViT(img_size=224, patch_size=16, embed_dim=768, num_heads=12, depth=12, **kwargs)#img_size=224, patch_size=16

    return model

