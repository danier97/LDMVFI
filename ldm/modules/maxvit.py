import torch
from torch import nn, einsum
import torch.nn.functional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from inspect import isfunction


# Code adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/max_vit.py

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, c=None):
        if exists(c):
            return self.fn(self.norm(x), self.norm(c)) + x
        return self.fn(self.norm(x)) + x


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x, c=None):
        c = default(c, x)
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        c = rearrange(c, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q = self.to_q(x)
        k = self.to_k(c)
        v = self.to_v(c)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)


class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net


class MaxAttentionBlock(nn.Module):
    def __init__(self, in_channels, heads=8, dim_head=64, dropout=0., window_size=8):
        super().__init__()
        w = window_size
        layer_dim = dim_head * heads

        self.rearrange_block_in = Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)  # block-like attention
        self.attn_block = PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w))
        self.ff_block = PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout))
        self.rearrange_block_out = Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)')

        self.rearrange_grid_in = Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)  # grid-like attention
        self.attn_grid = PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w))
        self.ff_grid = PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout))
        self.rearrange_grid_out = Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)')


    def forward(self, x):

        # block attention
        x = self.rearrange_block_in(x)        
        x = self.attn_block(x)
        x = self.ff_block(x)
        x = self.rearrange_block_out(x)

        # grid attention
        x = self.rearrange_grid_in(x)
        x = self.attn_grid(x)
        x = self.ff_grid(x)
        x = self.rearrange_grid_out(x) 
        
        ## output stage
        return x

class SpatialCrossAttentionWithMax(nn.Module):
    def __init__(self, in_channels, heads=8, dim_head=64, ctx_dim=None, dropout=0., window_size=8):
        super().__init__()
        w = window_size
        layer_dim = dim_head * heads
        if ctx_dim == None:
            self.proj_in = MBConv(layer_dim*2, layer_dim, downsample=False)
        else:
            self.proj_in = MBConv(ctx_dim, layer_dim, downsample=False)

        self.rearrange_block_in = Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)  # block-like attention
        self.attn_block = PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w))
        self.ff_block = PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout))
        self.rearrange_block_out = Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)')

        self.rearrange_grid_in = Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)  # grid-like attention
        self.attn_grid = PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w))
        self.ff_grid = PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout))
        self.rearrange_grid_out = Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)')

        self.out_conv = nn.Sequential(
            SqueezeExcitation(dim=layer_dim*2),
            nn.Conv2d(layer_dim*2, layer_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, context=None):
        context = default(context, x)

        # MBConv
        c = self.proj_in(context)

        # block attention
        x = self.rearrange_block_in(x)   
        c = self.rearrange_block_in(c)
        x = self.attn_block(x, c)
        x = self.ff_block(x)
        x = self.rearrange_block_out(x)
        c = self.rearrange_block_out(c)

        # grid attention
        x = self.rearrange_grid_in(x)
        c = self.rearrange_grid_in(c)
        x = self.attn_grid(x, c)
        x = self.ff_grid(x)
        x = self.rearrange_grid_out(x)
        
        return x


class SpatialTransformerWithMax(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding) to inner_dim (d) using conv1x1
    Then reshape to b, t, d.
    Then apply standard transformer action (BasicTransformerBlock).
    Finally, reshape to image and pass to output conv1x1 layer, to restore the channel size of input.
    The dims of the input and output of the block are the same (arg in_channels).
    """
    def __init__(self, in_channels, n_heads, d_head, dropout=0., context_dim=None, w=2):
        super().__init__()
        self.in_channels = in_channels
        self.context_dim = context_dim
        inner_dim = n_heads * d_head

        self.proj_in = MBConv(context_dim, inner_dim, downsample=False)

        self.rearrange_block_in = Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)  # block-like attention
        self.attn_block = PreNormResidual(inner_dim, Attention(dim = inner_dim, dim_head = d_head, dropout = dropout, window_size = w))
        self.ff_block = PreNormResidual(inner_dim, FeedForward(dim = inner_dim, dropout = dropout))
        self.rearrange_block_out = Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)')

        self.rearrange_grid_in = Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)  # grid-like attention
        self.attn_grid = PreNormResidual(inner_dim, Attention(dim = inner_dim, dim_head = d_head, dropout = dropout, window_size = w))
        self.ff_grid = PreNormResidual(inner_dim, FeedForward(dim = inner_dim, dropout = dropout))
        self.rearrange_grid_out = Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)')

    def forward(self, x, context=None):
        context = default(context, x)

        # down sample context if necessary
        # this is due to the implementation of max crossattn here
        if context.shape[2] != x.shape[2]:
            stride = context.shape[2] // x.shape[2]
            context = torch.nn.functional.avg_pool2d(context, kernel_size=stride, stride=stride)

        # MBConv
        c = self.proj_in(context)

        # block attention
        x = self.rearrange_block_in(x)   
        c = self.rearrange_block_in(c)
        x = self.attn_block(x, c)
        x = self.ff_block(x)
        x = self.rearrange_block_out(x)
        c = self.rearrange_block_out(c)

        # grid attention
        x = self.rearrange_grid_in(x)
        c = self.rearrange_grid_in(c)
        x = self.attn_grid(x, c)
        x = self.ff_grid(x)
        x = self.rearrange_grid_out(x)
        
        return x