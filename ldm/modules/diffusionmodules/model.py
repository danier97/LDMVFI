# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np

from ldm.modules.attention import LinearAttention, SpatialCrossAttentionWithPosEmb
from ldm.modules.maxvit import SpatialCrossAttentionWithMax, MaxAttentionBlock

from cupy_module import dsepconv


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class IdentityWrapper(nn.Module):
    """
    A wrapper for nn.Identity that allows additional input.
    """
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Identity()

    def forward(self, x, context=None):
        return self.layer(x)



class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none", 'max'], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    elif attn_type == 'max':
        return MaxAttentionBlock(in_channels, heads=1, dim_head=in_channels)
    else:
        return LinAttnBlock(in_channels)




class FIEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch # 128
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult) # 3
        self.num_res_blocks = num_res_blocks # 2
        self.resolution = resolution # 256
        self.in_channels = in_channels # 3

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution # 256
        in_ch_mult = (1,)+tuple(ch_mult) # (1,1,2,4)
        self.in_ch_mult = in_ch_mult # (1,1,2,4)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = int(ch*in_ch_mult[i_level])
            block_out = int(ch*ch_mult[i_level])
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            # if i_level != self.num_resolutions-1:
            down.downsample = Downsample(block_in, resamp_with_conv)
            curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels, # 3
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, ret_feature=False):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        phi_list = []
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            # if i_level != self.num_resolutions-1:
            hs.append(self.down[i_level].downsample(hs[-1]))
            phi_list.append(hs[-1])

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if ret_feature:
            return h, phi_list
        return h


class FlowEncoder(FIEncoder):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0, resamp_with_conv=True, in_channels, resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla", **ignore_kwargs):
        super().__init__(
            ch=ch, 
            out_ch=out_ch,
            ch_mult=ch_mult, 
            num_res_blocks=num_res_blocks, 
            attn_resolutions=attn_resolutions, 
            dropout=dropout, 
            resamp_with_conv=resamp_with_conv, 
            in_channels=in_channels, 
            resolution=resolution, 
            z_channels=z_channels, 
            double_z=double_z, 
            use_linear_attn=use_linear_attn, 
            attn_type=attn_type, 
            **ignore_kwargs
        )



class FlowDecoderWithResidual(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", num_head_channels=32, num_heads=1, cond_type=None,
                 **ignorekwargs):
        super().__init__()

        def KernelHead(c_in):
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    # torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1)
                )
        # end

        def OffsetHead(c_in):
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=5 ** 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    # torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=5 ** 2, out_channels=5 ** 2, kernel_size=3, stride=1, padding=1)
                )


        def MaskHead(c_in):
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=5 ** 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    # torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=5 ** 2, out_channels=5 ** 2, kernel_size=3,
                             stride=1, padding=1),
                    torch.nn.Sigmoid()
                )

        def ResidualHead(c_in):
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    # torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
                )
        

        self.ch = ch # 128
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult) # 3
        self.num_res_blocks = num_res_blocks # 2
        self.resolution = resolution # 256
        self.in_channels = in_channels # 3
        self.give_pre_end = give_pre_end # False
        self.tanh_out = tanh_out # False

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult) # (1,1,2,4)
        block_in = int(ch*ch_mult[self.num_resolutions-1]) # 512
        curr_res = resolution // 2**(self.num_resolutions-1) # 64
        self.z_shape = (1,z_channels,curr_res,curr_res) # (1,3,64,64)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): # 2,1,0
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = int(ch*ch_mult[i_level])
            # ResBlocks
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            # CrossAttention
            if num_head_channels == -1:
                dim_head = block_in // num_heads
            else:
                num_heads = block_in // num_head_channels
                dim_head = num_head_channels # 32
            if cond_type == 'cross_attn':
                cross_attn = SpatialCrossAttentionWithPosEmb(in_channels=block_in, 
                                                             heads=num_heads,
                                                             dim_head=dim_head)
            elif cond_type == 'max_cross_attn':
                cross_attn = SpatialCrossAttentionWithMax(in_channels=block_in,
                                                          heads=num_heads,
                                                          dim_head=dim_head)
            elif cond_type == 'max_cross_attn_frame':
                cross_attn = SpatialCrossAttentionWithMax(in_channels=block_in,
                                                          heads=num_heads,
                                                          dim_head=dim_head,
                                                          ctx_dim=6)
            else:
                cross_attn = IdentityWrapper()

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.cross_attn = cross_attn

            # Upsample
            # if i_level != self.num_resolutions-1: ## THIS IS ORIGINAL CODE
            # if i_level != 0:
            up.upsample = Upsample(block_in, resamp_with_conv)
            curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        block_in,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.moduleAlpha1 = OffsetHead(c_in=block_in)
        self.moduleAlpha2 = OffsetHead(c_in=block_in)
        self.moduleBeta1 = OffsetHead(c_in=block_in)
        self.moduleBeta2 = OffsetHead(c_in=block_in)
        self.moduleKernelHorizontal1 = KernelHead(c_in=block_in)
        self.moduleKernelHorizontal2 = KernelHead(c_in=block_in)
        self.moduleKernelVertical1 = KernelHead(c_in=block_in)
        self.moduleKernelVertical2 = KernelHead(c_in=block_in)
        self.moduleMask = MaskHead(c_in=block_in)
        self.moduleResidual = ResidualHead(c_in=block_in)
        self.modulePad = torch.nn.ReplicationPad2d([2, 2, 2, 2])

    def forward(self, z, cond_dict):
        phi_prev_list = cond_dict['phi_prev_list']
        phi_next_list = cond_dict['phi_next_list']
        frame_prev = cond_dict['frame_prev']
        frame_next = cond_dict['frame_next']

        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)): # [2,1,0]
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            ctx = None
            if phi_prev_list[i_level] is not None:
                ctx = torch.cat([phi_prev_list[i_level], phi_next_list[i_level]], dim=1)
            h = self.up[i_level].cross_attn(h, ctx)
            # if i_level != self.num_resolutions-1:
            # if i_level != 0:
            h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        alpha1 = self.moduleAlpha1(h)
        alpha2 = self.moduleAlpha2(h)
        beta1 = self.moduleBeta1(h)
        beta2 = self.moduleBeta2(h)
        v1 = self.moduleKernelVertical1(h)
        v2 = self.moduleKernelVertical2(h)
        h1 = self.moduleKernelHorizontal1(h)
        h2 = self.moduleKernelHorizontal2(h)
        mask1 = self.moduleMask(h)
        mask2 = 1.0 - mask1
        warped1 = dsepconv.FunctionDSepconv(self.modulePad(frame_prev), v1, h1, alpha1, beta1, mask1)
        warped2 = dsepconv.FunctionDSepconv(self.modulePad(frame_next), v2, h2, alpha2, beta2, mask2)
        warped = warped1 + warped2
        out = warped + self.moduleResidual(h)
        return out