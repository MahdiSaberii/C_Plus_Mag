"""
Time-conditioned U-Net used as the learned proximal operator.
Adapted from OpenAI's guided-diffusion (https://github.com/openai/guided-diffusion).
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import checkpoint, normalization, timestep_embedding, zero_module


def create_model(num_channels=32, channel_mult=(3, 4, 5), num_res_blocks=1, InOut_Chn=2,
                 use_time_emb=True, use_norm=True, use_checkpoint=False):
    return UNetModel(
        in_channels=InOut_Chn,
        model_channels=num_channels,
        out_channels=InOut_Chn,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=True,
        use_time_emb=use_time_emb,
        use_norm=use_norm,
    )


class TimestepBlock(nn.Module):
    """Any module whose forward() takes timestep embeddings as a second argument."""

    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to the children that support it."""

    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class Upsample(nn.Module):
    """Nearest-neighbour 2x upsampling."""

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


class Downsample(nn.Module):
    """2x average-pool downsampling."""

    def __init__(self):
        super().__init__()
        self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_scale_shift_norm: use FiLM-like (scale, shift) conditioning.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param use_time_emb: if True, condition on the timestep embedding.
    """

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_scale_shift_norm=False,
                 use_checkpoint=False, up=False, down=False, use_time_emb=False):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_time_emb = use_time_emb

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd, self.x_upd = Upsample(), Upsample()
        elif down:
            self.h_upd, self.x_upd = Downsample(), Downsample()
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.use_time_emb:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.use_time_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = out_rest(out_norm(h) * (1 + scale) + shift)
            else:
                h = self.out_layers(h + emb_out)
        else:
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from https://github.com/hojonathanho/diffusion.
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.attention(self.qkv(self.norm(x)))
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """QKV attention (heads split before q, k, v)."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # more stable than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class UNetModel(nn.Module):
    """
    U-Net with timestep embedding and an attention block at the bottleneck.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the U-Net.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param use_time_emb: condition the residual blocks on the timestep (unroll index).
    :param use_norm: normalize the real/imag input channels and undo it at the output.
    """

    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, dropout=0,
                 channel_mult=(1, 2, 4, 8), use_checkpoint=False, num_heads=1,
                 use_scale_shift_norm=False, use_time_emb=False, use_norm=False):
        super().__init__()
        self.model_channels = model_channels
        self.use_time_emb = use_time_emb
        self.use_norm = use_norm

        res_kwargs = dict(use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                          use_time_emb=use_time_emb)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels), **res_kwargs)
                ))
                ch = int(mult * model_channels)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout, out_channels=ch, down=True, **res_kwargs)
                ))
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, **res_kwargs),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, time_embed_dim, dropout, **res_kwargs),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout,
                                   out_channels=int(model_channels * mult), **res_kwargs)]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks:
                    layers.append(ResBlock(ch, time_embed_dim, dropout, out_channels=ch, up=True, **res_kwargs))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='selu')

    def norm(self, x):
        # Channels are [real frames..., imag frames...]: normalize the real and the imaginary group separately.
        b, c, h, w = x.shape
        flat = x.view(b, 2, c // 2 * h * w)
        mean = flat.mean(dim=2).view(b, 2, 1, 1, 1)
        std  = flat.std(dim=2).view(b, 2, 1, 1, 1).clamp_min(1e-6)
        return ((x.view(b, 2, c // 2, h, w) - mean) / std).view(b, c, h, w), mean, std

    def unnorm(self, x, mean, std):
        b, c, h, w = x.shape
        return (x.view(b, 2, c // 2, h, w) * std + mean).view(b, c, h, w)

    def forward(self, x, timesteps=None):
        """
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps (unroll indices).
        :return: an [N x C x H x W] Tensor of outputs.
        """
        if self.use_norm:
            x, mean, std = self.norm(x)

        emb = None
        if self.use_time_emb:
            emb = self.time_embed(timestep_embedding(timesteps.to(dtype=x.dtype), self.model_channels))

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(th.cat([h, hs.pop()], dim=1), emb)

        x = self.out(h)
        if self.use_norm:
            x = self.unnorm(x, mean, std)
        return x
