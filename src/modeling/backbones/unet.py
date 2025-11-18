import torch
import torch.nn as nn
from src.core.registry import MODEL_REGISTRY
from ..components.common import TimestepEmbedSequential, GroupNorm32, zero_module
from ..components.embeddings import PositionalEmbedding
from ..components.blocks import ResBlock, Downsample, Upsample
from ..components.attention import AttentionBlock

@MODEL_REGISTRY.register("UNetModel")
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embeddings.
    """
    def __init__(self, img_size, base_channels, conv_resample=True, n_heads=1, n_head_channels=-1,
                 channel_mults="", num_res_blocks=2, dropout=0, attention_resolutions="32,16,8",
                 biggan_updown=True, in_channels=1):
        super().__init__()
        self.dtype = torch.float32
        
        # Parse configuration
        if channel_mults == "":
            if img_size == 512: channel_mults = (0.5, 1, 1, 2, 2, 4, 4)
            elif img_size == 256: channel_mults = (1, 1, 2, 2, 4, 4)
            elif img_size == 128: channel_mults = (1, 1, 2, 3, 4)
            elif img_size == 64: channel_mults = (1, 2, 3, 4)
            else: channel_mults = (1, 2, 3, 4) # Fallback
            
        attention_ds = [img_size // int(res) for res in attention_resolutions.split(",")]

        self.image_size = img_size
        self.in_channels = in_channels
        self.model_channels = base_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        
        # Time Embedding
        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
                PositionalEmbedding(base_channels, 1),
                nn.Linear(base_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
                )

        # Downsampling Path
        ch = int(channel_mults[0] * base_channels)
        self.down = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(self.in_channels, base_channels, 3, padding=1))])
        channels = [ch]
        ds = 1
        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=base_channels * mult)]
                ch = base_channels * mult
                if ds in attention_ds:
                    layers.append(AttentionBlock(ch, n_heads=n_heads, n_head_channels=n_head_channels))
                self.down.append(TimestepEmbedSequential(*layers))
                channels.append(ch)
            if i != len(channel_mults) - 1:
                out_channels = ch
                self.down.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout, out_channels=out_channels, down=True) if biggan_updown 
                    else Downsample(ch, conv_resample, out_channels=out_channels)))
                ds *= 2
                ch = out_channels
                channels.append(ch)

        # Middle Path
        self.middle = TimestepEmbedSequential(
                ResBlock(ch, time_embed_dim, dropout),
                AttentionBlock(ch, n_heads=n_heads, n_head_channels=n_head_channels),
                ResBlock(ch, time_embed_dim, dropout)
                )

        # Upsampling Path
        self.up = nn.ModuleList([])
        for i, mult in reversed(list(enumerate(channel_mults))):
            for j in range(num_res_blocks + 1):
                inp_chs = channels.pop()
                layers = [ResBlock(ch + inp_chs, time_embed_dim, dropout, out_channels=base_channels * mult)]
                ch = base_channels * mult
                if ds in attention_ds:
                    layers.append(AttentionBlock(ch, n_heads=n_heads, n_head_channels=n_head_channels))
                if i and j == num_res_blocks:
                    out_channels = ch
                    layers.append(ResBlock(ch, time_embed_dim, dropout, out_channels=out_channels, up=True) if biggan_updown 
                                  else Upsample(ch, conv_resample, out_channels=out_channels))
                    ds //= 2
                self.up.append(TimestepEmbedSequential(*layers))

        # Output
        self.out = nn.Sequential(
                GroupNorm32(32, ch),
                nn.SiLU(),
                zero_module(nn.Conv2d(base_channels * channel_mults[0], self.out_channels, 3, padding=1))
                )

    def forward(self, x, time):
        time_embed = self.time_embedding(time)
        skips = []
        h = x.type(self.dtype)
        for module in self.down:
            h = module(h, time_embed)
            skips.append(h)
        h = self.middle(h, time_embed)
        for module in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, time_embed)
        return self.out(h).type(x.dtype)