import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)


class UNet2D(nn.Module):
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2D",
        up_block_types: Tuple[str, ...] = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        initialize: Optional[str] = None,
        skip_connection: bool = False,
    ) -> None:
        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block
        self.block_out_channels = block_out_channels
        self.skip_connection = skip_connection

        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                "Must provide the same number of down and up block types."
            )
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                "Must provide the same number of block_out_channels as down blocks."
            )

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=(
                    attention_head_dim
                    if attention_head_dim is not None
                    else output_channel
                ),
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        if mid_block_type is not None:
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=None,
                dropout=dropout,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=(
                    attention_head_dim
                    if attention_head_dim is not None
                    else block_out_channels[-1]
                ),
                resnet_groups=norm_num_groups,
                attn_groups=attn_norm_num_groups,
                add_attention=add_attention,
            )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=(
                    attention_head_dim
                    if attention_head_dim is not None
                    else output_channel
                ),
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)

        num_groups_out = (
            norm_num_groups
            if norm_num_groups is not None
            else min(block_out_channels[0] // 4, 32)
        )
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=num_groups_out,
            eps=norm_eps,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

        if initialize == "identity":
            self._init_identity()
        elif initialize == "zeros":
            self._init_zeros()

    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_zeros(self) -> None:
        self.apply(self.init_weights)

    def _init_identity(self) -> None:
        for name, param in self.named_parameters():
            if "resnets" in name and "conv2" in name:
                if "weight" in name or "bias" in name:
                    nn.init.zeros_(param)
            if "resnets" in name and "norm" in name:
                if "weight" in name:
                    nn.init.ones_(param)
                if "bias" in name:
                    nn.init.zeros_(param)

    def forward(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
    ):
        if self.skip_connection:
            noise = sample[:, 3:]

        skip_sample = sample
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample,
                    skip_sample=skip_sample,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample)

            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(sample, temb=None)

        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(
                    sample,
                    res_samples,
                    skip_sample=skip_sample,
                )
            else:
                sample = upsample_block(sample, res_samples)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample = sample + skip_sample
        if self.skip_connection:
            sample = sample + noise

        if not return_dict:
            return (sample,)
        return sample
