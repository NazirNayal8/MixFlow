import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

from diffusers import UNet2DModel
from easydict import EasyDict as edict

from .unet import UNet2D


@dataclass
class CommonNoisePredictorCfg:
    input_mode: Literal["sample_only", "noise_only", "sample_and_noise"]
    gaussian: bool
    gaussian_mode: Optional[Literal["mean", "mean_var"]]
    normalize_output: bool


@dataclass
class UNetCfg(CommonNoisePredictorCfg):
    name: Literal["unet"]
    sample_size: Optional[Union[int, Tuple[int, int]]]
    in_channels: int
    out_channels: int
    down_block_types: List[str]
    up_block_types: List[str]
    block_out_channels: List[int]
    layers_per_block: int
    mid_block_type: Optional[str] = None
    initialize: Optional[str] = None
    skip_connection: bool = False
    class_encoder: bool = False
    num_classes: Optional[int] = None


@dataclass
class UNetClassCondCfg(CommonNoisePredictorCfg):
    name: Literal["unet_class_cond"]
    sample_size: Optional[Union[int, Tuple[int, int]]]
    in_channels: int
    out_channels: int
    center_input_sample: bool
    time_embedding_type: str
    freq_shift: int
    flip_sin_to_cos: bool
    down_block_types: List[str]
    up_block_types: List[str]
    block_out_channels: List[int]
    layers_per_block: int
    mid_block_scale_factor: float
    downsample_padding: int
    downsample_type: str
    upsample_type: str
    dropout: float
    act_fn: str
    attention_head_dim: Optional[int]
    norm_num_groups: int
    attn_norm_num_groups: Optional[int]
    norm_eps: float
    resnet_time_scale_shift: str
    add_attention: bool
    class_embed_type: Optional[str]
    num_class_embeds: Optional[int]
    num_train_timesteps: Optional[int]
    class_encoder: Optional[bool]
    num_classes: Optional[int]


class UNet(UNet2D):
    def __init__(self, cfg: UNetCfg):
        super().__init__(
            sample_size=cfg.sample_size,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            down_block_types=tuple(cfg.down_block_types),
            mid_block_type=cfg.mid_block_type,
            up_block_types=tuple(cfg.up_block_types),
            block_out_channels=tuple(cfg.block_out_channels),
            layers_per_block=cfg.layers_per_block,
            initialize=cfg.initialize,
            skip_connection=cfg.skip_connection,
        )
        self.cfg = cfg

        if cfg.class_encoder:
            self.class_embedding = nn.Embedding(
                cfg.num_classes,
                cfg.sample_size * cfg.sample_size * cfg.in_channels,
            )

    def forward(self, x, class_labels):
        if self.cfg.class_encoder:
            class_embedding = self.class_embedding(class_labels)
            class_embedding = class_embedding.view(
                x.shape[0], self.cfg.in_channels, x.shape[2], x.shape[3]
            )
            output = super().forward(class_embedding)
        else:
            output = super().forward(x)

        data = edict()
        if self.cfg.gaussian:
            if self.cfg.gaussian_mode == "mean_var":
                mean, log_var = torch.chunk(output, 2, dim=1)
            elif self.cfg.gaussian_mode == "mean":
                mean = output
                log_var = torch.zeros_like(mean)
            else:
                raise ValueError("Invalid gaussian mode")
            data["mean"] = mean
            data["log_var"] = log_var
            random_noise = torch.randn_like(mean)
            data["sampling_random_noise"] = random_noise
            output = mean + torch.exp(log_var * 0.5) * random_noise

        if self.cfg.normalize_output:
            output = torch.tanh(output)

        return output, data


class UNetClassCond(UNet2DModel):
    def __init__(self, cfg: UNetClassCondCfg):
        cfg_dict = cfg.__dict__.copy()
        del cfg_dict["name"]
        for key in CommonNoisePredictorCfg.__dataclass_fields__.keys():
            cfg_dict.pop(key, None)
        cfg_dict.pop("class_encoder", None)
        cfg_dict.pop("num_classes", None)

        super().__init__(**cfg_dict)
        self.cfg = cfg

    def forward(self, x, class_labels):
        t = torch.zeros(x.shape[0], device=x.device)
        if getattr(self, "class_embedding", None) is None:
            class_labels = None

        output = super().forward(
            x,
            t,
            class_labels=class_labels,
            return_dict=False,
        )[0]
        data = edict()

        if self.cfg.gaussian:
            if self.cfg.gaussian_mode == "mean_var":
                mean, log_var = torch.chunk(output, 2, dim=1)
            elif self.cfg.gaussian_mode == "mean":
                mean = output
                log_var = torch.zeros_like(mean)
            else:
                raise ValueError("Invalid gaussian mode")

            data["mean"] = mean
            data["log_var"] = log_var
            random_noise = torch.randn_like(mean)
            data["sampling_random_noise"] = random_noise
            output = mean + torch.exp(log_var * 0.5) * random_noise

        return output, data


NoisePredictorCfg = Union[UNetCfg, UNetClassCondCfg]


class NoisePredictor(nn.Module):
    def __init__(self, cfg: NoisePredictorCfg):
        super().__init__()
        self.cfg = cfg

        if cfg.gaussian and cfg.gaussian_mode == "mean_var":
            cfg.out_channels = cfg.out_channels * 2

        if cfg.name == "unet":
            self.network = UNet(cfg)
        elif cfg.name == "unet_class_cond":
            self.network = UNetClassCond(cfg)
        else:
            raise ValueError(f"Unsupported noise predictor: {cfg.name}")

    def forward(self, x, class_labels=None):
        data = edict()
        if self.cfg.input_mode == "sample_only":
            inp = x
        elif self.cfg.input_mode == "noise_only":
            random_noise = torch.randn_like(x)
            data["input_random_noise"] = random_noise.clone()
            inp = random_noise
        elif self.cfg.input_mode == "sample_and_noise":
            random_noise = torch.randn_like(x)
            data["input_random_noise"] = random_noise.clone()
            inp = torch.cat([x, random_noise], dim=1)
        else:
            raise ValueError(f"Invalid input mode: {self.cfg.input_mode}")

        predicted_noise, network_data = self.network(inp, class_labels)
        data.update(network_data)

        return predicted_noise, data
