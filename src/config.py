from dataclasses import dataclass
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Literal, Type, TypeVar, Any, Optional, Union
from dacite import Config, from_dict
from src.dataset import DatasetCfg
from src.dataset.datamodule import DataLoaderCfg
from src.model.wrapper import ModelCfg, OptimizerCfg, TrainCfg, TestCfg, CurvatureCfg

@dataclass
class WandbCfg:
    entity: Optional[str]
    project: Optional[str]
    mode: Optional[Literal["disabled", "offline", "online", "run"]]
    name: Optional[str]
    id: Optional[str]

@dataclass
class CheckpointingCfg:
    every_n_train_steps: int
    save_top_k: int
    load: Optional[str]
    resume: bool

@dataclass
class TrainerCfg:
    monitor_gradients: bool
    max_steps: int
    num_nodes: int
    val_check_interval: Union[int, float]
    gradient_clip_val: Optional[float]
    num_sanity_val_steps: int
    limit_val_batches: Union[int, float]
    accumulate_grad_batches: int
    precision: Optional[str]
    profile: bool
    task_steps: int
    log_every_n_steps: int


@dataclass
class RootCfg:
    wandb: WandbCfg
    mode: Literal["train", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    train: TrainCfg
    test: TestCfg


T = TypeVar("T")
TYPE_HOOKS = {
    Path: Path,
}

def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg, resolve=True),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )

def separate_loss_cfg_wrappers(joined: dict) -> list[Any]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: Any

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]

def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {list[Any]: separate_loss_cfg_wrappers},
    )
