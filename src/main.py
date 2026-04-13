import hydra
import os
import warnings
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from .config import load_typed_root_config
from pathlib import Path
from pytorch_lightning.loggers.wandb import WandbLogger
from src.misc.callbacks import GPUMemoryMonitor, GradientMonitor, TrainingSpeedCallback
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from src.misc.local_logger import LocalLogger
from src.misc.wandb_tools import update_checkpoint_path
from src.misc.step_tracker import StepTracker
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import AdvancedProfiler
from src.dataset.datamodule import DataModule
from src.model.wrapper import DiffusionModel


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main"
)
def train(cfg_dict: DictConfig):
    
    cfg = load_typed_root_config(cfg_dict)
    
    if cfg_dict.output_dir is None or cfg_dict.job_array:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None and not cfg_dict.job_array:
            wandb_extra_kwargs.update(
                {"id": cfg_dict.wandb.id, "resume": "must"})
        else:
            # this makes sure the run is resumable from the same id when using
            #  a job array
            wandb_extra_kwargs.update(
                {"id": f"{output_dir.parent.name}_{output_dir.name}",
                 "resume": "allow"}
            )
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    callbacks.append(GPUMemoryMonitor())
    callbacks.append(TrainingSpeedCallback())

    if cfg.trainer.monitor_gradients:
        callbacks.append(GradientMonitor())

    callbacks.extend([
        ModelCheckpoint(
            output_dir / "checkpoints",
            filename="step-{step}",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the latest k ckpt, can do offline test later
            save_last=True,
            save_on_train_epoch_end=False,
        ),
        # ModelCheckpoint(
        #     output_dir / "checkpoints_periodic",
        #     filename="periodic-{step}",
        #     every_n_train_steps=20000,
        #     save_top_k=-1,
        #     monitor="info/global_step",
        #     mode="max",
        #     save_last=False,
        #     save_on_train_epoch_end=False,
        # ),
        ModelCheckpoint(
            output_dir / "checkpoints_best",
            filename="best-{step}-fid-{val/fid:.2f}",
            every_n_train_steps=2000,
            save_top_k=5,
            monitor="val/fid",
            mode="min",
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
    ]
    )

    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = "_"

    # Prepare the checkpoint for loading.
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "last.ckpt"

    if cfg_dict.job_array and os.path.exists(checkpoint_path):
        print(f"Continuing from checkpoint: {checkpoint_path}.")
        resume = True
    else:
        checkpoint_path = update_checkpoint_path(
            cfg.checkpointing.load, cfg.wandb)
        resume = cfg_dict.checkpointing.resume

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    if cfg_dict.job_array:
        # get the step that was reached in the previous run
        previous_max_steps = (
            torch.load(checkpoint_path)["global_step"]
            if cfg_dict.mode == "train" and resume
            else 0
        )
        max_steps = min(
            previous_max_steps + cfg_dict.trainer.task_steps,
            cfg_dict.trainer.max_steps
        )
    else:
        max_steps = cfg_dict.trainer.max_steps

    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"

    # do the sanity checks only at the beginning of the job array
    num_sanity_val_steps = cfg.trainer.num_sanity_val_steps
    if cfg_dict.job_array and max_steps > cfg_dict.trainer.task_steps:
        num_sanity_val_steps = 0
    
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy=strategy,
        callbacks=callbacks,
        check_val_every_n_epoch=None,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=max_steps,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_val_batches=cfg.trainer.limit_val_batches,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        profiler=(
            AdvancedProfiler(dirpath=output_dir, filename="profile")
            if cfg.trainer.profile
            else None
        ),
    )

    datamodule = DataModule(
        data_loader_cfg=cfg.data_loader,
        dataset_cfg=cfg.dataset,
        step_tracker=step_tracker,
        global_rank=trainer.global_rank,
        test_num_samples=cfg.test.num_samples if cfg.mode == "test" else None,
    )

    model = DiffusionModel(
        model_cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        train_cfg=cfg.train,
        test_cfg=cfg.test,
        step_tracker=step_tracker,
        output_dir=output_dir,
        max_steps=cfg.trainer.max_steps
    )


    if cfg.mode == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path if resume else None)
    elif cfg.mode == "test":
        trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)





if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    train()
