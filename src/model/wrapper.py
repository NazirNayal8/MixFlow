import torch
import json
import os
import numpy as np
import torch.nn.functional as F
from src.misc.step_tracker import StepTracker
from typing import Optional, Dict, Any, List
from pathlib import Path
from pytorch_lightning import LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from src.visualization.layout import vcat, hcat, add_border
from src.visualization.annotation import add_label
from diffusers.optimization import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from dataclasses import dataclass
from src.schedulers import get_scheduler, SchedulerCfg
from src.model.diffusion.noise_predictor import NoisePredictor, NoisePredictorCfg
from src.model.loss import LossCfg, get_loss
from src.misc.image_io import prep_image, save_image

from src.misc.metrics import OptimizedFID

from src.model.denoiser import get_denoiser, DenoiserCfg
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from easydict import EasyDict as edict


@dataclass
class TrainCfg:
    t_eps: Optional[float]
    discrete_t: bool
    use_cfg: bool
    cfg_scale: float
    cfg_p_cond: float
    use_ema: bool
    use_torch_compile: bool
    ema_decay: float
    increasing_ema: bool
    class_conditional: bool
    activate_w_mode: bool
    num_w_values: int
    interpolate_distrubutions: bool
    discrete_w: bool
    num_inference_timesteps: int
    log_predictions: bool
    log_predictions_interval: int
    optimized_fid: bool
    block_diffusion_gradient_to_noise_predictor: bool


@dataclass
class CurvatureCfg:
    data_shape: List[int]
    num_classes: Optional[int]


@dataclass
class TestCfg:
    num_samples: int
    num_inference_timesteps: int
    w: Optional[float]
    output_path: Optional[str]
    log_images: bool
    ode_solver: str
    use_scipy: bool
    optimized_fid: bool
    generate_for_fid: bool
    compute_is: bool
    curvature: bool
    curvature_cfg: Optional[CurvatureCfg]


@dataclass
class LRSchedulerCfg:
    name: str
    frequency: int
    interval: str
    num_warmup_steps: int


@dataclass
class OptimizerCfg:
    name: str
    lr: float
    scale_lr: bool
    scheduler: Optional[LRSchedulerCfg]


@dataclass
class ModelCfg:
    scheduler: SchedulerCfg
    denoiser: DenoiserCfg
    use_noise_predictor: bool
    noise_predictor: NoisePredictorCfg
    aux_losses: Optional[List[LossCfg]]


def unnormalize(x):
    return (x * 0.5 + 0.5).clip(0.0, 1.0)


def get_ema_multi_avg_fn_increasing(decay=0.999):
    """Get the function applying exponential moving average (EMA) across multiple params.
    Following the RectifiedFlow EMA implementation.
    """

    @torch.no_grad()
    def ema_update(ema_param_list, current_param_list, num_updates):
        # foreach lerp only handles float and complex
        if isinstance(num_updates, torch.Tensor):
            num_updates = num_updates.item()
        decay_t = min(decay, (1 + num_updates) / (10 + num_updates))
        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(
            ema_param_list[0]
        ):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay_t)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list):
                p_ema.copy_(p_ema * decay_t + p_model * (1 - decay_t))

    return ema_update


class DiffusionModel(LightningModule):

    def __init__(
        self,
        model_cfg: ModelCfg,
        optimizer_cfg: OptimizerCfg,
        train_cfg: TrainCfg,
        test_cfg: TestCfg,
        step_tracker: StepTracker,
        output_dir: Optional[Path],
        max_steps: int,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.step_tracker = step_tracker
        self.output_dir = output_dir
        self.max_steps = max_steps

        self.scheduler = get_scheduler(model_cfg.scheduler)

        self.denoiser = get_denoiser(model_cfg.denoiser)
        self.denoiser.train()

        if train_cfg.use_ema:

            if train_cfg.increasing_ema:
                multi_avg_fn = get_ema_multi_avg_fn_increasing(train_cfg.ema_decay)
            else:
                multi_avg_fn = get_ema_multi_avg_fn(train_cfg.ema_decay)

            self.ema = AveragedModel(self.denoiser, multi_avg_fn=multi_avg_fn)
            print("Using Exponential Moving Average (EMA)")

        if model_cfg.use_noise_predictor:
            self.noise_predictor = NoisePredictor(model_cfg.noise_predictor)

        if model_cfg.aux_losses is not None:
            self.aux_losses = [get_loss(loss_cfg) for loss_cfg in model_cfg.aux_losses]

        if train_cfg.use_torch_compile:
            import torch._dynamo

            torch._dynamo.config.optimize_ddp = False  # <-- add this

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            effective_batch_size = (
                self.trainer.accumulate_grad_batches
                * self.trainer.num_devices
                * self.trainer.num_nodes
                * self.trainer.datamodule.data_loader_cfg.train.batch_size
            )

            self.lr = (
                self.optimizer_cfg.lr * effective_batch_size
                if self.optimizer_cfg.scale_lr
                else self.optimizer_cfg.lr
            )

    def on_before_zero_grad(self, *args, **kwargs):
        if self.train_cfg.use_ema:
            self.ema.update_parameters(self.denoiser)

    def on_save_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"].keys()):
            if "temp_module" in key or "FID" in key:
                del checkpoint["state_dict"][key]
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"].keys()):
            if "temp_module" in key or "FID" in key:
                del checkpoint["state_dict"][key]

    def set_timesteps(self, num=None):
        num_inference_timesteps = (
            self.train_cfg.num_inference_timesteps if num is None else num
        )
        if self.global_rank == 0:
            print("Setting Max Timesteps T: ", num_inference_timesteps)

        self.scheduler.set_timesteps(num_inference_timesteps)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.set_timesteps()

    def on_train_batch_start(self, batch, batch_idx):
        self.set_timesteps(self.model_cfg.scheduler.num_train_timesteps)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.set_timesteps(self.test_cfg.num_inference_timesteps)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.set_timesteps()

    def on_validation_batch_end(self, batch, batch_idx, dataloader_idx=0):
        if self.global_rank == 0:
            print("Setting Max Timesteps T: ", self.test_cfg.num_inference_timesteps)
        self.scheduler.set_timesteps(self.model_cfg.scheduler.num_train_timesteps)

    def is_noise_loggable(self):
        return True

    @torch.compile
    def forward(self, x_t, ts, **kwargs):
        """
        Forward pass of the model.
        """

        denoiser_pred = self.denoiser(x_t, ts, **kwargs, return_dict=False)[0]
        return denoiser_pred

    def training_step(self, batch, batch_idx):

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
            self.log("step_tracker", self.step_tracker.get_step())

        x = batch["image"]

        B, C, H, W = x.shape

        t = torch.rand(B, device=x.device)

        # NOTE: this trick is used generally in FM codes, not sure how important it is.
        if self.train_cfg.t_eps is not None:
            t = t * (1 - self.train_cfg.t_eps) + self.train_cfg.t_eps

        if self.train_cfg.discrete_t:
            t = t * (self.model_cfg.scheduler.num_train_timesteps - 1)

        # to be used for any extra information needed to calculate some losses
        # each loss function will search for the extra data it needs
        extra_data = edict()
        if self.model_cfg.use_noise_predictor:
            noise, data = self.noise_predictor(x, class_labels=batch["label"])
            extra_data.update(data)
            noise_frobenius = torch.norm(noise, p="fro")
            self.log("stats/noise_frobenius", noise_frobenius)

            if "input_random_noise" in data:
                random_noise_frobenius = torch.norm(data.input_random_noise, p="fro")
                self.log("stats/original_noise_frobenius", random_noise_frobenius)

        else:
            noise = torch.randn_like(x).to(x.device)

        forward_kwargs = {}
        if self.train_cfg.activate_w_mode:
            assert (
                self.model_cfg.use_noise_predictor
            ), "w mode is only supported for noise predictor training"
            # sample a w uniformly from [0, 1]

            if self.train_cfg.discrete_w:
                # sample w from a discrete set of values
                w = torch.randint(
                    0,
                    self.train_cfg.num_w_values + 1,
                    (B,),
                    device=x.device,
                    dtype=torch.long,
                )
            else:
                w = torch.rand(B, device=x.device)

            if self.model_cfg.denoiser.activate_w_mode:
                forward_kwargs["w"] = w
            # interpolate standard_gaussian noise with predicted noise
            extra_data["predicted_noise"] = noise
            random_noise = torch.randn_like(noise).to(x.device)
            extra_data["w_random_noise"] = random_noise
            w_expanded = w.view(B, 1, 1, 1)
            if self.train_cfg.discrete_w:
                w_expanded = w_expanded.float() / self.train_cfg.num_w_values

            if self.train_cfg.interpolate_distrubutions:

                mean, log_var = extra_data["mean"], extra_data["log_var"]
                var = torch.exp(log_var)
                mean = w_expanded * mean
                var = w_expanded * var + (1 - w_expanded) * 1.0
                std = torch.sqrt(var)
                noise = mean + std * random_noise
            else:
                noise = w_expanded * noise + (1 - w_expanded) * random_noise
        
        if self.train_cfg.block_diffusion_gradient_to_noise_predictor:
            noisy_x = self.scheduler.add_noise(x, noise.detach(), t)
        else:
            noisy_x = self.scheduler.add_noise(x, noise, t)

        if self.train_cfg.class_conditional:
            forward_kwargs["class_labels"] = batch["label"]

        if self.train_cfg.use_cfg:
            forward_kwargs["cond_mask"] = torch.where(
                torch.rand(B, device=x.device) < self.train_cfg.cfg_p_cond, 0.0, 1.0
            )

        if self.train_cfg.use_torch_compile:
            denoiser_pred = self.forward(noisy_x, t, **forward_kwargs)
        else:
            denoiser_pred = self.denoiser(
                noisy_x, t, **forward_kwargs, return_dict=False
            )[0]

        # log noise and predicted noise next to each other for visualization
        if (
            self.train_cfg.log_predictions
            and batch_idx % self.train_cfg.log_predictions_interval == 0
            and self.global_rank == 0
        ):
            if self.is_noise_loggable():
                denoised_image = self.scheduler.step(
                    denoiser_pred[0], t[0], noisy_x[0]
                ).pred_original_sample
            else:
                denoised_image = self.scheduler.get_denoised(
                    denoiser_pred[0], t[0], noisy_x[0]
                )

            vis_image = add_label(unnormalize(x[0]), "Image")
            vis_noise = add_label(unnormalize(noise[0]), "Noise Predictor")
            noisy_x_vis = add_label(unnormalize(noisy_x[0]), "Noisy Image")
            vis_pred_noise = add_label(unnormalize(denoiser_pred[0]), "Denoiser")
            vis_denoised_image = add_label(
                unnormalize(denoised_image), "Denoised Image"
            )

            full_vis = hcat(
                vis_image, vis_noise, noisy_x_vis, vis_pred_noise, vis_denoised_image
            )

            self.logger.log_image(
                "Predictions",
                [prep_image(add_border(full_vis))],
                step=self.step_tracker.get_step(),
                caption=[f"Noise level: {t[0]}"],
            )

        if self.model_cfg.scheduler.prediction_type == "epsilon":
            loss = F.mse_loss(denoiser_pred.float(), noise.float(), reduction="mean")
        elif self.model_cfg.scheduler.prediction_type == "sample":
            loss = F.mse_loss(denoiser_pred.float(), x.float(), reduction="mean")
        elif self.model_cfg.scheduler.prediction_type == "v_prediction":
            if self.train_cfg.block_diffusion_gradient_to_noise_predictor:
                velocity = self.scheduler.get_velocity(x, noise.detach(), t)    
            else:
                velocity = self.scheduler.get_velocity(x, noise, t)
            loss = F.mse_loss(denoiser_pred.float(), velocity.float(), reduction="mean")
        else:
            raise ValueError("Invalid prediction type")

        self.log("info/global_step", self.global_step)
        self.log("loss/diffusion", loss)
        total_loss = loss
        if self.model_cfg.aux_losses is not None:
            for loss_fn in self.aux_losses:
                loss = loss_fn(x, noise, denoiser_pred, t, extra_data)
                self.log(f"loss/{loss_fn.cfg.name}", loss)
                total_loss += loss * loss_fn.cfg.weight

        self.log("loss/total", total_loss)

        if self.global_rank == 0:
            print(f"Train Step {self.global_step}: Loss {total_loss.item():.6f}")

        del extra_data

        return total_loss

    def step(self, x_t, ts, class_labels=None, w=None):

        B = x_t.shape[0]

        if self.train_cfg.use_ema:
            model = self.ema
        else:
            model = self.denoiser

        kwargs = {"class_labels": class_labels}
        if self.train_cfg.activate_w_mode and self.model_cfg.denoiser.activate_w_mode:
            kwargs["w"] = w

        if self.train_cfg.use_cfg:
            kwargs["cond_mask"] = torch.ones(B, device=x_t.device)

        t = ts
        denoiser_pred = model(x_t, t, **kwargs, return_dict=False)[0]

        if self.train_cfg.use_cfg:
            kwargs["cond_mask"] = torch.zeros(B, device=x_t.device)
            uncond_pred = model(
                x_t,
                t,
                **kwargs,
                return_dict=False,
            )[0]
            cfg_scale = self.train_cfg.cfg_scale
            denoiser_pred = (1 + cfg_scale) * denoiser_pred - cfg_scale * uncond_pred

        denoised_x = self.scheduler.step(denoiser_pred, ts, x_t).prev_sample

        return denoised_x

    def get_source_sample(self, x, class_labels=None, w=None):
        if (
            self.model_cfg.use_noise_predictor
            and self.model_cfg.noise_predictor.input_mode == "noise_only"
        ):
            x_0, data = self.noise_predictor(x, class_labels=class_labels)
            if self.train_cfg.activate_w_mode:
                w_expanded = w.view(x.shape[0], 1, 1, 1)
                if self.train_cfg.discrete_w:
                    w_expanded = w_expanded.float() / self.train_cfg.num_w_values

                if self.train_cfg.interpolate_distrubutions:
                    mean, log_var = data["mean"], data["log_var"]
                    var = torch.exp(log_var)
                    mean = w_expanded * mean
                    var = w_expanded * var + (1 - w_expanded) * 1.0
                    std = torch.sqrt(var)
                    x_0 = mean + std * torch.randn_like(mean).to(x.device)
                else:
                    x_0 = w_expanded * x_0 + (1 - w_expanded) * torch.randn_like(
                        x_0
                    ).to(x.device)
        else:
            x_0 = torch.randn_like(x).to(x.device)
            data = edict()

        return x_0, data

    def sample(self, x_t, class_labels=None, w=None, return_trajectories=False):

        traj_subset = 8
        if return_trajectories:
            traj_subset = x_t.shape[0]

        # class labels might be needed for the noise predictor, but not for the
        # denoiser, if that's the case, we set it to None
        if not self.train_cfg.class_conditional:
            class_labels = None

        if self.train_cfg.use_ema:
            model = self.ema
        else:
            model = self.denoiser
        model_kwargs = {"class_labels": class_labels}
        if self.train_cfg.activate_w_mode and self.model_cfg.denoiser.activate_w_mode:
            model_kwargs["w"] = w

        x_t, trajectories, nfe = self.scheduler.sample(
            x0=x_t,
            model=model,
            model_kwargs=model_kwargs,
            solver_kwargs={},
            method=self.model_cfg.scheduler.solver,
        )

        return x_t, trajectories[:, :traj_subset], nfe

    # @rank_zero_only
    def on_validation_start(self):

        if self.train_cfg.optimized_fid:
            self.FID = OptimizedFID(
                dataset_name=self.trainer.datamodule.dataset_cfg.name,
                normalize=True,
            ).to(self.device)
        else:
            self.FID = FrechetInceptionDistance(normalize=True).to(self.device)

    # @rank_zero_only
    def validation_step(self, batch, batch_idx):

        x = batch["image"]

        kwargs = {"class_labels": batch["label"]}

        # NOTE: for now in evaluation
        if self.train_cfg.activate_w_mode:
            if self.train_cfg.discrete_w:
                # sample w from a discrete set of values
                w = torch.randint(
                    0,
                    self.train_cfg.num_w_values + 1,
                    (x.shape[0],),
                    device=x.device,
                    dtype=torch.long,
                )
            else:
                w = torch.rand(x.shape[0], device=x.device)
            kwargs["w"] = w

        x0, data = self.get_source_sample(x, **kwargs)

        x_sampled, noise_levels, nfe = self.sample(x0, **kwargs)

        self.FID.update(unnormalize(x), real=True)
        self.FID.update(unnormalize(x_sampled), real=False)
        self.log(
            "val/fid", self.FID, on_step=False, on_epoch=True, metric_attribute="FID"
        )

        # visualizations
        if batch_idx < 5:
            vis_real = add_label(hcat(*unnormalize(x)), "Real")
            vis_sampled = add_label(hcat(*unnormalize(x_sampled)), "Sampled")
            self.logger.log_image(
                "Generations",
                [prep_image(add_border(vcat(vis_real, vis_sampled)))],
                step=self.step_tracker.get_step(),
                caption=["Real vs Sampled"],
            )

            self.logger.log_image(
                "noise-levels",
                [
                    prep_image(
                        add_border(hcat(*[vcat(*unnormalize(x)) for x in noise_levels]))
                    )
                ],
                step=self.step_tracker.get_step(),
                caption=["Noise Levels"],
            )

    # @rank_zero_only
    def on_validation_end(self):

        fid = self.FID.compute()
        if self.global_rank == 0:
            print(f"Validation Step {self.global_step}: FID {fid:.4f}")

    def on_test_start(self):

        if self.test_cfg.optimized_fid:
            self.FID = OptimizedFID(
                dataset_name=self.trainer.datamodule.dataset_cfg.name,
                normalize=True,
            ).to(self.device)
        else:
            self.FID = FrechetInceptionDistance(normalize=True).to(self.device)

        if self.test_cfg.compute_is:
            self.IS = InceptionScore(
                normalize=True,
            ).to(self.device)

        self.scheduler.set_timesteps(self.test_cfg.num_inference_timesteps)
        print("Setting Max Timesteps T: ", self.test_cfg.num_inference_timesteps)

        self.scheduler.set_solver(self.test_cfg.ode_solver, self.test_cfg.use_scipy)

        if self.test_cfg.curvature:
            self.curvatures = []

        self.num_generated_images = 0
        self.nfes = []

    def on_test_end(self):

        output_dir = self.test_cfg.output_path
        print("Num Generated Samples: ", self.num_generated_images)

        if self.test_cfg.curvature:
            if self.trainer.world_size > 1:
                self.curvatures = self.all_gather(self.curvatures)
                
                self.curvatures = [x.flatten() for x in self.curvatures]

            self.curvatures = np.concatenate([x.cpu().numpy() for x in self.curvatures])

            c_results = {
                "mean_curvature": np.mean(self.curvatures).item(),
                "std_curvature": np.std(self.curvatures).item(),
                "num_samples": self.test_cfg.num_samples,
                "nfes": self.test_cfg.num_inference_timesteps,
            }

            with open(os.path.join(output_dir, "curvature.json"), "w") as f:
                json.dump(c_results, f)
            with open(os.path.join(output_dir, "curvature_values.json"), "w") as f:
                json.dump(self.curvatures.tolist(), f)

            return

        result = {}

        fid = self.FID.compute().item()
        result["FID"] = fid

        if self.test_cfg.compute_is:
            is_score_mean, is_score_std = self.IS.compute()
            result["IS"] = is_score_mean.item()
            result["IS_std"] = is_score_std.item()
            print(
                f"Test Step {self.global_step}: IS {is_score_mean:.4f} ± {is_score_std:.4f}"
            )

        result["num_inference_timesteps"] = self.test_cfg.num_inference_timesteps

        if self.trainer.world_size > 1:
            self.nfes = self.all_gather(self.nfes)
            # a list of tensors on cuda, bring to cpu and concatenate
            self.nfes = np.concatenate([x.cpu().numpy() for x in self.nfes]).tolist()
        result["mean_nfe"] = sum(self.nfes) / len(self.nfes)
        if self.test_cfg.w is not None:
            result["w"] = self.test_cfg.w

        result["ode_solver"] = self.test_cfg.ode_solver

        result["v1"] = "v1"

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(result, f)

        with open(os.path.join(output_dir, "nfes.json"), "w") as f:
            json.dump(self.nfes, f)

        print(f"Test Step {self.global_step}: FID {fid:.4f}")

    def test_step(self, batch, batch_idx):

        if self.num_generated_images >= self.test_cfg.num_samples:
            return

        x = batch["image"]
        B = x.shape[0]
        class_labels = batch["label"]

        if self.num_generated_images + B > self.test_cfg.num_samples:
            B = self.test_cfg.num_samples - self.num_generated_images
            x = x[:B]
            class_labels = class_labels[:B] if class_labels is not None else None
            if B == 1:
                x = x.unsqueeze(0)
                class_labels = class_labels.unsqueeze(0)
            assert x.ndim == 4
            assert class_labels.ndim == 1

        if B == 0:
            return
        self.num_generated_images += B

        if self.test_cfg.curvature:
            self.compute_curvature(B)
            return

        self.FID.update(unnormalize(x), real=True)

        kwargs = {"class_labels": batch["label"]}

        if self.train_cfg.activate_w_mode:
            if self.test_cfg.w is None:
                if self.train_cfg.discrete_w:
                    w = torch.randint(
                        0,
                        self.train_cfg.num_w_values,
                        (B,),
                        device=x.device,
                        dtype=torch.long,
                    )
                else:
                    w = torch.rand(B, device=x.device)
                kwargs["w"] = w
            else:
                kwargs["w"] = torch.full((B,), self.test_cfg.w, device=x.device)

        x0, data = self.get_source_sample(x, **kwargs)
        x_sampled, noise_levels, nfe = self.sample(x0, **kwargs)

        self.FID.update(unnormalize(x_sampled), real=False)
        if self.test_cfg.compute_is:
            self.IS.update(unnormalize(x_sampled))

        self.nfes.extend([nfe])

        if self.test_cfg.log_images:
            vis_real = add_label(vcat(*unnormalize(x)), "Real")
            vis_sampled = add_label(vcat(*unnormalize(x_sampled)), "Sampled")
            path = os.path.join(
                self.test_cfg.output_path, f"generations/batch_{batch_idx}.png"
            )

            save_image(add_border(hcat(vis_real, vis_sampled)), path)
            path = os.path.join(
                self.test_cfg.output_path, f"noise_levels/batch_{batch_idx}.png"
            )
            save_image(
                add_border(hcat(*[vcat(*unnormalize(e)) for e in noise_levels])), path
            )

        if self.test_cfg.generate_for_fid:
            # save each image in the path
            idx = 0
            folder_path = os.path.join(self.test_cfg.output_path, f"gen_for_fid")
            random_string = str(np.random.randint(0, 100000))
            for i in range(B):
                path = os.path.join(
                    folder_path, f"image_{batch_idx * 512 + idx}_{random_string}.png"
                )
                save_image(unnormalize(x_sampled[i]), path)
                idx += 1

    def compute_curvature(self, batch_size):

        if self.test_cfg.curvature_cfg.num_classes is not None:
            class_labels = torch.randint(
                0,
                self.test_cfg.curvature_cfg.num_classes,
                (batch_size,),
                device=self.device,
            )
        else:
            class_labels = None

        kwargs = {"class_labels": class_labels}
        if self.train_cfg.activate_w_mode:
            if self.test_cfg.w is not None:
                w = torch.full(
                    (batch_size,),
                    self.test_cfg.w,
                    device=self.device,
                )
            else:
                if self.train_cfg.discrete_w:
                    w = torch.randint(
                        0,
                        self.test_cfg.w,
                        (batch_size,),
                        device=self.device,
                        dtype=torch.long,
                    )
                else:
                    w = torch.rand(batch_size, device=self.device)
            kwargs["w"] = w

        dummy = torch.zeros(
            (batch_size, *self.test_cfg.curvature_cfg.data_shape),
            device=self.device,
        )

        x0, _ = self.get_source_sample(dummy, **kwargs)

        if not self.train_cfg.class_conditional:
            kwargs["class_labels"] = None

        _, traj, _ = self.sample(
            x0, return_trajectories=True, **kwargs
        )  # *, (T + 1, B, C, H, W), *

        dt = 1.0 / self.test_cfg.num_inference_timesteps
        v_ref = traj[-1] - traj[0]  # (B, C, H, W)
        vs = (traj[1:] - traj[:-1]) / dt  # (T, B, C, H, W)
        # compute curvature
        curvature = torch.mean(
            (v_ref.unsqueeze(0).expand_as(vs) - vs) ** 2, dim=(0, 2, 3, 4)
        ) 

        self.curvatures.extend([curvature])

    def configure_optimizers(self):

        if self.optimizer_cfg.name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer_cfg.name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer {self.optimizer_cfg.name} not supported.")

        if self.optimizer_cfg.scheduler is not None:
            if self.optimizer_cfg.scheduler.name == "cosine_with_warmup":
                lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=self.optimizer_cfg.scheduler.num_warmup_steps,
                    num_training_steps=self.max_steps,
                )
            elif self.optimizer_cfg.scheduler.name == "constant_with_warmup":
                lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=self.optimizer_cfg.scheduler.num_warmup_steps,
                )
            elif self.optimizer_cfg.scheduler.name == "linear":
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.optimizer_cfg.scheduler.frequency,
                    gamma=0.1,
                )
            else:
                raise ValueError(
                    f"Scheduler {self.optimizer_cfg.scheduler.name} not supported."
                )

        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.optimizer_cfg.scheduler.interval,
                "frequency": self.optimizer_cfg.scheduler.frequency,
            },
        }
