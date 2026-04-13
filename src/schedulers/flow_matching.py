import numpy as np
import torch
from dataclasses import dataclass
from typing import Literal
from easydict import EasyDict as edict
from src.misc.ode_solver import ODESolver


@dataclass
class FlowMatchingCfg:
    name: Literal["flow_matching"]
    solver: str  # used only for validation during training
    num_train_timesteps: int # dummy value, for interface compatibility


def flatten(x):
    return x.detach().cpu().numpy().reshape((-1,))


def unflatten(x, shape, device):
    return torch.from_numpy(x.reshape(shape)).to(device).float()


class FlowMatching:
    """
    A Generic implementation for Flow Matching with multiple solver options.
    """

    def __init__(self, cfg: FlowMatchingCfg):
        self.cfg = cfg
        self.cfg.prediction_type = "v_prediction"
        self.num_steps = None
        self.solver = ODESolver(cfg.solver)

    def set_timesteps(self, num_steps):
        """
        Set the number of steps for the scheduler.
        """
        self.num_steps = num_steps

    def set_solver(self, solver, use_scipy=False):
        """
        Set the solver for the scheduler.
        """
        self.solver = ODESolver(solver, use_scipy=use_scipy)

    def add_noise(self, sample, noise, timestep):
        """
        timestep is expected to be between 0 and 1.

        Different from other diffusion schedulers, this scheduler
        assumes that t=0 is full noise, and t=1 is sample.
        """
        B = sample.shape[0]
        t = timestep.view(B, 1, 1, 1)
        x_t = t * sample + (1 - t) * noise
        return x_t

    def sample(self, x0, model, model_kwargs, solver_kwargs, method):
        """
        x0: the initial sample
        model: the model to use for sampling
        model_kwargs: the kwargs to pass to the model
        solver_lib: the library to use for solving the ODE, most notably rtol and atol (1e-5)
        method: the method to use for sampling
        num_steps: the number of steps to use for sampling
        """

        def v_func(t, x):
            return model(x, t, **model_kwargs, return_dict=False)[0]

        # NOTE: here we add 1 to num_steps because each step in the solver integrates between two time points.
        t_span = torch.linspace(0, 1, self.num_steps + 1).to(x0.device)
        sol, nfe = self.solver.sample(v_func, x0, t_span, **solver_kwargs)

        return sol[-1], sol, nfe

    def get_velocity(self, sample, noise, timestep):
        """
        An adapter for `scale_noise` function in the diffuser library
        in order to unify the interface with the other schedulers.
        """
        return sample - noise

    def step(self, model_output, timestep, sample):
        x1 = sample + (1 - timestep) * model_output
        return edict(pred_original_sample=x1)
