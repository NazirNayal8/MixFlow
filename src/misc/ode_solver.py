import torch
import warnings

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    warnings.warn(
        "torchdiffeq not found. Install with 'pip install torchdiffeq'. "
        "Only 'euler', 'heun', and 'midpoint' methods implemented manually will be available. "
        "NFE counting will be based on manual step counts for these."
    )
    odeint = None
    TORCHDIFFEQ_AVAILABLE = False

# Helper class to count function evaluations
class _ODEFuncWrapper:
    """Wraps the user-provided ODE function to count evaluations."""
    def __init__(self, func):
        self.func = func
        self.nfe = 0

    def __call__(self, t, y):
        self.nfe += 1
        return self.func(t, y)

    def reset(self):
        self.nfe = 0

class ODESolver:
    """
    A unified interface for solving Ordinary Differential Equations (ODEs)
    using various methods, suitable for tasks like sampling in flow matching.

    Supports methods via torchdiffeq ('euler', 'heun', 'midpoint', 'rk45', 'dopri5')
    or falls back to manual implementations for 'euler', 'heun', 'midpoint' if
    torchdiffeq is unavailable. Returns the solution trajectory and the
    number of function evaluations (NFEs).
    """
    SUPPORTED_METHODS = ['euler', 'manual_heun', 'heun', 'midpoint', 'rk45', 'dopri5']
    # Methods known to be adaptive in torchdiffeq [[6]]
    ADAPTIVE_METHODS = ['rk45', 'dopri5']
    # Methods available in torchdiffeq (mapping rk45 to dopri5) [[7]]
    TORCHDIFFEQ_METHODS = ['euler', 'heun2', 'midpoint', 'dopri5']


    def __init__(self, method, use_scipy=False):
    
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods are: {self.SUPPORTED_METHODS}"
            )
        self.use_scipy = use_scipy
        # Map 'rk45' to 'dopri5' as torchdiffeq uses dopri5 for adaptive RK4(5)
        self.method_name = 'dopri5' if method == 'rk45' else method
        self.method_name = 'heun2' if method == 'heun' else self.method_name

        if self.method_name in self.ADAPTIVE_METHODS and not TORCHDIFFEQ_AVAILABLE:
             raise ImportError(
                 f"Method '{self.method_name}' requires torchdiffeq for adaptive stepping. "
                 "Please install it using 'pip install torchdiffeq'."
             )

        # Determine if torchdiffeq will be used for the chosen method
        self._use_torchdiffeq = TORCHDIFFEQ_AVAILABLE and (self.method_name in self.TORCHDIFFEQ_METHODS)

        print(f"ODESolver initialized with method: {self.method_name} (requested: {method}). "
              f"Using torchdiffeq: {self._use_torchdiffeq}")

    def _euler_step(self, func_wrapper, y0, t0, dt, index, num_steps):
        """Performs a single Euler step."""
        # func_wrapper handles NFE counting
        dy_dt = func_wrapper(t0, y0)
        y1 = y0 + dy_dt * dt
        return y1

    def _heun_step(self, func_wrapper, y0, t0, dt, index, num_steps):
        """Performs a single Heun step (Predictor-Corrector)."""
        # func_wrapper handles NFE counting (2 evals per step)
        k1 = func_wrapper(t0, y0)
        if index < num_steps - 1:
            y_pred = y0 + k1 * dt
            k2 = func_wrapper(t0 + dt, y_pred)
            y1 = y0 + (k1 + k2) * dt / 2.0
        else:
            # Last step, use k1 only
            y1 = y0 + k1 * dt
        return y1

    def _midpoint_step(self, func_wrapper, y0, t0, dt, index, num_steps):
        """Performs a single Midpoint step."""
        # func_wrapper handles NFE counting (2 evals per step)
        k1 = func_wrapper(t0, y0) # Slope at the start
        y_mid_est = y0 + k1 * dt / 2.0 # Estimate y at midpoint
        k2 = func_wrapper(t0 + dt / 2.0, y_mid_est) # Slope at the midpoint
        y1 = y0 + k2 * dt # Update using midpoint slope
        return y1

    def _manual_integrate(self, func_wrapper, y0, t_span):
        """
        Manually integrates using Euler, Heun or Midpoint for fixed steps.
        Assumes t_span represents evenly spaced time points.
        """
        if self.method_name not in ['euler', 'heun', 'midpoint', 'manual_heun']:
             # This path should ideally not be reached if __init__ checks pass
             raise NotImplementedError(
                 f"Manual integration only implemented for 'euler', 'heun', 'midpoint'. "
                 f"Method '{self.method_name}' requires torchdiffeq."
             )

        y_trajectory = [y0]
        y_current = y0

        # Select the appropriate step function
        if self.method_name == 'euler':
            step_func = self._euler_step
        elif self.method_name == 'manual_heun':
            step_func = self._heun_step
        else: # midpoint
            step_func = self._midpoint_step

        num_steps = len(t_span) - 1
        for i in range(num_steps):
            t0 = t_span[i]
            t1 = t_span[i+1]
            # Ensure dt is compatible with potential tensor times
            dt = t1 - t0
            y_current = step_func(func_wrapper, y_current, t0, dt, i, num_steps)
            y_trajectory.append(y_current)

        # NFE count is tracked by func_wrapper automatically
        return torch.stack(y_trajectory)


    def sample(self, eval_func, x0: torch.Tensor, t_span: torch.Tensor, **kwargs):
        """
        Solves the ODE dy/dt = eval_func(t, y) from initial state x0 over time points t_span.

        Args:
            eval_func (callable): A function that computes the derivative dy/dt.
                                  It must accept (t, y) where t is a scalar tensor
                                  and y is a tensor of the same shape as x0,
                                  and return a tensor of the same shape as y.
            x0 (torch.Tensor): The initial state tensor (y(t0)).
            t_span (torch.Tensor): A 1D tensor containing the time points
                                   (including the initial time t0) at which
                                   to evaluate the solution. The solver will
                                   return the state y at each time point in t_span.
                                   Must be sorted in ascending order.
            **kwargs: Additional keyword arguments passed to the underlying solver.
                      Common options for torchdiffeq adaptive solvers ('dopri5', 'rk45'):
                      - rtol (float): Relative tolerance.
                      - atol (float): Absolute tolerance.
                      Common options for torchdiffeq fixed solvers ('euler', 'heun', 'midpoint'):
                      - options (dict): e.g., options={'step_size': 0.1} if t_span
                                        only contains start and end times (though
                                        providing all steps in t_span is preferred for fixed methods).

        Returns:
            tuple[torch.Tensor, int]: A tuple containing:
                - solution (torch.Tensor): The solution trajectory.
                    Shape: (len(t_span), *x0.shape).
                    The first dimension corresponds to the time points in t_span.
                - nfe (int): The total number of function evaluations used by the solver.
                    This is most informative for adaptive step methods [[6]].
        """
        # Ensure t_span is on the same device as x0 for compatibility
        if t_span.device != x0.device:
             t_span = t_span.to(x0.device)

        # Wrap the evaluation function to count calls
        func_wrapper = _ODEFuncWrapper(eval_func)

        if self._use_torchdiffeq:
            # Use torchdiffeq's odeint [[1]]
            # Default tolerances for adaptive methods if not provided
            if self.method_name in self.ADAPTIVE_METHODS:
                kwargs.setdefault('rtol', 1e-5)
                kwargs.setdefault('atol', 1e-5)

            if self.use_scipy: 
                solution = odeint(
                    func_wrapper, # Pass the wrapped function
                    x0,
                    t_span,
                    method="scipy_solver",
                    options={"solver": "RK45"},
                    **kwargs
                )
            else:
                # Note: torchdiffeq's odeint itself doesn't directly return NFE.
                # We rely on our wrapper to count calls.
                solution = odeint(
                    func_wrapper, # Pass the wrapped function
                    x0,
                    t_span,
                    method=self.method_name,
                    **kwargs
                )
        elif self.method_name in ['euler', 'heun', 'midpoint', 'manual_heun']:
             # Fallback to manual implementation if torchdiffeq is not available
             # or if specifically requested and torchdiffeq is installed but not used
             warnings.warn(
                 f"Using manual implementation for '{self.method_name}'. "
                 "This assumes fixed step sizes based on t_span and might be slower. "
                 "NFE count will reflect manual step evaluations."
             )
             # Manual methods require fixed steps, check t_span differences
             diffs = t_span[1:] - t_span[:-1]
             # Use torch.isclose for robust floating point comparison
             if len(diffs) > 0 and not torch.all(torch.isclose(diffs, diffs[0])):
                 warnings.warn(
                     f"Manual '{self.method_name}' solver performs best with evenly spaced t_span "
                     "for consistent step sizes. Step sizes vary in the provided t_span."
                 )
             solution = self._manual_integrate(func_wrapper, x0, t_span)
        else:
             # This case should ideally be caught in __init__, but as a safeguard:
             raise RuntimeError(
                 f"Method '{self.method_name}' is not supported without torchdiffeq."
             )

        nfe = func_wrapper.nfe
        return solution, nfe

