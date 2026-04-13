
import torch
from pytorch_lightning.callbacks import Callback
import time


class GPUMemoryMonitor(Callback):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated()
        print(
            f"Batch {batch_idx}: Max GPU Memory Allocated: {max_memory / (1024**2):.2f} MB")
        torch.cuda.reset_peak_memory_stats()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated()
        print(
            f"Batch {batch_idx}: Max GPU Memory Allocated: {max_memory / (1024**2):.2f} MB")
        torch.cuda.reset_peak_memory_stats()


class GradientMonitor(Callback):

    def __init__(self):
        super().__init__()

    def on_after_backward(self, trainer, pl_module):

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.norm().item()}")
        print("\n")


class TrainingSpeedCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Record the start time at the beginning of each training batch
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Calculate the time taken for the current batch
        elapsed_time = time.time() - self.start_time

        # Log the time taken to Weights and Biases
        pl_module.log('train/time_per_iteration', elapsed_time,
                      on_step=True, on_epoch=False)

        # Optionally, you can also print it to the console
        print(f"Time per iteration: {elapsed_time:.4f} seconds")
