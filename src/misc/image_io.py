import io
import PIL.Image
from pathlib import Path
from typing import Union
import skvideo.io

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from matplotlib.backends.backend_agg import FigureCanvasAgg

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def get_image_grid(img, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img.cpu(), dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]

    # if C == 1:
    #     fig = PIL.Image.fromarray(img[:, :, 0], 'L')
    # if C == 3:
    #     fig = PIL.Image.fromarray(img, 'RGB')

    return img
   


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4), f"Expected 3 or 4 channels, got {image.shape[0]}."

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


def save_video(
    images: list[FloatImage],
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    # Image.fromarray(prep_image(image)).save(path)
    frames = []
    for image in images:
        frames.append(prep_image(image))

    writer = skvideo.io.FFmpegWriter(path, 
                                     outputdict={'-pix_fmt': 'yuv420p', '-crf': '21', 
                                                 '-vf': f'setpts=1.*PTS'})
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()

def get_hist_image(x, title, bins=50, dpi=100, figsize=(5, 4)):
    
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    
    # Do some plotting here
    ax = fig.add_subplot(111)
    data =torch.clamp(x, min=-5, max=5).detach().cpu().flatten().numpy()
    ax.hist(data, bins=None)
    ax.axis('on')
    # ax.set_xlim(-7, 7)
    # ax.set_ylim(0, 500)
    ax.set_title(title)
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    
    rgba = np.asarray(buf)
    row, col, ch = rgba.shape
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    background=(255,255,255)
    R, G, B = background
    rgb[:,:,0] = r #* a + (1.0 - a) * R
    rgb[:,:,1] = g #* a + (1.0 - a) * G
    rgb[:,:,2] = b #* a + (1.0 - a) * B
    rgb = np.asarray( rgb, dtype='uint8')
    # convert to a NumPy array
    return torch.tensor(rgb/(255.0)).permute(2, 0, 1)