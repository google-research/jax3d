# Copyright 2022 The jax3d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image utils."""

from absl import app
import imageio
import jax.numpy as jnp
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.utils.typing import PathLike, f32  # pylint: disable=g-multiple-import
import mediapy
import numpy as np
import PIL.Image


def imread(path: PathLike) -> np.ndarray:
  """Load image from the given path.

  Like `imageio.imread` but with GCS compatibility.

  Args:
    path: Image path to read.

  Returns:
    img: The open image.
  """
  path = j3d.Path(path)
  if _is_tiff(path):  # tiff requires pre-caching the file in-memory
    open_cm = j3d.utils.open_seekable(path, 'rb')
    with open_cm as f:
      img = imageio.imread(f, format=path.suffix)
  else:
    img = mediapy.read_image(path)
  # For consistency, single-channel images are (h, w, 1)
  return img[..., None] if img.ndim == 2 else img


def imwrite(path: PathLike, img: np.ndarray) -> None:
  """Save image to the given path.

  Like `imageio.imwrite` but:
   * With GCS compatibility.
   * Does not downcast uint16 -> uint8 png images

  Args:
    path: Destination path
    img: Image to save
  """
  path = j3d.Path(path)
  if _is_tiff(path):  # tiff requires pre-caching the file in-memory
    open_cm = j3d.utils.open_seekable(path, 'wb')
  else:
    open_cm = path.open('wb')

  write_kwargs = dict()
  if path.suffix.lower() == '.png':
    if img.dtype == np.uint16:  # Avoid downcasting uint16 -> uint8
      # TODO(epot): Add support for multi-channels uint16
      if img.ndim == 3 and img.shape[-1] > 1:
        raise ValueError(
            f'Only single-channel image supported for uint16. Got: {img.shape}'
        )
      write_kwargs = dict(prefer_uint8=False)
    elif img.dtype != np.uint8:
      raise ValueError(
          'To avoid implicit down/upcasting, dtype should be uint8/uint16. Got '
          f'{img.dtype}'
      )

  with open_cm as f:
    imageio.imwrite(f, img, format=path.suffix, **write_kwargs)


def _is_tiff(path: j3d.Path) -> bool:
  """Some files format requires pr."""
  return path.suffix.lower() in ('.tif', '.tiff')


def generate_canvas(height: int,
                    width: int,
                    color0: f32[3],
                    color1: f32[3],
                    checker_size: int = 8) -> f32['height width 3']:
  """Generates a checkerboard canvas."""
  rows = jnp.floor_divide(jnp.linspace(0, height-1, height), checker_size)
  cols = jnp.floor_divide(jnp.linspace(0, width-1, width), checker_size)
  canvas = rows[:, None] + cols[None, :]
  canvas = jnp.mod(canvas, 2)
  canvas = jnp.where(canvas[..., None], color0, color1)
  return canvas


def apply_canvas(foreground: f32['h w 3'], alpha: f32['h w 1']) -> f32:
  """Blends foreground onto a checkerboard canvas, according to alpha [0; 1].

  Args:
    foreground: foreground image with values [0; alpha]
    alpha: alpha image with values [0; 1]
      0 = transparent foreground
      1 = opaque foreground

  Returns:
    Foreground alpha-composited on the background canvas.
  """
  color0 = jnp.asarray([0.7] * 3, dtype=foreground.dtype)
  color1 = jnp.asarray([0.85] * 3, dtype=foreground.dtype)
  canvas = generate_canvas(foreground.shape[0], foreground.shape[1],
                           color0, color1)
  return foreground + canvas * (1 - alpha)


def _preload_modules() -> None:
  """Pre-load image libs to avoid race-condition in multi-thread."""
  PIL.Image.preinit()


# Automatically execute the pre-loading.
app.call_after_init(_preload_modules)
