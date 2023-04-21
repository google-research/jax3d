# Copyright 2023 The jax3d Authors.
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

"""Provide utils from Kubric.

# Copyright 2022 The Kubric Authors.
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
"""


import colorsys
import contextlib
import logging
import multiprocessing
from typing import Optional, Union

from etils import epath
import numpy as np
import png
import tensorflow as tf

PathLike = Union[str, epath.Path]

logger = logging.getLogger(__name__)


def as_path(path: PathLike) -> epath.Path:
  """Convert str or pathlike object to epath.Path.

  Instead of pathlib.Path, we use `epath` because it transparently
  supports paths to GCS buckets such as "gs://kubric-public/GSO".

  Args:
    path: str or pathlike object.

  Returns:
    epath.Path object.
  """
  return epath.Path(path)


@contextlib.contextmanager
def gopen(filename: PathLike, mode: str = "w"):
  """Simple contextmanager to open a file using tf.io.gfile (and ensure the parent dir exists)."""
  filename = as_path(filename)
  if mode[0] in {"w", "a"}:  # if writing mode ...
    # ensure directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing to '%s'", filename)
  with tf.io.gfile.GFile(str(filename), mode=mode) as fp:
    yield fp


def write_png(data: np.ndarray, filename: PathLike) -> None:
  """Writes data as a png file (and convert datatypes if necessary)."""

  if data.dtype in [np.uint32, np.uint64]:
    max_value = np.amax(data)
    if max_value > 65535:
      logger.warning("max_value %d exceeds uint16 bounds for %s.",
                     max_value, filename)
      raise ValueError(
          f"max value of {max_value} exceeds uint16 bounds for {filename}")
    data = data.astype(np.uint16)
  elif data.dtype in [np.float32, np.float64]:
    min_value = np.amin(data)
    max_value = np.amax(data)
    if min_value < 0.0 or max_value > 1.0:
      raise ValueError(
          f"Values need to be in range [0, 1] but got [{min_value}, {max_value}]"
          f" for {filename}")
    data = (data * 65535).astype(np.uint16)
  elif data.dtype in [np.uint8, np.uint16]:
    pass
  else:
    raise NotImplementedError(f"Cannot handle {data.dtype}.")

  bitdepth = 8 if data.dtype == np.uint8 else 16

  assert data.ndim == 3, data.shape
  height, width, channels = data.shape
  greyscale = (channels == 1)
  alpha = (channels == 4)
  w = png.Writer(width=width, height=height, greyscale=greyscale,
                 bitdepth=bitdepth, alpha=alpha)

  if channels == 2:
    # Pad two-channel images with a zero channel.
    data = np.concatenate([data, np.zeros_like(data[:, :, :1])], axis=-1)

  # pypng expects 2d arrays
  # see https://pypng.readthedocs.io/en/latest/ex.html#reshaping
  data = data.reshape(height, -1)
  with gopen(filename, "wb") as fp:
    w.write(fp, data)


def write_palette_png(data: np.ndarray, filename: PathLike,
                      palette: Optional[np.ndarray] = None):
  """Writes grayscale data as pngs to path using a fixed palette.

  (e.g. for segmentations)

  Args:
    data: grayscale data
    filename: file path to write
    palette: list of evenly spaced colors in HSL space starting with black

  Returns:
    Grayscale data converted to png with a fixed palette.
  """
  assert data.ndim == 3, data.shape
  height, width, channels = data.shape
  assert channels == 1, "Must be grayscale"

  if data.dtype in [np.uint16, np.uint32, np.uint64]:
    max_value = np.amax(data)
    if max_value > 255:
      logger.warning("max_value %d exceeds uint bounds for %s.",
                     max_value, filename)
    data = data.astype(np.uint8)
  elif data.dtype == np.uint8:
    pass
  else:
    raise NotImplementedError(f"Cannot handle {data.dtype}.")

  if palette is None:
    palette = hls_palette(np.max(data) + 1)

  w = png.Writer(width=width, height=height, palette=palette, bitdepth=8)
  with gopen(filename, "wb") as fp:
    w.write(fp, data[:, :, 0])


def multi_write_image(data: np.ndarray, path_template: str, write_fn=write_png,
                      max_write_threads=16, **kwargs):
  """Write a batch of images to a series of files using a ThreadPool.

  Args:
    data: Batch of images to write. Shape = (batch_size, height, width,a
      channels)
    path_template: a template for the filenames (e.g. "rgb_frame_{:05d}.png").
      Will be formatted with the index of the image.
    write_fn: the function used for writing the image to disk.
      Must take an image array as its first and a filename as its second
      argument. May take other keyword arguments. (Defaults to the write_png
      function)
    max_write_threads: number of threads to use for writing images.
      (default = 16)
    **kwargs: additional kwargs to pass to the write_fn.
  """
  num_threads = min(data.shape[0], max_write_threads)
  with multiprocessing.pool.ThreadPool(num_threads) as pool:
    args = [(img, path_template.format(i)) for i, img in enumerate(data)]

    def write_single_image_fn(arg):
      write_fn(*arg, **kwargs)

    for result in pool.imap_unordered(write_single_image_fn, args):
      if isinstance(result, Exception):
        logger.warning("Exception while writing image %s", result)

    pool.close()
    pool.join()


def write_rgb_batch(data, directory, file_template="rgb_{:05d}.png",
                    max_write_threads=16):
  assert data.ndim == 4 and data.shape[-1] == 3, data.shape
  path_template = str(as_path(directory) / file_template)
  multi_write_image(data, path_template, write_fn=write_png,
                    max_write_threads=max_write_threads)


def write_segmentation_batch(data, directory,
                             file_template="segmentation_{:05d}.png",
                             max_write_threads=16):
  assert data.ndim == 4 and data.shape[-1] == 1, data.shape
  assert data.dtype in [np.uint8, np.uint16, np.uint32, np.uint64], data.dtype
  path_template = str(as_path(directory) / file_template)
  palette = hls_palette(np.max(data) + 1)
  multi_write_image(data, path_template, write_fn=write_palette_png,
                    max_write_threads=max_write_threads, palette=palette)

# FROM PLOTTING


def hls_palette(n_colors, first_hue=0.01, lightness=.5, saturation=.7):
  """Get a list of colors where the first is black and the rest are evenly spaced in HSL space."""
  hues = np.linspace(0, 1, int(n_colors) + 1)[:-1]
  hues = (hues + first_hue) % 1
  palette = [(0., 0., 0.)] + [
      colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues]
  return np.round(np.array(palette) * 255).astype(np.uint8)
