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

"""Common image utility functions."""

import io
from typing import IO, Optional, Sequence, Tuple, Union

import frozendict
import jax.numpy as jnp
import numpy as np
import OpenEXR as exr
from PIL import Image
from skimage import exposure

SRGB_GAMMA = 2.4
_EPS = 1e-6

# Default channel names for EXR files with the given number of channels.
_DEFAULT_EXR_CHANNELS = frozendict.frozendict({
    3: 'RGB',
    4: 'RGBA',
})


# Convert linear radiance to/from gamma encoded sRGB values. Note these
# transforms fudge the small portion of the sRGB curve that is linear, but is
# close enough for our intents and purposes. May need to revisit for
# underexposed images though. (Ref: https://en.wikipedia.org/wiki/SRGB)
def np_linear_to_srgb_gamma(image):
  return exposure.adjust_gamma(image, gamma=1.0 / SRGB_GAMMA, gain=1.0)


def np_srgb_gamma_to_linear(image):
  return exposure.adjust_gamma(image, gamma=SRGB_GAMMA, gain=1.0)


def linear_to_srgb_gamma(linear_value):
  """Convert a linear radiance value to SRGB gamma encoded value."""
  return jnp.power(jnp.maximum(linear_value, _EPS), 1.0 / SRGB_GAMMA)


def srgb_gamma_to_linear(gamma_value):
  """Convert an SRGB gamma encoded value to a linear radiance value."""
  return jnp.power(jnp.maximum(gamma_value, _EPS), SRGB_GAMMA)


def image_to_byte_array(image: np.ndarray, image_format: str) -> bytes:
  """Returns encoded image bytes using given format.

  Args:
    image: Image array.
    image_format: Image format string. Valid values are 'PNG' and 'JPEG'.
  """
  image = Image.fromarray(np.squeeze(image))
  image_byte_array = io.BytesIO()
  image.save(image_byte_array, image_format)
  return image_byte_array.getvalue()


def image_to_exr_file(image: np.ndarray,
                      output_file: Union[str, IO[bytes]],
                      channels: Optional[Sequence[str]] = None):
  """Writes an image array as an EXR file.

  Args:
    image: Image array, HWC.
    output_file: Output path or file handle.
    channels: Names of the channels in the image. Length must match the channel
      dimension of the image. 3- and 4-channel images default to 'RGB' and
      'RGBA', respectively.
  """
  height, width, num_channels = image.shape
  if channels is None:
    channels = _DEFAULT_EXR_CHANNELS.get(num_channels)
    if channels is None:
      raise ValueError(
          f'Must specify channels when their count is {num_channels}.')
  if len(channels) != num_channels:
    raise ValueError(f'Image channel count {num_channels} does not match given '
                     f'channels: {channels}')

  if image.dtype != np.float32:
    image = image.astype(np.float32)

  header = exr.Header(width, height)
  header_channels = header['channels']
  channel_set = set(channels)
  for channel in ['R', 'G', 'B']:
    if channel not in channel_set:
      del header_channels[channel]
  for channel in channels:
    if channel not in header_channels:
      header_channels[channel] = exr.Imath.Channel(
          exr.Imath.PixelType(exr.FLOAT))

  channel_data = {}
  for i, name in enumerate(channels):
    channel_data[name] = image[:, :, i].tobytes()

  exr_out = exr.OutputFile(output_file, header)
  exr_out.writePixels(channel_data)


def byte_array_to_image(byte_array: bytes) -> np.ndarray:
  """Convert from JPEG/PNG byte string to numpy image array."""
  return np.array(Image.open(io.BytesIO(byte_array)))


def resize(image_array: np.ndarray,
           size: Tuple[int, int],
           resample: int = Image.BICUBIC) -> np.ndarray:
  """Resizes the image to the specified size.

  Args:
    image_array: Image to be resized.
    size: Desired (W,H) pixel size.
    resample: Resampling method to use.

  Returns:
    Resized image, in array form.
  """
  image = Image.fromarray(image_array)
  image = image.resize(size, resample=resample)
  return np.asarray(image)


def pad_image_to_square(image_array: np.ndarray) -> np.ndarray:
  """Pads an image to be square, returning the original if it is already square.

  The larger of the width and height dimensions is used as the square dimension.

  Args:
    image_array: Image to be padded.

  Returns:
    Square image, a padded copy if not already a square.
  """
  image = Image.fromarray(image_array)

  width, height = image.size
  if width == height:
    return image_array
  dim = max(width, height)

  square_image = Image.new(mode=image.mode, size=(dim, dim))
  if width == dim:
    y0 = (dim - height) // 2
    dst_box = (0, y0)
  else:
    x0 = (dim - width) // 2
    dst_box = (x0, 0)

  square_image.paste(image, dst_box)
  return np.asarray(square_image)
