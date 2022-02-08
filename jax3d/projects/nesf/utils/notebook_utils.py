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

"""Utils for colab/jupyter."""

import base64
import io
import traceback
from typing import Any, Optional, Tuple

from jax import numpy as jnp
import numpy as np

__all__ = [
    'display_array_as_img',
]

Array = Any

_MIN_IMG_SHAPE: Tuple[int, int] = (10, 10)


def display_array_as_img() -> None:
  """If called, 2d/3d imgage arrays will be plotted as images in colab/jupyter.

  Usage:

  >>> jax3d.utils.display_array_as_img()
  >>> np.zeros((28, 28, 3))  # Displayed as image

  """
  print('Display big np/jax arrays as image for nicer IPython display')
  # The array type is not exposed in the public API (registering jnp.ndarray
  # does not works), so dynamically extracting the type
  jax_array_cls = type(jnp.zeros(shape=()))  # DeviceArrayBase

  # Register the new representation fo np and jax array
  import IPython  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  formatter = IPython.get_ipython().display_formatter.formatters['text/html']
  formatter.for_type(jax_array_cls, _array_repr_html)
  formatter.for_type(np.ndarray, _array_repr_html)


def _array_repr_html(array: Array) -> Optional[str]:
  """Returns the HTML `<img/>` repr, or `None` if array is not an image."""
  try:
    img = _get_image(array)
    if img is not None:
      return _html_img_repr(img)
    else:
      return None
  except Exception:
    # IPython display silence exceptions, so display it here
    traceback.print_exc()
    raise


def _html_img_repr(img: Array) -> str:
  """Generates the image and returns the HTML `<img />`."""
  from matplotlib.backends import backend_agg  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  import matplotlib.pyplot as plt  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  fig = plt.Figure()
  backend_agg.FigureCanvasAgg(fig)
  plt_image = fig.gca().imshow(img)
  fig.colorbar(plt_image)
  buffer = io.BytesIO()
  fig.savefig(buffer)
  bytes_value = buffer.getvalue()

  # Alternativelly could try using `IPython.display.Image`
  img_str = base64.b64encode(bytes_value).decode('ascii')  # pytype: disable=bad-return-type
  return f'<img src="data:image/png;base64,{img_str}" alt="Img" />'


def _get_image(img: Array) -> Optional[Array]:
  """Returns the normalized img, or `None` if the input is not an image."""
  # Image should have reasonable dimensions
  if (
      len(img.shape) not in {2, 3}
      or img.shape[0] < _MIN_IMG_SHAPE[0]
      or img.shape[1] < _MIN_IMG_SHAPE[1]
  ):
    return None

  if len(img.shape) == 2:
    return img

  assert len(img.shape) == 3
  if img.shape[-1] == 1:
    return img.squeeze(-1)
  elif img.shape[-1] == 3:
    return img
  else:  # More than 3 channels, too big to be displayed
    return None
