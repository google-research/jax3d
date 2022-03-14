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

"""Numpy utils.

And utils intended to work on both `xnp.ndarray` and `v3d.DataclassArray`.
"""

from __future__ import annotations

from typing import Any, Optional

from etils import enp
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
from jax3d.visu3d import array_dataclass
from jax3d.visu3d.typing import Shape

# Maybe some of those could live in `enp` ?


def size_of(shape: Shape) -> int:
  """Returns the size associated with the shape."""
  # TODO(b/198633198): Warning: In TF `bool(shape) == True` for `shape==()`
  if not len(shape):  # pylint: disable=g-explicit-length-test
    size = 1  # Special case because `np.prod([]) == 1.0`
  else:
    size = enp.lazy.np.prod(shape)
  return size


def get_xnp(x: Any) -> enp.NpModule:
  """Returns the np module associated with the given array or DataclassArray."""
  if isinstance(x, array_dataclass.DataclassArray):
    xnp = x.xnp
  elif enp.lazy.is_array(x):
    xnp = enp.lazy.get_xnp(x)
  else:
    raise TypeError(
        f'Unexpected array type: {type(x)}. Could not infer numpy module.')
  return xnp


def is_array(x: Any, xnp: Optional[enp.NpModule] = None) -> bool:
  """Returns whether `x` is an array or DataclassArray.

  Args:
    x: array to check
    xnp: If given, raise an error if the array is from a different numpy module.

  Returns:
    True if `x` is `xnp.ndarray` or `v3d.DataclassArray`
  """
  try:
    infered_xnp = get_xnp(x)
  except TypeError:
    return False
  else:
    if xnp is None:
      return True
    else:
      return infered_xnp is xnp


def normalize(x: FloatArray['*d'], axis: int = -1) -> FloatArray['*d']:
  """Normalize the vector to the unit norm."""
  return x / enp.compat.norm(x, axis=axis, keepdims=True)


def append_row(
    x: FloatArray['*d daxis'],
    value: float,
    *,
    axis: int,  # Axis is required as `row` imply `axis=0` while we want `=-1`
) -> FloatArray['*d daxis+1']:
  """Like `np.append`, but broadcast the value to `x` shape."""
  xnp = enp.get_np_module(x)
  value = xnp.asarray(value)
  if value.ndim == 0:
    shape = list(x.shape)
    shape[axis] = 1
    value = xnp.broadcast_to(value, shape)
  elif value.ndim == 1:
    # TODO(epot): support actual row: append_row(x, [0, 0, 0, 1]). Might require
    # adding a `broadcast_to` which support arbitrary array.
    assert x.shape[axis] == len(value)
    raise NotImplementedError()
  else:
    raise ValueError(
        f'`append_row` does not support appending rank > 1. Got {value.shape}.')
  return xnp.append(x, value, axis=axis)
