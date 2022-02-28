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

"""Numpy utils."""

from __future__ import annotations

from typing import Any

from etils import enp
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
from jax3d.visu3d import array_dataclass

# Maybe some of those could live in `enp` ?


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
