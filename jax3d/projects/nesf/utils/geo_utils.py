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

"""Geometric utils."""

from typing import Tuple, Union

import jax.numpy as jnp
from jax3d.projects.nesf.utils.typing import Array, ArrayLike, f32, i32  # pylint: disable=g-multiple-import
import numpy as np


_MinMaxValue = Union[int, float, ArrayLike[Array['d']]]


def interp(
    x: Array['*d'],
    from_: Tuple[_MinMaxValue, _MinMaxValue],
    to: Tuple[_MinMaxValue, _MinMaxValue],
    axis: int = -1,
) -> f32['*d']:
  """Linearly scale the given value by the given range.

  Somehow similar to `np.interp` or `scipy.interpolate.inter1d` with some
  differences like support scaling an axis by a different factors and
  extrapolate values outside the boundaries.

  `from_` and `to` are expected to be `(min, max)` tuples and the function
  interpolate between the two ranges.

  Example: Normalizing a uint8 image to `(-1, 1)`.

  ```python
  img = jnp.array([
      [0, 0],
      [127, 255],
  ])
  img = j3d.interp(img, (0, 255), (0, 1))
  img == jnp.array([
      [-1, -1],
      [0.498..., 1],
  ])
  ```

  `min` and `max` can be either float values or array like structure, in which
  case the numpy broadcasting rules applies (x should be a `Array[... d]` and
  min/max values should be `Array[d]`.

  Example: Converting normalized 3d coordinates to world coordinates.

  `coords[:, 0]` is interpolated from `(0, h)` to `(-1, 1)` and `coords[:, 1]`
  from `(0, w)` to `(-1, 1)`,...

  ```python
  coords = j3d.interp(coords, from_=(-1, 1), to=(0, (h, w, d)), to=(-1, 1))
  ```

  * `coords[:, 0]` is interpolated from `(-1, 1)` to `(0, h)`
  * `coords[:, 1]` is interpolated from `(-1, 1)` to `(0, w)`
  * `coords[:, 2]` is interpolated from `(-1, 1)` to `(0, d)`

  Args:
    x: Array to scale
    from_: Range of x.
    to: Range to which normalize x.
    axis: Axis on which normalizing. Only relevant if `from_` or `to` items
      contains range value.

  Returns:
    Float tensor with same shape as x, but with normalized coordinates.
  """
  # Could add an `axis` argument.
  # Could add an `fill_values` argument to indicates the behavior if input
  # values are outside the input range. (`error`, `extrapolate` or `truncate`).

  if axis != -1:
    raise NotImplementedError(
        'Only last axis supported for now. Please send a feature request.'
    )

  # Note: This should be static arguments so we use numpy instead of jnp
  from_ = tuple(np.array(v) for v in from_)
  to = tuple(np.array(v) for v in to)

  # `a` can be scalar or array of shape=(x.shape[-1],), same for `b`
  a, b = _linear_interp_factors(*from_, *to)  # pytype: disable=wrong-arg-types
  return  a * x + b


def _linear_interp_factors(
    old_min: _MinMaxValue,
    old_max: _MinMaxValue,
    new_min: _MinMaxValue,
    new_max: _MinMaxValue,
) -> Tuple[Union[float, f32['d']], Union[float, f32['d']]]:
  """Resolve the `y = a * x + b` equation and returns the factors."""
  a = (new_min - new_max) / (old_min - old_max)
  b = (old_min * new_max - new_min * old_max) / (old_min - old_max)
  return a, b


def get_coords_grid(
    grid_shape: Tuple[int, ...],
) -> i32['*grid_shape len(grid_shape)']:
  """Returns an array containing the coordinates grid.

  For example: `get_coords_grid((28, 28))` returns `f32[28, 28, 2]`.

  Args:
    grid_shape: Shape of the output grid

  Returns:
    grid coordinates of shape (*grid_shape, len(grid_shape))
  """
  coord_ranges = [jnp.arange(coord) for coord in grid_shape]
  coord_grids = jnp.meshgrid(*coord_ranges, indexing='ij')
  return jnp.stack(coord_grids, axis=-1)
