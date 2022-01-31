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

from typing import Tuple

from etils.array_types import i32  # pylint: disable=g-multiple-import
import jax.numpy as jnp


def get_coords_grid(
    grid_shape: Tuple[int, ...],) -> i32['*grid_shape len(grid_shape)']:
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
