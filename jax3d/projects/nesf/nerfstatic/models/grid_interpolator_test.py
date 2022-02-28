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

"""Tests for jax3d.projects.nesf.nerfstatic.models.grid_interpolator."""

from typing import List

from jax import random
import jax.numpy as jnp
from jax3d.projects.nesf.nerfstatic.models.grid_interpolator import compute_corner_indices
from jax3d.projects.nesf.nerfstatic.models.grid_interpolator import compute_corner_weights
from jax3d.projects.nesf.nerfstatic.models.grid_interpolator import GridInterpolator
from jax3d.projects.nesf.nerfstatic.models.grid_interpolator import TrilinearInterpolation
from jax3d.projects.nesf.utils.typing import f32
import numpy as np


def test_compute_corner_indices_1d_simple():
  p = jnp.asarray([0.5], dtype=jnp.float32)
  indices = compute_corner_indices(grid_shape=[2], points=p)
  np.testing.assert_array_equal(indices, [[0], [1]])


def test_compute_corner_indices_1d_full():
  p = jnp.asarray([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=jnp.float32)
  indices = compute_corner_indices(grid_shape=[6], points=p)
  np.testing.assert_array_equal(
      indices,
      [[[0], [1]], [[1], [2]], [[2], [3]], [[3], [4]], [[4], [5]]])


def _compute_base_index(grid_size: List[int], pos: f32['dim']):
  """This function computes the index of corner upper/top/left next to pos."""
  pos = pos * (np.asarray(grid_size) - 1)
  return pos.astype(np.int32)


def test_compute_corner_indices_3d_simple():
  p = jnp.asarray([0.15, 0.25, 0.75], dtype=jnp.float32)
  grid_shape = [11, 11, 11]
  indices = compute_corner_indices(grid_shape=grid_shape, points=p)
  base_index = _compute_base_index(grid_shape, p)
  expected_indices = []
  for z in range(2):
    for y in range(2):
      for x in range(2):
        expected_indices.append(base_index + np.asarray([x, y, z]))
  expected_indices = np.stack(expected_indices, axis=0)
  np.testing.assert_array_equal(indices, expected_indices)


def test_compute_corner_weight_1d_simple():
  p = jnp.asarray([[0.1],
                   [0.5],
                   [0.7]], dtype=jnp.float32)
  weights = compute_corner_weights(grid_shape=[2], points=p)
  np.testing.assert_allclose(weights, [[0.9, 0.1],
                                       [0.5, 0.5],
                                       [0.3, 0.7]])


def test_compute_corner_weight_1d_full():
  p = jnp.asarray([[0.11], [0.33], [0.55], [0.77], [0.99]], dtype=jnp.float32)
  weights = compute_corner_weights(grid_shape=[11], points=p)
  np.testing.assert_allclose(
      weights,
      [[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]], atol=1e-6)


def test_compute_corner_weight_3d_simple():
  p = jnp.asarray([0.11, 0.22, 0.77], dtype=jnp.float32)
  weights = compute_corner_weights(grid_shape=[11, 11, 11], points=p)
  base_weights = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]
  expected_weights = []
  for z in range(2):
    wz = base_weights[2][z]
    for y in range(2):
      wy = wz * base_weights[1][y]
      for x in range(2):
        wx = wy * base_weights[0][x]
        expected_weights.append(wx)
  np.testing.assert_allclose(weights, expected_weights, atol=1e-6)
  np.testing.assert_allclose(jnp.sum(weights), 1.0)


def test_grid_interpolator_1d_simple():
  scenes = jnp.asarray([[0], [0]])
  p = jnp.asarray([[0.4]])
  corners = jnp.asarray([[-1], [1]], dtype=jnp.float32)
  embeddings = np.random.normal(size=(1, 2, 17))
  g = GridInterpolator(interpolation=TrilinearInterpolation())
  init_variables = g.init(random.PRNGKey(0), voxel_embeddings=embeddings,
                          grid_indexes=scenes, points=corners)
  corner_latents = g.apply(
      init_variables, voxel_embeddings=embeddings,
      grid_indexes=scenes, points=corners)
  p_latent = g.apply(init_variables, voxel_embeddings=embeddings,
                     grid_indexes=scenes[:1], points=p)

  px = p[0][0] / 2 + 0.5
  expected = corner_latents[0] * (1 - px) + corner_latents[1] * px
  np.testing.assert_allclose(p_latent[0], expected, rtol=1e-05)


def test_grid_interpolator_2d_simple():
  scenes = jnp.asarray([[[0], [0]], [[0], [0]]])
  p = jnp.asarray([[-0.6, -0.4]])
  corners = jnp.asarray([[[-1, -1], [-1, 1]], [[1, -1], [1, 1]]],
                        dtype=jnp.float32)
  embeddings = np.random.normal(size=(1, 2, 2, 27))
  g = GridInterpolator(interpolation=TrilinearInterpolation())
  init_variables = g.init(random.PRNGKey(0), voxel_embeddings=embeddings,
                          grid_indexes=scenes, points=corners)
  corner_latents = g.apply(init_variables, voxel_embeddings=embeddings,
                           grid_indexes=scenes, points=corners)
  p_latent = g.apply(init_variables, voxel_embeddings=embeddings,
                     grid_indexes=scenes[0, :1], points=p)

  px = p[0][0] / 2 + 0.5
  py = p[0][1] / 2 + 0.5
  exp = corner_latents[0] * (1 - px) + corner_latents[1] * px
  exp = exp[0] * (1 - py) + exp[1] * py
  np.testing.assert_allclose(p_latent[0], exp, rtol=1e-03)


def test_grid_interpolator_multiple_grids():
  scenes = jnp.asarray([[0], [7]])
  p = jnp.asarray([[0.4], [-0.4]])
  embeddings = np.random.normal(size=(8, 2, 17))
  g = GridInterpolator(interpolation=TrilinearInterpolation())
  init_variables = g.init(random.PRNGKey(0), voxel_embeddings=embeddings,
                          grid_indexes=scenes, points=p)
  g.apply(init_variables, voxel_embeddings=embeddings,
          grid_indexes=scenes, points=p)
