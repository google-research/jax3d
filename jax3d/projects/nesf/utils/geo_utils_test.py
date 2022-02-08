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

"""Tests for j3d.utils.geo_utils."""

import jax.numpy as jnp
import jax3d.projects.nesf as j3d


def test_interp_scalar():

  vals = jnp.array([
      [-1, -1],
      [-1, 0],
      [-1, 1],
      [0.5, 1],
      [1, 1],
  ])
  assert jnp.allclose(
      j3d.interp(vals, from_=(-1, 1), to=(0, 256)),
      jnp.array([
          [0, 0],
          [0, 128],
          [0, 256],
          [192, 256],
          [256, 256],
      ]),
  )
  assert jnp.allclose(
      j3d.interp(vals, from_=(-1, 1), to=(0, 1)),
      jnp.array([
          [0, 0],
          [0, 0.5],
          [0, 1],
          [0.75, 1],
          [1, 1],
      ]),
  )

  vals = jnp.array([
      [255, 255, 0],
      [255, 128, 0],
      [255, 0, 128],
  ])
  assert jnp.allclose(
      j3d.interp(vals, from_=(0, 255), to=(0, 1)),
      jnp.array([
          [1, 1, 0],
          [1, 128/255, 0],
          [1, 0, 128/255],
      ]),
  )
  assert jnp.allclose(
      j3d.interp(vals, from_=(0, 255), to=(-1, 1)),
      jnp.array([
          [1, 1, -1],
          [1, 0.00392163, -1],
          [1, -1, 0.00392163],
      ]),
  )


def test_interp_coords():

  coords = jnp.array([
      [-1, -1],
      [-1, 0],
      [-1, 1],
      [0.5, 1],
      [1, 1],
  ])
  assert jnp.allclose(j3d.interp(coords, (-1, 1), (0, (1024, 256))), jnp.array([
      [0, 0],
      [0, 128],
      [0, 256],
      [768, 256],
      [1024, 256],
  ]))

  coords = jnp.array([
      [[0, 0], [0, 1024]],
      [[256, 256], [0, 768]],
  ])
  assert jnp.allclose(j3d.interp(coords, (0, (256, 1024)), (0, 1)), jnp.array([
      [[0, 0], [0, 1]],
      [[1, 0.25], [0, 0.75]],
  ]))


def test_get_coords_grid():
  assert jnp.array_equal(
      j3d.get_coords_grid((2, 3)),
      jnp.array([
          [[0, 0], [0, 1], [0, 2]],
          [[1, 0], [1, 1], [1, 2]],
      ]),
  )
