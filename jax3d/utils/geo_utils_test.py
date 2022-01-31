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

"""Tests for jax3d.utils.geo_utils."""

import jax.numpy as jnp
import jax3d.public_api as j3d


def test_get_coords_grid():
  assert jnp.array_equal(
      j3d.get_coords_grid((2, 3)),
      jnp.array([
          [[0, 0], [0, 1], [0, 2]],
          [[1, 0], [1, 1], [1, 2]],
      ]),
  )
