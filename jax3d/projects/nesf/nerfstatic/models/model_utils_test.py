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

"""Tests for model_utils."""

from jax3d.projects.nesf.nerfstatic.models import model_utils
import numpy as np


def test_generate_grid():
  num_scenes = 2
  grid_size = (3, 4, 5)
  x = model_utils.generate_grid(num_scenes, grid_size)
  assert x.shape == (2, 3 * 4 * 5, 3)

  # z-axis
  np.testing.assert_allclose(x[0, 0], np.array([-1, -1, -1]))
  np.testing.assert_allclose(x[0, 1], np.array([-1, -1, -0.5]))
  np.testing.assert_allclose(x[0, 2], np.array([-1, -1, 0.0]))

  # y-axis
  np.testing.assert_allclose(x[0, 0 * 5], np.array([-1, -1, -1]))
  np.testing.assert_allclose(x[0, 1 * 5], np.array([-1, -1 / 3, -1]))
  np.testing.assert_allclose(x[0, 2 * 5], np.array([-1, 1 / 3, -1]))

  # x-axis
  np.testing.assert_allclose(x[0, 0 * 4 * 5], np.array([-1, -1, -1]))
  np.testing.assert_allclose(x[0, 1 * 4 * 5], np.array([0, -1, -1]))
  np.testing.assert_allclose(x[0, 2 * 4 * 5], np.array([1, -1, -1]))
