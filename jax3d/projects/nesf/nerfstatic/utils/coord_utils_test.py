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

"""Tests for jax3d.projects.nesf.nerfstatic.utils.coord_utils."""

from jax3d.projects.nesf.nerfstatic.utils import coord_utils
import numpy as np


def test_coord_utils():
  positions = np.array([123, 456, 789])
  rotations = np.array([
      [3.85584300e-01, -6.05560465e-01, 6.96147449e-01],
      [9.22672611e-01, 2.53063438e-01, -2.90919488e-01],
      [-1.26111691e-07, 7.54490172e-01, 6.56311344e-01],
  ])
  rot = coord_utils.make_transform_matrix(
      positions=positions,
      rotations=rotations,
  )
  expected_rot = np.array([
      [3.85584300e-01, -6.05560465e-01, 6.96147449e-01, 123],
      [9.22672611e-01, 2.53063438e-01, -2.90919488e-01, 456],
      [-1.26111691e-07, 7.54490172e-01, 6.56311344e-01, 789],
      [0, 0, 0, 1],
  ])
  assert np.allclose(rot, expected_rot)

  # Adding a batch size should not modify the result
  rot = coord_utils.make_transform_matrix(
      positions=positions[None, ...],
      rotations=rotations[None, ...],
  )
  assert np.allclose(rot, expected_rot[None, ...])
