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

"""Utils for geometric transformations."""

from jax3d.projects.nesf.utils.typing import f32
import numpy as np


def make_transform_matrix(
    positions: f32['... 3'],
    rotations: f32['... 3 3'],
) -> f32['... 4 4']:
  """Create the 4x4 transformation matrix.

  Note: This function uses numpy.

  Args:
    positions: Translation applied after the rotation.
      Last column of the transformation matrix
    rotations: Rotation. Top-left 3x3 matrix of the transformation matrix.

  Returns:
    transformation_matrix:
  """
  # Create the 4x4 transformation matrix
  rot_pos = np.broadcast_to(np.eye(4), (*positions.shape[:-1], 4, 4)).copy()
  rot_pos[..., :3, :3] = rotations
  rot_pos[..., :3, 3] = positions
  return rot_pos
