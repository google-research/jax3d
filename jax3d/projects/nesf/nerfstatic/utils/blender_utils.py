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

"""Blender utils."""

import einops
from jax3d.projects.nesf.utils.typing import f32
import numpy as np
from scipy.spatial import transform


def blender_quat2sfm_rot(blender_quaternions: f32['n 4']) -> f32['n 3 3']:
  """Convert blender quaternions to SFM rotation matrix.

  Blender and SFM uses different axis convensions.

  This function convert the rotation from pixel -> blender world coordinates
  into the inverse rotation world -> pixel, as expected by SFM.

  Args:
    blender_quaternions: Blender quaternions to convert

  Returns:
    scipy quaternions: Scipy quaternions
  """
  # Blender and scipy uses different axis convensions, so convert one
  # to the other.
  # Additionally, Kubric provide rotation matrix from pixels to world
  # coordinates while SFM expect rotation matrix from world to pixels
  #
  blender2scipy_quats = np.array([
      [1, 0, 0, 0],
      [0, 0, 0, 1],
      [0, 0, -1, 0],
      [0, 1, 0, 0],
  ])
  # Add/remove trailing 1 dimension for the batch matmul
  blender_quaternions = einops.rearrange(blender_quaternions, 'n d -> n d 1')
  scipy_quaternions = blender2scipy_quats @ blender_quaternions
  scipy_quaternions = einops.rearrange(scipy_quaternions, 'n d 1 -> n d')
  scipy_rot = transform.Rotation.from_quat(scipy_quaternions).as_matrix()
  return scipy_rot


def blender_quat2rot(quaternion: f32['... 4']) -> f32['... 3 3']:
  """Convert quaternion to rotation matrix.


  Equivalent to, but support batched case:

  ```python
  rot3x3 = mathutils.Quaternion(quaternion).to_matrix()
  ```

  Args:
    quaternion:

  Returns:
    rotation matrix
  """

  # Note: Blender first cast to double values for numerical precision while
  # we're using float32.
  q = np.sqrt(2) * quaternion

  q0 = q[..., 0]
  q1 = q[..., 1]
  q2 = q[..., 2]
  q3 = q[..., 3]

  qda = q0 * q1
  qdb = q0 * q2
  qdc = q0 * q3
  qaa = q1 * q1
  qab = q1 * q2
  qac = q1 * q3
  qbb = q2 * q2
  qbc = q2 * q3
  qcc = q3 * q3

  # Note: idx are inverted as blender and numpy convensions do not
  # match (x, y) -> (y, x)
  rotation = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float32)
  rotation[..., 0, 0] = 1.0 - qbb - qcc
  rotation[..., 1, 0] = qdc + qab
  rotation[..., 2, 0] = -qdb + qac

  rotation[..., 0, 1] = -qdc + qab
  rotation[..., 1, 1] = 1.0 - qaa - qcc
  rotation[..., 2, 1] = qda + qbc

  rotation[..., 0, 2] = qdb + qac
  rotation[..., 1, 2] = -qda + qbc
  rotation[..., 2, 2] = 1.0 - qaa - qbb
  return rotation
