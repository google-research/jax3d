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

"""Tests for jax3d.projects.nesf.nerfstatic.utils.camera_utils."""

import chex
from jax3d.projects.nesf.nerfstatic.utils import camera_utils
from jax3d.projects.nesf.nerfstatic.utils import types
import numpy as np
import pytest


def _batched_array(val, dtype=None):
  """Returns the array with leading `1` dimension."""
  return np.array(val, dtype=dtype)[None, ...]


# TODO(epot): Support the np.array case. Camera should work for both batched
# and non-batched arrays.
@pytest.mark.parametrize('np_array', [_batched_array])
def test_camera(np_array):
  camera = camera_utils.Camera.from_position_and_quaternion(
      positions=np_array([2., 0., 1.,]),
      quaternions=np_array([0.1, 0.2, 0.3, 0.4]),
      resolution=(2, 2),
      focal_px_length=280.,
  )
  rays = camera.pixel_centers2rays()
  expected_rays = types.Rays(
      scene_id=None,
      origin=np_array([
          [[2., 0., 1.],
           [2., 0., 1.]],
          [[2., 0., 1.],
           [2., 0., 1.]],
      ]),
      direction=np_array([
          [[-0.27698026, -0.24996764, -0.92779206],
           [-0.2750864, -0.24938536, -0.92851193]],
          [[-0.27663719, -0.25217938, -0.92729576],
           [-0.27474675, -0.25160123, -0.92801457]],
      ]),
  )
  chex.assert_tree_all_close(rays, expected_rays, ignore_nones=True)
