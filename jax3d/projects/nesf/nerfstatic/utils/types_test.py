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

"""Tests for jax3d.projects.nesf.nerfstatic.utils.types."""

import jax.numpy as jnp
from jax3d.projects.nesf.nerfstatic.utils import types
import numpy as np


def test_bounding_box_simple():
  bbox = types.BoundingBox3d(
      min_corner=jnp.asarray([0, 0, 0]),
      max_corner=jnp.asarray([1, 1, 1]))

  rays = types.Rays(origin=jnp.asarray([10, 10, 10]),
                    direction=jnp.asarray([1, 1, 1]),
                    scene_id=None)
  assert bbox.intersect_rays(rays) == (-10, -9)


def test_bounding_box_zero_dir():
  bbox = types.BoundingBox3d(
      min_corner=jnp.asarray([0, 0, 0]),
      max_corner=jnp.asarray([1, 1, 1]))

  rays = types.Rays(origin=jnp.asarray([10, 0.5, 0.5]),
                    direction=jnp.asarray([1, 0, 0]),
                    scene_id=None)
  assert bbox.intersect_rays(rays) == (-10, -9)


def test_bounding_box_no_intersection():
  bbox = types.BoundingBox3d(
      min_corner=jnp.asarray([0, 0, 0]),
      max_corner=jnp.asarray([1, 1, 1]))

  rays = types.Rays(origin=jnp.asarray([10, 10, 10]),
                    direction=jnp.asarray([1, 0, 0]),
                    scene_id=None)
  i = bbox.intersect_rays(rays)
  assert i[1] < i[0]


def test_point_cloud():
  h, w = 6, 8
  normalize = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
  rays = types.Rays(scene_id=np.zeros((h, w, 1), dtype=np.int32),
                    origin=np.random.rand(h, w, 3),
                    direction=normalize(np.random.randn(h, w, 3)))
  views = types.Views(rays=rays,
                      depth=np.random.rand(h, w, 1),
                      semantics=np.random.randint(0, 5, size=(h, w, 1)))

  # Construct point cloud.
  point_cloud = views.point_cloud

  # Only valid points.
  assert np.all(point_cloud.points >= -1)
  assert np.all(point_cloud.points <= 1)

  # Size matches expected value.
  assert (point_cloud.size ==
          point_cloud.points.shape[0] ==
          point_cloud.semantics.shape[0])
