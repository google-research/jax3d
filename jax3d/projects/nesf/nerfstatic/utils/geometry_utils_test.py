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

"""Tests for geometry_utils."""

import jax
import jax.numpy as jnp
from jax3d.projects.nesf.nerfstatic.utils import geometry_utils as geom
from jax3d.projects.nesf.nerfstatic.utils import types
import numpy as np


def _make_rays(origin, direction):
  n, _ = origin.shape
  return types.Rays(scene_id=jnp.zeros((n, 1), dtype=jnp.int32),
                    origin=origin,
                    direction=direction)


def test_scale():
  transform = geom.Scale(scale=jnp.array([1, 2, 3]))
  rays = _make_rays(origin=np.array([[1, 1, 0]]),
                    direction=np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0]]))
  rays2 = transform.forward(rays)
  rays3 = transform.backward(rays2)

  jax.tree_map(np.testing.assert_allclose, rays, rays3)
  np.testing.assert_allclose(rays2.origin[0, 1], 2)  # pytype: disable=attribute-error


def test_rotate():
  transform = geom.Rotate(axis=jnp.array([0, 0, 1]), radians=np.pi/2)
  rays = _make_rays(origin=np.array([[1, 1, 0]]),
                    direction=np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0]]))
  rays2 = transform.forward(rays)
  rays3 = transform.backward(rays2)

  jax.tree_map(np.testing.assert_allclose, rays, rays3)
  np.testing.assert_allclose(rays2.origin[0], np.array([-1, 1, 0]))  # pytype: disable=attribute-error
  np.testing.assert_allclose(rays2.direction[0],
                             np.array([1/np.sqrt(2), 1/np.sqrt(2), 0]))


def test_translate():
  transform = geom.Translate(offset=jnp.array([1, 2, 3]))
  rays = _make_rays(origin=np.array([[1, 1, 0]]),
                    direction=np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0]]))
  rays2 = transform.forward(rays)
  rays3 = transform.backward(rays2)

  jax.tree_map(np.testing.assert_allclose, rays, rays3)
  np.testing.assert_allclose(rays2.origin[0], np.array([2, 3, 3]))  # pytype: disable=attribute-error
  np.testing.assert_allclose(rays2.direction[0],
                             np.array([1/np.sqrt(2), -1/np.sqrt(2), 0]))


def test_compose():
  transform = geom.Compose(transforms=[
      geom.Scale(scale=np.array([1, 2, 1])),
      geom.Rotate(axis=np.array([0, 0, 1]), radians=np.pi/2),
      geom.Translate(offset=jnp.array([1, -1, 3])),
  ])
  rays = _make_rays(origin=np.array([[1, 0.5, -3]]),
                    direction=np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0]]))
  rays2 = transform.forward(rays)
  rays3 = transform.backward(rays2)

  jax.tree_map(np.testing.assert_allclose, rays, rays3)
  np.testing.assert_allclose(rays2.origin[0], np.zeros(3))  # pytype: disable=attribute-error
  np.testing.assert_allclose(rays2.direction[0],
                             np.array([2/np.sqrt(2), 1/np.sqrt(2), 0]))


def test_inverse():
  transform = geom.Inverse(transform=geom.Compose(transforms=[
      geom.Scale(scale=np.array([1, 2, 1])),
      geom.Rotate(axis=np.array([0, 0, 1]), radians=np.pi/2),
      geom.Translate(offset=jnp.array([1, -1, 3])),
  ]))
  rays = _make_rays(origin=np.array([[0, 0, 0]]),
                    direction=np.array([[1.4142135, 0.70710677, 0.]]))
  rays2 = transform.forward(rays)
  rays3 = transform.backward(rays2)

  jax.tree_map(np.testing.assert_allclose, rays, rays3)
  np.testing.assert_allclose(rays2.origin[0], np.array([1, 0.5, -3]))  # pytype: disable=attribute-error
  np.testing.assert_allclose(rays2.direction[0],
                             np.array([1/np.sqrt(2), -1/np.sqrt(2), 0]))


def test_identity():
  transform = geom.Identity()
  rays = _make_rays(origin=np.array([[0, 0, 0]]),
                    direction=np.array([[1.4142135, 0.70710677, 0.]]))
  rays2 = transform.forward(rays)
  rays3 = transform.backward(rays2)

  jax.tree_map(np.testing.assert_allclose, rays, rays2)
  jax.tree_map(np.testing.assert_allclose, rays, rays3)


def test_sample_points():
  n = 20
  k = 7
  sample_points = types.SamplePoints(
      scene_id=np.random.randint(0, 5, size=(n, 1)),
      position=np.random.randn(n, k, 3),
      direction=np.random.randn(n, 3))
  rays = geom._sample_points_to_rays(sample_points)
  sample_points2 = geom._rays_to_sample_points(rays, sample_points.batch_shape)
  jax.tree_map(np.testing.assert_allclose, sample_points, sample_points2)
