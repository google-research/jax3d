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

# pylint: disable=line-too-long
"""Invertible transforms for 3D geometry.

This library includes a base class, Transform, that represents an invertible
transform from one coordinate system to another. Each implementation of
Transform exposes two methods: forward() and backward(). We are guaranteed
that transform.backward(transfrom.forward(rays)) == rays.

# Usage

```
transform = geom.Inverse(
    transform=geom.Compose(transforms=[
        geom.Scale(scale=np.array([1, 2, 1])),
        geom.Rotate(axis=np.array([0, 0, 1]), radians=np.pi/2),
        geom.Translate(offset=jnp.array([1, -1, 3])),
    ])
)

rays = ...
new_rays = transform.forward(rays)
processed_rays = ... # Do calculations like intersection with an axis-aligned box here.
original_rays = transform.backward(processed_rays)
```

"""
# pylint: enable=line-too-long

from typing import Sequence, Tuple, Union

import chex
import jax.numpy as jnp

from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import f32


@chex.dataclass
class Transform:
  """An invertible transform."""

  def forward(self, rays: Union[types.Rays, types.SamplePoints],
              ) -> Union[types.Rays, types.SamplePoints]:
    """Convert from old to new coordinate system."""
    if isinstance(rays, types.SamplePoints):
      sample_points = rays
      rays = _sample_points_to_rays(sample_points)
      rays = self._forward(rays)
      return _rays_to_sample_points(rays, sample_points.batch_shape)
    elif isinstance(rays, types.Rays):
      return self._forward(rays)
    else:
      raise NotImplementedError(type(rays))

  def backward(self, rays: Union[types.Rays, types.SamplePoints],
               ) -> Union[types.Rays, types.SamplePoints]:
    """Convert from new to old coordinate system."""
    if isinstance(rays, types.SamplePoints):
      sample_points = rays
      rays = _sample_points_to_rays(sample_points)
      rays = self._backward(rays)
      return _rays_to_sample_points(rays, sample_points.batch_shape)
    elif isinstance(rays, types.Rays):
      return self._backward(rays)
    else:
      raise NotImplementedError(type(rays))

  def _forward(self, rays: types.Rays) -> types.Rays:
    raise NotImplementedError()

  def _backward(self, rays: types.Rays) -> types.Rays:
    raise NotImplementedError()


@chex.dataclass
class Scale(Transform):
  """Apply a coordinate-wise scaling transform.

  Attributes:
    scale: Amount to scale each coordinate.
  """
  scale: f32['3']

  def _forward(self, rays: types.Rays) -> types.Rays:
    return rays.replace(origin=(rays.origin * self.scale),
                        direction=(rays.direction * self.scale))

  def _backward(self, rays: types.Rays) -> types.Rays:
    return rays.replace(origin=(rays.origin / self.scale),
                        direction=(rays.direction / self.scale))


@chex.dataclass
class Rotate(Transform):
  """Apply a counter-clockwise rotation about an axis.

  Attributes:
    axis: The axis to rotate around.
    radians: The amount to rotate counter-clockwis.
  """
  axis: f32['3']
  radians: float

  def _forward(self, rays: types.Rays) -> types.Rays:
    rotmat = self._rotation_matrix
    return rays.replace(origin=jnp.dot(rays.origin, rotmat.T),
                        direction=jnp.dot(rays.direction, rotmat.T))

  def _backward(self, rays: types.Rays) -> types.Rays:
    rotmat = self._rotation_matrix
    return rays.replace(origin=jnp.dot(rays.origin, rotmat),
                        direction=jnp.dot(rays.direction, rotmat))

  @property
  def _rotation_matrix(self):
    """Construct a rotation matrix."""
    axis = self.axis / jnp.linalg.norm(self.axis)
    sin = jnp.sin(self.radians)
    cos = jnp.cos(self.radians)
    kx, ky, kz = axis
    K = jnp.array([[0, -kz, ky],  # pylint: disable=invalid-name
                   [kz, 0, -kx],
                   [-ky, kx, 0]])
    I = jnp.eye(3)  # pylint: disable=invalid-name
    return I + sin * K + (1-cos) * jnp.dot(K, K)


@chex.dataclass
class Translate(Transform):
  """Apply a translation.

  Attributes:
    offset: amount by which to translate.
  """
  offset: f32['3']

  def _forward(self, rays: types.Rays) -> types.Rays:
    return rays.replace(origin=(rays.origin + self.offset),
                        direction=rays.direction)

  def _backward(self, rays: types.Rays) -> types.Rays:
    return rays.replace(origin=(rays.origin - self.offset),
                        direction=rays.direction)


@chex.dataclass
class Compose(Transform):
  """Compose several transforms in a row.

  Attributes:
    transforms: Transforms to apply. Earlier transforms are applied before
      latter ones.
  """
  transforms: Sequence[Transform]

  def _forward(self, rays: types.Rays) -> types.Rays:
    for transform in self.transforms:
      rays = transform._forward(rays)  # pylint: disable=protected-access
    return rays

  def _backward(self, rays: types.Rays) -> types.Rays:
    for transform in reversed(self.transforms):
      rays = transform._backward(rays)  # pylint: disable=protected-access
    return rays


@chex.dataclass
class Inverse(Transform):
  """Invert a transform.

  Attributes:
    transform: Transform to invert.
  """
  transform: Transform

  def _forward(self, rays: types.Rays) -> types.Rays:
    return self.transform._backward(rays)  # pylint: disable=protected-access

  def _backward(self, rays: types.Rays) -> types.Rays:
    return self.transform._forward(rays)  # pylint: disable=protected-access


@chex.dataclass
class Identity(Transform):
  """The identity transform."""

  def _forward(self, rays: types.Rays) -> types.Rays:
    return rays

  def _backward(self, rays: types.Rays) -> types.Rays:
    return rays


def _rays_to_sample_points(rays: types.Rays,
                           batch_shape: Tuple[int, ...]) -> types.SamplePoints:
  """Converts Rays to SamplePoints.

  Note: This operation is destructive. It assumes that each point along the
  same ray has the same scene_id and direction.

  Args:
    rays: Input rays. Each ray corresponds to one entry in 'position'.
    batch_shape: Desired batch_shape for output.

  Returns:
    SamplePoints instance. Each entry in 'position' corresponds to an entry
      in 'rays.origin'.
  """
  assert len(batch_shape) >= 1
  assert batch_shape[-1] > 0

  # Verify shapes.
  b = rays.batch_shape
  c = rays.origin.shape[-1]
  chex.assert_shape(rays.scene_id, b + (1,))
  chex.assert_shape(rays.origin, b + (c,))
  chex.assert_shape(rays.direction, b + (c,))

  # Construct scene_id if necessary.
  scene_id = rays.scene_id
  if scene_id is None:
    scene_id = jnp.zeros(rays.batch_shape + (1,), dtype=jnp.int32)

  # Construct output. The first entry along each ray is chosen for scene_id
  # and direction.
  scene_id = jnp.reshape(scene_id, batch_shape + (1,))[..., 0, :]
  position = jnp.reshape(rays.origin, batch_shape + (c,))
  direction = jnp.reshape(rays.direction, batch_shape + (c,))[..., 0, :]

  return types.SamplePoints(scene_id=scene_id,
                            position=position,
                            direction=direction)


def _sample_points_to_rays(sample_points: types.SamplePoints) -> types.Rays:
  """Converts SamplePoints to Rays."""
  # Verify shapes.
  b = sample_points.batch_shape
  assert len(b) >= 1, b
  c = sample_points.position.shape[-1]
  chex.assert_shape(sample_points.scene_id, b[:-1] + (1,))
  chex.assert_shape(sample_points.position, b + (c,))
  chex.assert_shape(sample_points.direction, b[:-1] + (c,))

  # Copy scene_id, direction to every point in positions.
  scene_id = jnp.broadcast_to(sample_points.scene_id[..., None, :],
                              sample_points.position.shape[:-1] + (1,))
  direction = jnp.broadcast_to(sample_points.direction[..., None, :],
                               sample_points.position.shape)

  # Merge ray and samples_along_ray dimensions.
  new_batch_shape = sample_points.batch_shape[0:-2] + (-1,)
  scene_id = jnp.reshape(scene_id, (*new_batch_shape, 1))
  origin = jnp.reshape(sample_points.position, (*new_batch_shape, c))
  direction = jnp.reshape(direction, (*new_batch_shape, c))

  return types.Rays(scene_id=scene_id, origin=origin, direction=direction)
