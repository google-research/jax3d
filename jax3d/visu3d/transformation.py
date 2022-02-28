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

"""Transformation utils."""

from __future__ import annotations

import dataclasses
from typing import TypeVar

import einops
from etils import edc
from etils import enp
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
from jax3d.visu3d import array_dataclass
from jax3d.visu3d import np_utils
from jax3d.visu3d import plotly
from jax3d.visu3d import ray as ray_lib
from jax3d.visu3d.lazy_imports import plotly_base
import numpy as np

_T = TypeVar('_T')

# TODO(epot): Better design
# * Should we have a `ray.to_camera_transform()` ?
# * Add vectorize


@edc.dataclass(kw_only=True)
@dataclasses.dataclass(frozen=True)
class Transform(array_dataclass.DataclassArray, plotly.Visualizable):
  """Affine transformation (Position, rotation and scale of an object).

  Attributes:
    R: Rotation/scale/skewing of the transformation ( `[[x0, y0, z0], [x1, y1,
      z1], [x2, y2, z2]]`)
    t: Translation of the transformation (`tx, ty, tz`)
  """
  R: FloatArray['*shape 3 3'] = array_dataclass.array_field(shape=(3, 3))  # pylint: disable=invalid-name
  t: FloatArray['*shape 3'] = array_dataclass.array_field(shape=(3,))

  @classmethod
  def from_matrix(cls, matrix: FloatArray['*shape 3 3']) -> Transform:
    """Constructs from a 4x4 transform matrix."""
    return cls(R=matrix[..., :3, :3], t=matrix[..., :3, 3])

  @classmethod
  def from_look_at(
      cls,
      pos: FloatArray['*shape 3'],
      end: FloatArray['*shape 3'],
  ) -> Transform:
    """Factory to create a transformation which look at.

    Used to convert camera to world coordinates.

    This transformation assume the following convensions:

    * World coordinates: Floor is (x, y), z pointing upward
    * Camera coordinates: See `v3d.CameraSpec` docstring.

    The transformation assume the `width` dimension of the camera is parallel
    to the floor of the world.

    Args:
      pos: Camera origin
      end: Camera target

    Returns:
      The camera -> world transform.
    """
    xnp = enp.lazy.get_xnp(pos, strict=False)
    pos = xnp.asarray(pos)
    end = xnp.asarray(end)
    _assert_shape(pos, 'pos')
    _assert_shape(end, 'end')

    cam_forward = np_utils.normalize(end - pos)

    # In world coordinates, `z` is pointing up
    world_up = xnp.array([0, 0, 1])
    # The width of the cam is parallel to the ground (prependicular to z), so
    # use cross-product.
    cam_w = xnp.cross(cam_forward, world_up)

    # Similarly, the height is pointing downward.
    cam_h = xnp.cross(cam_forward, cam_w)

    R = xnp.stack([cam_h, cam_w, cam_forward], axis=-1)  # pylint: disable=invalid-name
    return cls(
        t=pos,
        R=R,
    )

  @classmethod
  def from_ray(cls, ray: ray_lib.Ray) -> Transform:
    """Factory which create a transformation. See `from_look_at` for details."""
    return cls.from_look_at(
        pos=ray.pos,
        end=ray.end,
    )

  @property
  def x_dir(self) -> FloatArray['*shape 3']:
    """`x` axis of the transformation (`[x0, x1, x2]`)."""
    return self.R[:, 0]

  @property
  def y_dir(self) -> FloatArray['*shape 3']:
    """`y` axis of the transformation (`[y0, y1, y2]`)."""
    return self.R[:, 1]

  @property
  def z_dir(self) -> FloatArray['*shape 3']:
    """`z` axis of the transformation (`[z0, z1, z2]`)."""
    return self.R[:, 2]

  @property
  def x_ray(self) -> ray_lib.Ray:
    """Array pointing to `z`."""
    return ray_lib.Ray(pos=self.t, dir=self.x_dir)

  @property
  def y_ray(self) -> ray_lib.Ray:
    """Array pointing to `z`."""
    return ray_lib.Ray(pos=self.t, dir=self.y_dir)

  @property
  def z_ray(self) -> ray_lib.Ray:
    """Array pointing to `z`."""
    return ray_lib.Ray(pos=self.t, dir=self.z_dir)

  @property
  def matrix4x4(self) -> FloatArray['*shape 4 4']:
    """Returns the 4x4 transformation matrix.

    [R|t]
    [0|1]
    """
    t = einops.rearrange(self.t, '... d -> ... d 1')
    matrix3x4 = self.xnp.concatenate([self.R, t], axis=-1)
    assert matrix3x4.shape == (3, 4)
    last_row = self.xnp.asarray([[0, 0, 0, 1]])
    return self.xnp.concatenate([matrix3x4, last_row], axis=-2)

  @property
  def inv(self) -> Transform:
    """Returns the inverse camera transform."""
    # Might be a more optimized way than stacking/unstacking matrix
    return type(self).from_matrix(enp.compat.inv(self.matrix4x4))

  def __matmul__(self, other: _T) -> _T:
    """Apply the transformation."""
    self.assert_same_xnp(other)
    if enp.lazy.is_array(other):
      return self.apply_to_pos(other)
    elif isinstance(other, ray_lib.Ray):
      return self.apply_to_ray(other)
    elif isinstance(other, Transform):
      return self.apply_to_transform(other)
    else:
      raise TypeError(f'Unexpected type: {type(other)}')

  def apply_to_ray(self, ray: ray_lib.Ray) -> ray_lib.Ray:
    """Apply the transformation on the ray."""
    self.assert_same_xnp(ray)
    # TODO(epot): Make this an util and auto-convert np -> xnp
    if ray.xnp is not self.xnp:
      raise ValueError(f'Transformation is {self.xnp} but ray is {ray.xnp}')
    return ray.replace(
        pos=self.apply_to_pos(ray.pos),
        dir=self.apply_to_dir(ray.dir),
    )

  def apply_to_pos(self, point: FloatArray['*d 3']) -> FloatArray['*d 3']:
    """Apply the transformation on the point cloud."""
    self.assert_same_xnp(point)
    point = self.xnp.asarray(point)
    if point.shape[-1] != 3:
      raise ValueError(f'point shape should be `(..., 3)`. Got {point.shape}')
    return self.apply_to_dir(point) + self.t

  def apply_to_dir(
      self,
      direction: FloatArray['*d 3'],
  ) -> FloatArray['*d 3']:
    """Apply the transformation on the direction."""
    self.assert_same_xnp(direction)
    # Direction are invariant to translation
    return self.xnp.einsum('ij,...j->...i', self.R, direction)

  def apply_to_transform(
      self,
      transform: Transform,
  ) -> Transform:
    self.assert_same_xnp(transform)
    return type(transform)(
        R=self.R @ transform.R,
        t=self.apply_to_pos(transform.t),
    )

  # Internal functions

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    base = ray_lib.Ray(
        # We can use `np` for the display
        pos=np.broadcast_to(self.t, self.shape + (
            3,
            3,
        )),
        # R is [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]] so we transpose
        # so dir is [[x0, x1, x2], [y0, y1, y2], [z0, z1, z2]]
        dir=np.asarray(self.R.T),
    )
    return base.make_traces()


def _assert_shape(array: FloatArray['*d 4'], name: str) -> None:
  """Test that array shape end by 3."""
  if array.shape[-1] != 3:
    raise ValueError(f'{name!r} shape should end be (3,). Got {array.shape}')
