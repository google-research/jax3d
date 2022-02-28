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

"""Camera spec utils."""

from __future__ import annotations

import abc
import dataclasses
from typing import Optional

import einops
from etils import edc
from etils import enp
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
from jax3d.visu3d import array_dataclass
from jax3d.visu3d import camera
from jax3d.visu3d import np_utils
from jax3d.visu3d import ray as ray_lib

del abc  # TODO(epot): Why pytype don't like abc ?


class CameraSpec(array_dataclass.DataclassArray):  # (abc.ABC):
  """Camera intrinsics specification.

  Define the interface of camera model. See `PinholeCamera` for an example of
  class implementation.

  Support batching to allow to stack multiple cameras with the same resolution
  in a single `CameraSpec`.

  ```python
  cams = v3d.stack([PinholeCamera(...) for _ in range(10)])

  isinstance(cams, CameraSpec)
  assert cams.shape == (10,)
  assert cams.px_centers().shape == (10, h, w, 2)
  ```

  Attributes:
    resolution: Camera resolution (in px).
    h: Camera height resolution (in px).
    w: Camera width resolution (in px).
  """
  resolution: tuple[int, int]
  h: int
  w: int

  # @abc.abstractmethod
  def cam_to_px(
      self,
      points3d: FloatArray['*shape *d 3'],
  ) -> FloatArray['*shape *d 2']:
    """Project camera 3d coordinates to px 2d coordinates.

    Input can have arbitrary batch shape, including no batch shape for a
    single point as input.

    Args:
      points3d: 3d points in the camera frame.

    Returns:
      point2d: 2d pixel coordinates. With same batch shape as input.
    """
    raise NotImplementedError

  # @abc.abstractmethod
  def px_to_cam(
      self,
      points2d: Optional[FloatArray['*shape 2']] = None,
  ) -> FloatArray['*shape *d 3']:
    """Unproject 2d pixel coordinates in image space to camera space.

    Note: Points returned by this function are not normalized. Points
    are returned at z=1.

    Input can have arbitrary batch shape, including no batch shape for a
    single point as input.

    Args:
      points2d: 2d pixel coordinates. If `None`, default to `px_centers()`,

    Returns:
      point3d: 3d points in the camera frame. With same batch shape as input.
    """
    raise NotImplementedError

  # @abc.abstractmethod
  def px_centers(self) -> FloatArray['*shape h w 2']:
    """Returns 2D coordinates of centers of all pixels in the camera image.

    The pixel centers of camera image are returned as a float32 tensor of shape
    `(image_height, image_width, 2)`.

    This camera model uses the convention that top left corner of the image is
    `(0, 0)` and bottom right corner is `(image_width, image_height)`. So the
    center of the top left corner pixel is `(0.5, 0.5)`.

    Returns:
      2D image coordinates of center of all pixels of the camer image in tensor
      of shape `(image_height, image_width, 2)`.
    """
    raise NotImplementedError

  def make_cam(self, ray: ray_lib.Ray) -> camera.Camera:
    """Camera factory from the given camera ray (origin and direction)."""
    return camera.Camera.from_ray(spec=self, ray=ray)


# TODO(epot): Support batch mode (should have some `@vectorize`)
@edc.dataclass(kw_only=True)
@dataclasses.dataclass(frozen=True)
class PinholeCamera(CameraSpec, array_dataclass.DataclassArray):
  """Simple camera model.

  Camera convensions:
  In camera/pixel coordinates, follow numpy convensions:

  * u == x == h (orientation: ↓)
  * v == y == w (orientation: →)

  In camera frame coordinates:

  * `(0, 0, 1)` is at the center of the image
  * z is pointing forward
  * x, y are oriented like pixel coordinates (x: ↓, y: →)

  Attributes:
    K: Camera intrinsics parameters.
    resolution: (h, w) resolution
  """
  resolution: tuple[int, int]
  K: FloatArray['*shape 3 3'] = array_dataclass.array_field(shape=(3, 3))  # pylint: disable=invalid-name

  @classmethod
  def from_focal(
      cls,
      *,
      resolution: tuple[int, int],
      focal_in_px: float,
      xnp: Optional[enp.NpModule] = None,
  ) -> PinholeCamera:
    """Camera factory.

    Args:
      resolution: `(h, w)` resolution in pixel
      focal_in_px: Focal length in pixel
      xnp: `numpy`, `jax.numpy` or `tf.experimental.numpy`. Numpy module to use.
        Default to `numpy`.

    Returns:
      A `PinholeCamera` instance with provided intrinsics.
    """
    if xnp is None:
      xnp = enp.lazy.get_xnp(focal_in_px, strict=False)

    # TODO(epot): Could provide more customizability
    # * Support `focal_in_mm`
    # * Support custom central point (cx, cy)
    # * Support different focal for h, w (fx, fy)

    # Central point in pixel (offset of the (0, 0) pixel)
    # Because our pixel coordinates are (0, 1), we set the central point
    # to the middle.
    h, w = resolution
    ch = h / 2
    cw = w / 2

    K = xnp.array([  # pylint: disable=invalid-name
        [focal_in_px, 0, ch],
        [0, focal_in_px, cw],
        [0, 0, 1],
    ])
    return cls(
        K=K,
        resolution=resolution,
    )

  @property
  def h(self):
    return self.resolution[0]

  @property
  def w(self):
    return self.resolution[1]

  def cam_to_px(
      self,
      points3d,
  ):
    points3d = self.xnp.asarray(points3d)
    # K @ [X,Y,Z] -> s * [u, v, 1]
    # (3, 3) @ (..., 3) -> (..., 3)
    points2d = self.xnp.einsum('ij,...j->...i', self.K, points3d)
    # Normalize: s * [u, v, 1] -> [u, v, 1]
    # And only keep [u, v]
    points2d = (points2d[..., :2] / points2d[..., 2:3])
    return points2d

  def px_to_cam(
      self,
      points2d=None,
  ):
    xnp = self.xnp

    # By default, project the pixel centers
    if points2d is None:
      points2d = self.px_centers()
    else:
      points2d = xnp.asarray(points2d)

    # [u, v] -> [u, v, 1]
    # Concatenate (..., 2) with (..., 1) -> (..., 3)
    points2d = np_utils.append_row(points2d, 1., axis=-1)

    # [X,Y,Z] / s = K-1 @ [u, v, 1]
    # (3, 3) @ (..., 3) -> (..., 3)
    k_inv = enp.compat.inv(self.K)
    points3d = xnp.einsum('ij,...j->...i', k_inv, points2d)

    # TODO(epot): Option to return normalized rays ?
    # Set z to -1
    # [X,Y,Z] -> [X, Y, Z=1]
    points3d = points3d / xnp.expand_dims(points3d[..., 2], axis=-1)
    return points3d

  def px_centers(self):
    if self.xnp is enp.lazy.tnp:  # TF Compatibility
      tf = enp.lazy.tf
      points2d = tf.meshgrid(
          tf.range(self.w),
          tf.range(self.h),
          indexing='xy',
      )
      points2d = tf.cast(tf.stack(points2d, axis=-1), dtype=tf.float32)
    else:
      points2d = self.xnp.mgrid[0:self.h, 0:self.w]
      points2d = einops.rearrange(points2d, 'c h w -> h w c')
    points2d = points2d + .5
    assert points2d.shape == (self.h, self.w, 2)
    return points2d
