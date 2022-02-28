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

"""Camera util."""

from __future__ import annotations

import dataclasses

from etils import edc
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
from jax3d.visu3d import camera_spec
from jax3d.visu3d import plotly
from jax3d.visu3d import point as point_lib
from jax3d.visu3d import ray as ray_lib
from jax3d.visu3d import transformation
from jax3d.visu3d.lazy_imports import plotly_base
import numpy as np

# TODO(epot): Make this DataclassArray
# TODO(epot): Fix spec/cam_to_world xnp inconsistency


@edc.dataclass(kw_only=True)
@dataclasses.dataclass(frozen=True)
class Camera(plotly.Visualizable):
  """A camera located in space.

  Attributes:
    spec: Camera intrinsics parameters
    cam_to_world: Camera pose (`v3d.Transformation`)
  """
  spec: camera_spec.CameraSpec
  cam_to_world: transformation.Transform

  def __post_init__(self):
    # super().__post_init__()
    # TODO(epot): Remove once Camera is dataclass array.
    if self.spec.xnp is not self.cam_to_world.xnp:
      raise ValueError(
          f'Inconsistent camera numpy type: spec is {self.spec.xnp.__name__} '
          f'but cam_to_world is {self.cam_to_world.xnp.__name__}')

  @classmethod
  def from_look_at(
      cls,
      *,
      pos: FloatArray['*shape 3'],
      end: FloatArray['*shape 3'],
      spec: camera_spec.CameraSpec,
  ) -> Camera:
    """Factory which create a camera looking at `end`.

    This assume the camera is parallel to the floor. See `v3d.CameraSpec`
    for axis conventions.

    Args:
      pos: Camera position
      end: Camera direction (width horizontal to (x, y) floor)
      spec: Camera specifications.

    Returns:
      cam: Camera pointing to the ray.
    """
    ray = ray_lib.Ray.from_look_at(pos=pos, end=end)
    return cls.from_ray(spec=spec, ray=ray)

  @classmethod
  def from_ray(
      cls,
      *,
      ray: ray_lib.Ray,
      spec: camera_spec.CameraSpec,
  ) -> Camera:
    """Factory which create a camera from a ray.

    This assume the camera is parallel to the floor. See `v3d.CameraSpec`
    for axis conventions.

    Args:
      ray: Center ray of the camera.
      spec: Camera specifications.

    Returns:
      cam: Camera pointing to the ray.
    """
    cam_to_world = transformation.Transform.from_ray(ray)
    return cls(spec=spec, cam_to_world=cam_to_world)

  @property
  def resolution(self) -> tuple[int, int]:
    """`(h, w)` resolution in pixel."""
    return self.spec.resolution

  @property
  def h(self) -> int:
    """Height in pixel."""
    return self.spec.h

  @property
  def w(self) -> int:
    """Width in pixel."""
    return self.spec.w

  def rays(self, normalize: bool = True) -> ray_lib.Ray:
    """Creates the rays.

    Args:
      normalize: If `False`, returns camera rays in the `z=1` from the camera
        frame.

    Returns:
      rays: Pose
    """
    cam_dir = self.spec.px_to_cam()
    # Rotate the points from cam -> world
    world_dir = self.cam_to_world.apply_to_dir(cam_dir)
    # Position is (0, 0, 0) so no need to transform

    pos = self.spec.xnp.broadcast_to(self.cam_to_world.t, world_dir.shape)
    rays = ray_lib.Ray(pos=pos, dir=world_dir)
    if normalize:
      rays = rays.normalize()
    return rays

  def world_to_px(self, points: FloatArray['*d 3']) -> FloatArray['*d 2']:
    """Project the world coordinates back to pixel coordinates."""
    points_cam = self.cam_to_world.inv @ points
    return self.spec.cam_to_px(points_cam)

  def render(self, points: point_lib.Point) -> FloatArray['*shape h w 3']:
    """Project 3d points to the camera screen.

    Args:
      points: 3d points.

    Returns:
      img: The projected 3d points.
    """
    # TODO(epot): Support float colors and make this differentiable!
    if not isinstance(points, point_lib.Point):
      raise TypeError(
          f'Camera.render expect `v3d.Point` as input. Got: {points}.')

    rgb = points.rgb

    # Project 3d -> 2d coordinates
    px_coords = self.world_to_px(points.p)

    # Flatten pixels
    px_coords = px_coords.reshape((-1, 2))
    rgb = rgb.reshape((-1, 3))

    # Compute the valid coordinates
    h_coords = px_coords[..., 0]
    w_coords = px_coords[..., 1]
    # pyformat: disable
    valid_coords_mask = (
        (0 <= h_coords)
        & (h_coords < self.h)
        & (0 <= w_coords)
        & (w_coords < self.w)
    )
    # pyformat: enable
    rgb = rgb[valid_coords_mask]
    px_coords = px_coords[valid_coords_mask]

    px_coords = px_coords.astype(np.int32)

    # TODO(epot): Should we create a `xnp.array` ?
    # TODO(epot): The dtype should be cloned from point.rgb !
    img = np.zeros((*self.resolution, 3), dtype=np.uint8)
    img[px_coords[..., 0], px_coords[..., 1]] = rgb
    return img

  # Display functions

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    # TODO(epot): Add arrow to indicates the orientation

    corners_px = [  # Screen corners
        [0, 0],
        [self.spec.h, 0],
        [0, self.spec.w],
        [self.spec.h, self.spec.w],
    ]
    # Project the corners in the camera frame
    corners_cam = self.spec.px_to_cam(corners_px)
    # Convert cam to world coordinates
    corners_world = self.cam_to_world @ corners_cam

    return plotly.make_lines_traces(
        start=np.broadcast_to(self.cam_to_world.t, corners_world.shape),
        end=corners_world,
    )
