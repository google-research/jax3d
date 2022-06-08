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

"""Camera utils."""

import dataclasses
from typing import Optional, Tuple

import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.utils import blender_utils
from jax3d.projects.nesf.nerfstatic.utils import coord_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import f32
import numpy as np
from scipy.spatial import transform


@dataclasses.dataclass(frozen=True)
class Camera:
  """Camera location and position.

  Attributes:
    px2world_transform: Transformation matrix from pixel to world coordinates
    resolution: (h, w) image resolution
    focal_px_length: Focal length in pixels
    use_unreal_axes: Whether or not to use Unreals axes specifications.
  """
  px2world_transform: f32['... 4, 4']
  resolution: Tuple[int, int]
  focal_px_length: float
  use_unreal_axes: bool = False

  @classmethod
  def from_position_and_quaternion(
      cls,
      positions: f32['... 3'],
      quaternions: f32['... 4'],
      resolution: Tuple[int, int],
      focal_px_length: float,
      use_unreal_axes: bool = False,
  ) -> 'Camera':
    """Factory which build a camera from blender position and quaternions.

    Args:
      positions: Camera position, applied after the rotation
      quaternions: Quaternions, as given by blender
      resolution: (h, w) image resolution
      focal_px_length: Focal length in pixels
      use_unreal_axes: Whether or not to use Unreals axes specifications.

    Returns:
      The camera object
    """
    if use_unreal_axes:
      rotations = transform.Rotation.from_quat(quaternions).as_matrix()
    else:
      # Rotation matrix that rotates from world to object coordinates.
      # Warning: Rotations should be given in blender convensions as
      # scipy.transform uses different convensions.
      rotations = blender_utils.blender_quat2rot(quaternions)
    px2world_transform = coord_utils.make_transform_matrix(
        positions=positions,
        rotations=rotations,
    )
    return cls(
        px2world_transform=px2world_transform,
        resolution=resolution,
        focal_px_length=focal_px_length,
        use_unreal_axes=use_unreal_axes,
    )

  @property
  def h(self) -> int:
    """Height."""
    return self.resolution[0]

  @property
  def w(self) -> int:
    """Width."""
    return self.resolution[1]

  def pixel_centers2rays(
      self,
      scene_boundaries: Optional[types.BoundingBox3d] = None,
  ) -> types.Rays:
    """Returns the rays position/orientation projected by this camera.

    Args:
      scene_boundaries: Scene boundaries (all objects should be contained
        within this bounding box). If provided, rays positions are normalized
        to be contained within the `[-1, 1]` frame.

    Returns:
      rays: Rays to be normalized
    """
    # Thoughs: h, w axis order used here seems inconsistent with other order
    # but should not matter when h and w are equals.

    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32) + 0.5,  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32) + 0.5,  # Y-Axis (rows)
        indexing='xy',
    )

    if self.use_unreal_axes:
      camera_dirs = np.stack([
          np.ones_like(x),
          (x - self.w * 0.5) / self.focal_px_length,
          -(y - self.h * 0.5) / self.focal_px_length,
      ], axis=-1)
    else:
      camera_dirs = np.stack([
          (x - self.w * 0.5) / self.focal_px_length,
          -(y - self.h * 0.5) / self.focal_px_length,
          -np.ones_like(x),
      ], axis=-1)
    directions = (
        camera_dirs[None, ..., None, :]
        * self.px2world_transform[:, None, None, :3, :3]
    ).sum(axis=-1)
    origins = np.broadcast_to(
        self.px2world_transform[:, None, None, :3, -1],
        directions.shape,
    )

    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    if self.use_unreal_axes:
      maybe_invert_x = np.array([-1, 1, 1])
    else:
      maybe_invert_x = np.array([1, 1, 1])
    rays = types.Rays(
        scene_id=None,  # Should this really be a property of Ray ?
        origin=origins * maybe_invert_x,
        direction=directions * maybe_invert_x,
    )

    # Rescale (x, y, z) from [min, max] -> [-1, 1]
    if scene_boundaries:
      rays = scale_rays(rays, scene_boundaries)

    return rays.replace(
        origin=rays.origin.astype(np.float32),
        direction=rays.direction.astype(np.float32),
    )


def scale_rays(
    rays: types.Rays,
    scene_boundaries: types.BoundingBox3d,
) -> types.Rays:
  """Rescale scene boundaries."""
  origins = rays.origin
  directions = rays.direction

  # Rescale (x, y, z) from [min, max] -> [-1, 1]
  origins = j3d.interp(
      origins,
      from_=(scene_boundaries.min_corner, scene_boundaries.max_corner),
      to=(-1., 1.),
      axis=-1,
  )
  # We also need to rescale the camera direction by bbox.size.
  # The direction can be though of a ray from a point in space (the camera
  # origin) to another point in space (say the red light on the lego
  # bulldozer). When we scale the scene in a certain way, this direction
  # also needs to be scaled in the same way.
  directions = directions * 2 / scene_boundaries.size
  # (re)-normalize the rays
  directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
  return rays.replace(
      origin=origins,
      direction=directions,
  )
