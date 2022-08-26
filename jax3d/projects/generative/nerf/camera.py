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

"""Functions for handling cameras differentiably."""

from typing import Dict, Optional, Tuple, Union

import einops
import jax
import jax.numpy as jnp
import numpy as np

CameraType = Dict[str, Union[jnp.ndarray, float]]


def _compute_residual_and_jacobian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    xd: jnp.ndarray,
    yd: jnp.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray]:
  """Auxiliary function of radial_and_tangential_undistort().

  Args:
    x: Undistorted x coordinate from previous step.
    y: Undistorted y coordinate from previous step.
    xd: Distorted x coordinate.
    yd: Distorted y coordinate.
    k1: Radial distortion parameter 1.
    k2: Radial distortion parameter 2.
    k3: Radial distortion parameter 3.
    p1: Tangential distortion parameter 1.
    p2: Tangential distortion parameter 2.

  Returns:
    A tuple containing:
      The x residual fx(x, y).
      The y residual fy(x, y).
      The partial derivative dfx(x, y) / dx.
      The partial derivative dfx(x, y) / dy.
      The partial derivative dfy(x, y) / dx.
      The partial derivative dfy(x, y) / dy.
  """
  # let r(x, y) = x^2 + y^2;
  #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + k3 * r))

  # The perfect projection is:
  # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
  # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
  #
  # Let's define
  #
  # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
  # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
  #
  # We are looking for a solution that satisfies
  # fx(x, y) = fy(x, y) = 0;
  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: jnp.ndarray,
    yd: jnp.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes undistorted (x, y) from (xd, yd).

  Args:
    xd: Distorted x coordinate.
    yd: Distorted y coordinate.
    k1: Radial distortion parameter 1.
    k2: Radial distortion parameter 2.
    k3: Radial distortion parameter 3.
    p1: Tangential distortion parameter 1.
    p2: Tangential distortion parameter 2.
    eps: Numeric epsilon value.
    max_iterations: Number of solver iterations to use.

  Returns:
    A tuple containing:
      Undistorted x coordinates.
      Undistorted y coordinates.
  """
  # Initialize from the distorted point.
  x = xd.copy()
  y = yd.copy()

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = jnp.where(
        jnp.abs(denominator) > eps, x_numerator / denominator,
        jnp.zeros_like(denominator))
    step_y = jnp.where(
        jnp.abs(denominator) > eps, y_numerator / denominator,
        jnp.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

  return x, y


def make_camera(
    orientation: jnp.ndarray,
    position: jnp.ndarray,
    focal_length: Union[jnp.ndarray, float],
    principal_point: jnp.ndarray,
    image_size: jnp.ndarray,
    skew: Union[jnp.ndarray, float] = None,
    pixel_aspect_ratio: Union[jnp.ndarray, float] = None,
    radial_distortion: Optional[jnp.ndarray] = None,
    tangential_distortion: Optional[jnp.ndarray] = None) -> CameraType:
  """Create a dictionary containing standard values for representing a camera.

  All parameters may have arbitrary leading axes for the purpose of batching, so
  long as these shapes are mutually compatible. Functions in this module expect
  non-batched cameras, so `vmap` should be used by the caller to specify any
  batching behaviour.

  Args:
    orientation: 3x3 rotation matrix representing orientation of the camera.
    position: 3D vector representing the position of the camera viewpoint.
    focal_length: Scalar value equal to the focal length in pixels.
    principal_point: 2D vector representing the principal point in pixels.
    image_size: 2D vector representing image dimensions in pixels.
    skew: Scalar representing the camera skew.
    pixel_aspect_ratio: Scalar representing the pixel aspect ratio.
    radial_distortion: Optional 3D vector representing radial distortion.
    tangential_distortion: Optional 2D vector representing tangential
      distortion.

  Returns:
    A dictionary containing the camera parameters.
  """
  if skew is None:
    skew = jnp.zeros_like(focal_length)
  if pixel_aspect_ratio is None:
    pixel_aspect_ratio = jnp.ones_like(focal_length)
  camera = {
      "orientation": orientation,
      "position": position,
      "focal_length": focal_length,
      "principal_point": principal_point,
      "image_size": image_size,
      "skew": skew,
      "pixel_aspect_ratio": pixel_aspect_ratio,
  }
  if radial_distortion is not None:
    camera["radial_distortion"] = radial_distortion
  if tangential_distortion is not None:
    camera["tangential_distortion"] = tangential_distortion
  return camera


def look_at(eye_position: jnp.ndarray,
            target: jnp.ndarray,
            focal_length: Union[jnp.ndarray, float],
            image_size: jnp.ndarray,
            global_up: Optional[jnp.ndarray],
            principal_point: Optional[jnp.ndarray] = None) -> CameraType:
  """Generate a camera that looks towards a particular position.

  Args:
    eye_position: [3] world space position vector of the camera.
    target: [3] world space position vector the camera will be looking at.
    focal_length: [1] the focal length of the camera in pixels.
    image_size: [2] vector representing image dimensions in pixels.
    global_up: [3] unit vector indicating the "up" direction that the
      camera should be oriented relative to.
    principal_point: [2] optional principal point. Defaults to image center.

  Returns:
    A dictionary containing the camera parameters.
  """
  # Axis convention for the camera local space:
  # +X = right
  # -Y = up
  # +Z = forward
  camera_forward = target - eye_position
  camera_forward /= jnp.linalg.norm(camera_forward)
  camera_right = jnp.cross(camera_forward, global_up)
  camera_right /= jnp.linalg.norm(camera_right)
  camera_up = jnp.cross(camera_right, camera_forward)
  orientation = jnp.stack([-camera_right, camera_up, camera_forward], axis=0)
  if principal_point is None:
    principal_point = jnp.array(image_size) / 2

  return make_camera(orientation, eye_position, jnp.array(focal_length),
                     principal_point, image_size)


def pixel_to_local_rays(camera, pixels: jnp.ndarray):
  """Returns the local ray directions for the provided pixels."""
  scale_factor_x = camera["focal_length"]
  scale_factor_y = camera["focal_length"] * camera["pixel_aspect_ratio"]
  y = ((pixels[..., 1] - camera["principal_point"][1]) / scale_factor_y)
  x = ((pixels[..., 0] - camera["principal_point"][0] - y * camera["skew"]) /
       scale_factor_x)

  if "radial_distortion" in camera and "tangential_distortion" in camera:
    x, y = _radial_and_tangential_undistort(
        x,
        y,
        k1=camera["radial_distortion"][0],
        k2=camera["radial_distortion"][1],
        k3=camera["radial_distortion"][2],
        p1=camera["tangential_distortion"][0],
        p2=camera["tangential_distortion"][1])

  dirs = jnp.stack([x, y, jnp.ones_like(x)], axis=-1)
  return dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)


def pixels_to_points(camera, pixels: jnp.ndarray, depth: jnp.ndarray):
  """Returns world space points for pixel coordinates and depth values.

  Args:
    camera: camera object produced by `make_camera`.
    pixels: [A1, ..., An, 2] tensor or np.array containing 2d pixel positions.
    depth: [A1, ..., An] tensor or np.array containing pixel depth values.

  Returns:
    [A1, ..., An, 3] tensor containing 3d world space points.
  """
  rays_through_pixels = pixels_to_rays(camera, pixels)
  cosa = jnp.matmul(rays_through_pixels, camera["orientation"][2, :])
  points = (
      rays_through_pixels * depth[..., jnp.newaxis] / cosa[..., jnp.newaxis] +
      camera["position"])
  return points


def points_to_local_points(camera, points: jnp.ndarray):
  """Transforms world space points into points in the local camera space.

  Args:
    camera: camera object produced by `make_camera`.
    points: [A1, ..., An, 3] tensor containing 3d world space points.

  Returns:
    [A1, ..., An, 3] tensor containing 3d camera space points.
  """
  translated_points = points - camera["position"]
  local_points = (jnp.matmul(camera["orientation"], translated_points.T)).T
  return local_points


def pixels_to_rays(camera,
                   pixels: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Returns the rays for the provided pixels.

  Args:
    camera: camera object produced by `make_camera`.
    pixels: [A1, ..., An, 2] tensor or np.array containing 2d pixel positions.

  Returns:
    A tuple containing:
      An array of the ray origins in world coordinates.
      An array of the normalized ray directions in world coordinates.
  """

  batch_shape = pixels.shape[:-1]
  pixels = jnp.reshape(pixels, (-1, 2))

  local_rays_dir = pixel_to_local_rays(camera, pixels)
  rays_dir = jnp.matmul(
      camera["orientation"].T,
      local_rays_dir[..., jnp.newaxis],
      precision=jax.lax.Precision.HIGHEST)
  rays_dir = jnp.squeeze(rays_dir, axis=-1)

  # Normalize rays.
  rays_dir /= jnp.linalg.norm(rays_dir, axis=-1, keepdims=True)
  rays_dir = rays_dir.reshape((*batch_shape, 3))

  origins = camera["position"]
  for _ in batch_shape:
    origins = origins[None]

  origins += 0.0 * rays_dir
  return origins, rays_dir


def project(camera, points: jnp.ndarray):
  """Projects a 3D point (x,y,z) to a pixel position (x,y).

  Args:
    camera: camera object produced by `make_camera`.
    points: [A1, ..., An, 3] tensor or np.array containing world space points.

  Returns:
    [A1, ..., An, 2] tensor containing pixel coordinates.
  """
  batch_shape = points.shape[:-1]
  points = points.reshape((-1, 3))
  local_points = points_to_local_points(camera, points)

  # Get normalized local pixel positions.
  x = local_points[..., 0] / local_points[..., 2]
  y = local_points[..., 1] / local_points[..., 2]
  r2 = x**2 + y**2

  if "radial_distortion" in camera and "tangential_distortion" in camera:
    # Apply radial distortion.
    distortion = 1.0 + r2 * (
        camera["radial_distortion"][0] + r2 *
        (camera["radial_distortion"][1] + camera["radial_distortion"][2] * r2))

    # Apply tangential distortion.
    x_times_y = x * y
    x = (
        x * distortion + 2.0 * camera["tangential_distortion"][0] * x_times_y +
        camera["tangential_distortion"][1] * (r2 + 2.0 * x**2))
    y = (
        y * distortion + 2.0 * camera["tangential_distortion"][1] * x_times_y +
        camera["tangential_distortion"][0] * (r2 + 2.0 * y**2))

  # Map the distorted ray to the image plane and return the depth.
  pixel_x = camera["focal_length"] * x + camera["skew"] * y + camera[
      "principal_point"][0]
  pixel_y = (
      camera["focal_length"] * camera["pixel_aspect_ratio"] * y +
      camera["principal_point"][1])

  pixels = jnp.stack([pixel_x, pixel_y], axis=-1)
  return pixels.reshape((*batch_shape, 2))


def scale(camera, factor: float) -> CameraType:
  """Scales the camera by a given factor in image space.

  Args:
    camera: Camera object produced by `make_camera`.
    factor: The scale factor to apply.

  Returns:
    Modified camera with scaled resolution, focal length, and principal point.
  """
  new_camera = make_camera(
      orientation=camera["orientation"],
      position=camera["position"],
      focal_length=camera["focal_length"] * factor,
      principal_point=camera["principal_point"] * factor,
      skew=camera["skew"],
      pixel_aspect_ratio=camera["pixel_aspect_ratio"],
      radial_distortion=camera.get("radial_distortion"),
      tangential_distortion=camera.get("tangential_distortion"),
      image_size=jnp.array(camera["image_size"] * factor, dtype=int),
  )
  return new_camera


def make_neutral_camera(image_size: int = 256) -> CameraType:
  return make_camera(
      orientation=jnp.identity(3),
      position=jnp.zeros(3),
      focal_length=jnp.array([image_size]),
      principal_point=jnp.array([image_size // 2, image_size // 2]),
      image_size=jnp.array([image_size, image_size]))


def make_neutral_cameras(batch_size: int, image_size: int = 256) -> CameraType:
  """Generate neutral cameras."""
  camera = make_neutral_camera(image_size=image_size)
  cameras = jax.tree_map(
      lambda x: einops.repeat(x, "... -> N ...", N=batch_size), camera)
  return cameras


def camera_params_as_vector(camera: Dict[str, jnp.ndarray]) -> jnp.ndarray:
  """Extracts extrinsic and intrinsic matrix and flatten them to a vector."""
  # Extrinsic.
  rotation = camera["orientation"].reshape([-1])
  translation = camera["position"]
  last_row = jnp.array([0.0, 0.0, 0.0, 1.0])
  extrinsic = jnp.concatenate([rotation, translation, last_row],
                              axis=-1).reshape([16])

  # Intrinsic.
  skew = camera["skew"][0]
  ar = camera["pixel_aspect_ratio"][0]
  f = camera["focal_length"][0]
  cx = camera["principal_point"][..., 0]
  cy = camera["principal_point"][..., 1]
  intrinsic = jnp.array([f, skew, cx, 0.0, f * ar, cy, 0.0, 0.0,
                         1.0]).reshape([9])
  return jnp.concatenate([extrinsic, intrinsic], axis=-1)


def generate_pixel_grid(height: int, width: int) -> np.ndarray:
  """Compute image space coordinates for each pixel position."""
  row_indices = jnp.array(range(height))
  col_indices = jnp.array(range(width))
  pixel_indices = jnp.stack(jnp.meshgrid(col_indices, row_indices), axis=-1)

  # Offset coordinates by 0.5 to center sample points within each pixel.
  pixel_coordinates = pixel_indices.astype(jnp.float32) + 0.5
  return pixel_coordinates
