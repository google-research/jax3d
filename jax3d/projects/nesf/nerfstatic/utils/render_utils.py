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

"""Utilities for rendering."""


import functools
from typing import Any, Dict, Tuple

from absl import logging
import jax
import jax.numpy as jnp
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.models import volumetric_semantic_model as vsm
from jax3d.projects.nesf.nerfstatic.nerf import utils
from jax3d.projects.nesf.nerfstatic.utils import camera_utils
from jax3d.projects.nesf.nerfstatic.utils import geometry_utils
from jax3d.projects.nesf.nerfstatic.utils import semantic_utils
from jax3d.projects.nesf.nerfstatic.utils import stopwatch_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import PRNGKey, Tree, f32  # pylint: disable=g-multiple-import
import kubric.plotting as kb_plotting
from matplotlib import cm
import mediapy
import numpy as np


def colorize_semantics(logits: jnp.ndarray,
                       blend_logits: bool = False,
                       ) -> jnp.ndarray:
  """Convert semantic map or semantic logits map into RGB."""
  num_categories = 14
  if len(logits.shape) >= 3 and logits.shape[-1] == 1:
    semantic = logits[..., 0].astype(np.uint8)
  elif len(logits.shape) >= 3 and logits.shape[-1] > 1:
    if blend_logits:
      # Average colors weighted according to probability.
      num_categories = logits.shape[-1]
      weights = jax.nn.softmax(logits, axis=-1)
      semantic = jnp.arange(num_categories, dtype=jnp.uint8)
      semantic = jnp.broadcast_to(semantic, weights.shape)
    else:
      semantic = jnp.argmax(logits, axis=-1)
  elif len(logits.shape) == 2:
    semantic = logits.astype(np.uint8)
  else:
    raise ValueError(logits.shape)

  # Assign color to each pixel
  colormap = kb_plotting.hls_palette(num_categories) / 255.
  semantic_flat = jnp.reshape(semantic, -1)
  result = colormap[semantic_flat]
  result = jnp.reshape(result, semantic.shape + (3,))

  # Average colors weighted according to probability
  if len(logits.shape) >= 3 and logits.shape[-1] > 1 and blend_logits:
    result = jnp.einsum("...xy,...x->...y", result, weights)

  return result


def dataset_split(split: str) -> Tuple[str, bool]:
  """Parameters for get_dataset()."""
  split = split.lower()
  if split == "eval_train":
    return "train", False
  elif split == "eval_test":
    return "test", False
  elif split == "eval_novel_train":
    return "train", True
  elif split == "eval_train":
    return "test", True
  raise NotImplementedError(split)


def colorize_depth(depth):
  depth = jnp.where(depth < 0, 0., depth)
  depth = jnp.where(depth >= 1.0, 1.0, depth)
  print(depth.min(), depth.max())
  return cm.get_cmap("turbo")(depth)[..., 0:3]


def disparity_to_depth(disparity: f32["... 1"],
                       opacity: f32["... 1"],
                       max_depth: float,
                       ) -> f32["... 1"]:
  """Converts disparity and opacity to a depth map."""
  depth = opacity / disparity
  # depth = 1.0 / disparity
  # depth = depth * opacity + max_depth * (1-opacity)
  depth = max_depth - depth
  return depth


def normalize_depth(depth: f32["... 1"],
                    min_depth: float,
                    max_depth: float,
                    ) -> f32["... 3"]:
  """Normalizes a depth map between [0, 1]."""
  # Reduce to a (H, W) from (H, W, 1)
  depth = depth[..., 0]

  # Clip depth if necessary.
  logging.info("before clipping: (%s, %s)", jnp.min(depth), jnp.max(depth))
  depth = jnp.clip(depth, min_depth, max_depth)
  logging.info("after clipping: (%s, %s)", jnp.min(depth), jnp.max(depth))

  # Normalize dept map between near and far
  depth = (depth - min_depth) / (max_depth - min_depth)

  return jax.device_get(depth)


def spiral(rays, num_frames, num_rotations=1., z_delta=1.):
  """Construct camera paths in a spiral about the z-axis."""
  num_frames_per_rotation = num_frames / num_rotations
  rotation_angle_per_frame = (2*np.pi) / num_frames_per_rotation
  z_delta_per_frame = z_delta / num_frames_per_rotation

  def create_transform(angle, z):
    rotation = geometry_utils.Rotate(axis=jnp.array([0, 0, 1]), radians=angle)
    translation = geometry_utils.Translate(offset=np.array([0, 0, -1]) * z)
    transform = geometry_utils.Compose(transforms=[rotation, translation])
    return transform

  # Apply inverse translation for N/2 frames.
  rays = (
      create_transform(rotation_angle_per_frame * num_frames/2,
                       z_delta_per_frame * num_frames/2)
      .backward(rays)
  )

  # Apply forward
  results = []
  for i in range(num_frames):
    results.append(
        create_transform(i * rotation_angle_per_frame,
                         i * z_delta_per_frame)
        .forward(rays)
    )
  results = _nested_stack(results)
  return results


def truck(rays: types.Rays,
          num_frames: int,
          distance: float = 1.0,
          ) -> types.Rays:
  """Construct camera paths by moving the camera laterally."""
  delta_per_frame = distance / num_frames

  def rightward(rays):
    normalize = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    d = rays.direction
    h, w, _ = d.shape
    up = d[-1, w//2]
    forward = d[h//2, w//2]
    right = normalize(np.cross(forward, up))
    return right

  right = rightward(rays)

  def create_transform(delta):
    translation = geometry_utils.Translate(offset=np.array(right) * delta)
    return translation

  # Apply inverse translation for N/2 frames.
  rays = (
      create_transform(delta_per_frame * num_frames/2)
      .backward(rays)
  )

  # Apply forward
  results = []
  for i in range(num_frames):
    results.append(
        create_transform(i*delta_per_frame)
        .forward(rays)
    )
  results = _nested_stack(results)
  return results


def predict_fn_2d(
    rng: PRNGKey,
    rays: types.Rays,
    nerf_variables: Tree[jnp.ndarray],
    nerf_sigma_grid: f32["n x y z c"],
    *,
    semantic_variables: Tree[jnp.ndarray],
    semantic_model: vsm.VolumetricSemanticModel,
    ) -> types.RenderedRays:
  """Renders a batch of rays.

  Args:
    rng: jax3d random state.
    rays: types.Rays. Each ndarray has leading dimension of length
      'batch_size_per_device'.
    nerf_variables: NeRF Model's variables
    nerf_sigma_grid: NeRF sigma grid.
    semantic_variables: Semantic model variables.
    semantic_model: Semantic model for rendering.

  Returns:
    Nested collection of arrays, each of shape,
      [num_total_devices, batch_size_per_device, ...]
  """
  rng_names = ["params", "sampling", "data_augmentation"]
  rng, *rng_keys = jax.random.split(rng, len(rng_names) + 1)
  return jax.lax.all_gather(  # gathers over all devices
      semantic_model.apply(
          semantic_variables,
          rngs=dict(zip(rng_names, rng_keys)),
          rays=rays,
          sigma_grid=nerf_sigma_grid,
          randomized_sampling=False,
          is_train=False,
          nerf_model_weights=nerf_variables,),
      axis_name="batch")


class Renderer:
  """Utility for rendering images, videos, and more."""

  def __init__(self,
               nerf_states: semantic_utils.RecoveredNeRFModel,
               semantic_model: vsm.VolumetricSemanticModel,
               semantic_variables: Tree[jnp.ndarray],
               *,
               max_rays_per_render_call: int = 4096):
    self._nerf_states = nerf_states
    self._semantic_model = semantic_model
    self._semantic_variables = semantic_variables
    self._max_rays_per_render_call = max_rays_per_render_call

    # Construct function for rendering batches of rays. We build it now to
    # avoid repeated XLA compilations every time the function is called.
    self._checkpoint_render_fn = self._build_checkpoint_render_fn()

  def render_image(self, image_rays: types.Rays) -> types.RenderedRays:
    """Renders a single image."""
    assert len(image_rays.batch_shape) == 2

    all_nerf_variables = self._nerf_states.train_variables
    all_nerf_sigma_grids = self._nerf_states.train_sigma_grids

    # scene_id associated with pixel (0,0). It is assumed that all pixels in
    # this frame share the same scene_id.
    scene_id = image_rays.scene_id[0, 0, 0]

    # nerf_variables and nerf_sigma_grid's Tensors have shape [1, 1, ...].
    # TODO(svora): Clean up select and stack for non vmap case.
    nerf_variables = semantic_utils.select_and_stack([scene_id],
                                                     all_nerf_variables,
                                                     num_devices=1)
    nerf_sigma_grid = semantic_utils.select_and_stack([scene_id],
                                                      all_nerf_sigma_grids,
                                                      num_devices=1)
    rendered_rays = self._checkpoint_render_fn(
        nerf_variables=jax.tree_map(lambda x: x[0, 0], nerf_variables),
        nerf_sigma_grid=jax.tree_map(lambda x: x[0, 0], nerf_sigma_grid),
        rays=image_rays)

    # Copy results to host. This prevents device memory from getting cluttered.
    rendered_rays = jax.tree_map(jax.device_get, rendered_rays)

    return rendered_rays

  def render_video(self, video_rays: types.Rays) -> types.RenderedRays:
    """Renders a series of frames."""
    assert len(video_rays.batch_shape) == 3

    results = []
    stopwatch = stopwatch_utils.Stopwatch()
    num_frames, _, _ = video_rays.batch_shape
    for i in range(num_frames):
      logging.info("[%8.3f] Rendering frame %04d/%04d",
                   stopwatch.delta(), i+1, num_frames)
      rays = jax.tree_map(lambda x: x[i], video_rays)  # pylint: disable=cell-var-from-loop
      rendered_rays = self.render_image(rays)
      results.append(rendered_rays)
    results = _nested_stack(results)
    return results

  def render_epipolar_plane_image(self,
                                  image_rays: types.Rays,
                                  num_rows: int,
                                  row_idx: int,
                                  ) -> types.RenderedRays:
    """Renders a epipolar plane image."""
    assert len(image_rays.batch_shape) == 2

    # [num_frames, height, width]
    video_rays = truck(image_rays, num_rows)

    # [num_frames, width]
    center_rays = jax.tree_map(lambda x: x[:, row_idx], video_rays)

    # [1, num_frames, width]
    center_rays = jax.tree_map(lambda x: x[None, ...], center_rays)

    # [1, num_frames, width]
    rendered_video_rays = self.render_video(center_rays)

    # [num_frames, width]
    result = jax.tree_map(lambda x: x[0], rendered_video_rays)

    return result

  def _build_checkpoint_render_fn(self):
    """Constructs function for rendering bags of rays."""
    render_fn = functools.partial(predict_fn_2d,
                                  semantic_variables=self._semantic_variables,
                                  semantic_model=self._semantic_model)

    # Parallelize render_fn. All variables except for 'rays' are copied to
    # all devices
    render_pfn = jax.pmap(render_fn,
                          axis_name="batch",
                          in_axes=(None, 0, None, None))

    # Construct a function for processing sub-batches of rays.
    checkpoint_render_fn = functools.partial(
        utils.predict_2d_semantic,
        render_pfn=render_pfn,
        rng=j3d.RandomState(42),
        chunk=self._max_rays_per_render_call)

    return checkpoint_render_fn


def write_rendered_video_rays(directory: j3d.Path,
                              name: str,
                              rendered_video_rays: types.RenderedRays,
                              min_depth: float,
                              max_depth: float,
                              **kwargs):
  """Writes rgb, semantic, and depth videos."""
  write_rgb_video(directory / f"{name}_rgb.mp4",
                  rendered_video_rays,
                  **kwargs)
  write_semantic_video(directory / f"{name}_semantic.mp4",
                       rendered_video_rays,
                       **kwargs)
  write_depth_video(directory / f"{name}_depth.mp4",
                    rendered_video_rays,
                    min_depth,
                    max_depth,
                    **kwargs)


def write_rgb_video(filepath: j3d.Path,
                    rendered_video_rays: types.RenderedRays,
                    **kwargs):
  """Writes an RGB video."""
  return write_video(filepath, rendered_video_rays.rgb, **kwargs)


def write_semantic_video(filepath: j3d.Path,
                         rendered_video_rays: types.RenderedRays,
                         **kwargs):
  """Writes a semantic map video."""
  video = colorize_semantics(rendered_video_rays.semantic, blend_logits=True)
  return write_video(filepath, video, **kwargs)


def write_depth_video(filepath: j3d.Path,
                      rendered_video_rays: types.RenderedRays,
                      min_depth: float,
                      max_depth: float,
                      **kwargs):
  """Writes a depth map video."""
  video = disparity_to_depth(rendered_video_rays.disparity,
                             rendered_video_rays.opacity,
                             max_depth)
  video = colorize_depth(normalize_depth(video, min_depth, max_depth))
  return write_video(filepath, video, **kwargs)


def write_video(filepath: j3d.Path, video: f32["n h w 3"], **kwargs):
  """Writes video frames to disk."""
  filepath.parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(filepath, video, **kwargs)


def write_epipolar_plane_image(directory: j3d.Path,
                               name: str,
                               reference_views: types.Views,
                               rendered_rays: types.RenderedRays,
                               row_idx: int,
                               min_depth: float,
                               max_depth: float,
                               *,
                               blend_logits: bool = False):
  """Writes images related to epipolar plane visualizations."""
  rgb_reference = _draw_red_line(reference_views.rgb, row_idx)
  semantic_reference = colorize_semantics(reference_views.semantics,
                                          blend_logits=blend_logits)
  semantic_reference = _draw_red_line(semantic_reference, row_idx)
  epi_rgb = rendered_rays.rgb
  epi_semantic = colorize_semantics(rendered_rays.semantic,
                                    blend_logits=blend_logits)
  epi_depth = disparity_to_depth(rendered_rays.disparity,
                                 rendered_rays.opacity,
                                 max_depth)
  epi_depth = colorize_depth(normalize_depth(epi_depth, min_depth, max_depth))

  directory.mkdir(parents=True, exist_ok=True)
  mediapy.write_image(directory / f"{name}_rgb_reference.png", rgb_reference)
  mediapy.write_image(directory / f"{name}_semantic_reference.png",
                      semantic_reference)
  mediapy.write_image(directory / f"{name}_rgb.png", epi_rgb)
  mediapy.write_image(directory / f"{name}_depth.png", epi_depth)
  mediapy.write_image(directory / f"{name}_semantic.png", epi_semantic)


def write_semantic_images_video(
    filepath: j3d.Path,
    rendered_images: f32["..."],
    **kwargs):
  """Writes a semantic map video."""
  video = colorize_semantics(rendered_images[..., None], blend_logits=False)
  return write_video(filepath, video, **kwargs)


################################################################################
# Helper functions


def _nested_stack(x):
  return jax.tree_map(lambda *args: np.stack(args), *x)


def _draw_red_line(x: f32["h w c"], idx: int) -> f32["h w c"]:
  x = np.array(x)
  x[idx] = np.array([1, 0, 0])
  return x


def _get_kubric_camera(camera_metadata: Dict[str, Any], ids: int):
  return camera_utils.Camera.from_position_and_quaternion(
      positions=np.array(camera_metadata["positions"])[ids],
      quaternions=np.array(camera_metadata["quaternions"])[ids],
      resolution=(camera_metadata["height"], camera_metadata["width"]),
      # Assume square pixels: width / sensor_width == height / sensor_height
      focal_px_length=(camera_metadata["focal_length"] *
                       camera_metadata["width"] /
                       camera_metadata["sensor_width"]),
      use_unreal_axes=False,
  )
