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

# Lint as: python3
"""Video rendering library."""

import datetime
from typing import List, Tuple

from absl import logging
import flax
import jax

import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic import datasets
from jax3d.projects.nesf.nerfstatic.datasets import dataset_utils
from jax3d.projects.nesf.nerfstatic.models import models
from jax3d.projects.nesf.nerfstatic.nerf import utils
from jax3d.projects.nesf.nerfstatic.utils import config as nerf_config
from jax3d.projects.nesf.nerfstatic.utils import render_utils
from jax3d.projects.nesf.nerfstatic.utils import semantic_utils
from jax3d.projects.nesf.nerfstatic.utils import train_utils
from jax3d.projects.nesf.nerfstatic.utils import types


def render(params: nerf_config.ConfigParams):
  """Render the latest checkpoint."""

  ##############################################################################
  # Initialization

  # Root directory for all renders.
  root_dir = (params.train.train_dir / "renders" /
              (params.render.label or _timestamp()))

  # Load rays to start from.
  logging.info("Loading dataset...")
  dataset = _load_dataset(params)

  # Load NeRFs.
  logging.info("Loading NeRFs...")
  nerf_states = semantic_utils.load_all_nerf_variables(
      save_dir=params.train.nerf_model_ckpt,
      train_dataset=dataset,
      novel_dataset=None,
      recompute_sigma_grid_opts=(
          semantic_utils.RecomputeSigmaGridOptions.from_params(params.train)))

  # Load semantic model.
  logging.info("Loading semantic model...")
  _, batch = next(iter(dataset))
  semantic_model, semantic_variables = _initialize_semantic_model(
      batch, nerf_states, params)

  # Iterate over starting frames.
  i = 0
  for _, batch in iter(dataset):
    if i >= params.render.num_start_frames:
      break

    ############################################################################
    # Setup output space
    scene_name, frame_name = _get_lineage(batch.target_view,
                                          dataset.all_metadata)
    logging.info("Renders will be based on scene=%s, frame=%d",
                 scene_name, frame_name)

    logging.info("Initializing output directory...")
    output_dir = root_dir / f"scene{scene_name}_frame{frame_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Renders will be written to: %s", str(output_dir))

    ############################################################################
    # Rendering
    renderer = render_utils.Renderer(
        nerf_states=nerf_states,
        semantic_model=semantic_model,
        semantic_variables=semantic_variables,
        max_rays_per_render_call=params.render.max_rays_per_render_call)

    if params.render.enable_video:
      logging.info("Rendering video...")
      _render_video(renderer, batch, output_dir, params)

    # Render epipolar plane images
    if params.render.enable_epipolar:
      logging.info("Rendering epipolar plane images...")
      _render_epipolar_plane_images(renderer, batch, output_dir, params)

    # Render SparseConvNet videos
    if params.render.render_sparseconvnet:
      logging.info("Rendering videos for SparseConvNet...")
      cam_id = int(frame_name[9:])
      rendered_images = render_utils.render_sparseconvnet_video(
          params.datasets.data_dir,
          params.render.sparseconvnet_predictions_path,
          scene_name, cam_id, params.render.num_video_frames,
          params.datasets.enable_sqrt2_buffer)
      render_utils.write_semantic_images_video(
          filepath=(output_dir / "sparseconvnet_spiral.mp4"),
          rendered_images=rendered_images,
          fps=params.render.video_fps)

    i += 1


################################################################################
# Helper methods


def _timestamp():
  """Constructs the time a string."""
  now = datetime.datetime.now()
  fmt_str = "%Y-%m-%d_%H-%M-%S.%f"  # e.g. 2021-11-07_12-35-59.982734
  return now.strftime(fmt_str)


def _load_dataset(params: nerf_config.ConfigParams,
                  ) -> datasets.DatasetIterable:
  """Loads a dataset into memory."""
  split, is_novel_scenes = render_utils.dataset_split(
      params.render.dataset_split)
  dataset = datasets.get_dataset(
      split=split,
      args=params.datasets,
      model_args=params.models,
      example_type=datasets.ExampleType.IMAGE,
      is_novel_scenes=is_novel_scenes)
  return dataset


def _get_lineage(views: types.Views,
                 scene_metadatas: List[dataset_utils.DatasetMetadata],
                 ) -> Tuple[str, str]:
  scene_idx = views.rays.scene_id[0, 0, 0]
  scene_name = scene_metadatas[scene_idx].scene_name
  image_name = str(views.image_ids)
  return scene_name, image_name


def _initialize_semantic_model(batch: types.Batch,
                               nerf_states: semantic_utils.RecoveredNeRFModel,
                               params: nerf_config.ConfigParams,
                               ):
  """Initializes and restores a semantic model from checkpoint."""
  placeholder_batch, _ = batch.pop_image_id_stateless()
  placeholder_batch = jax.tree_map(lambda t: t[0, 0, ...], placeholder_batch)
  initialized_model = models.construct_volumetric_semantic_model(
      rng=j3d.RandomState(0),
      num_scenes=-1,
      placeholder_batch=placeholder_batch,
      args=params.models,
      nerf_model=nerf_states.model,
      nerf_sigma_grid=nerf_states.train_sigma_grids[0],
      nerf_variables=nerf_states.train_variables[0])
  semantic_model = initialized_model.model
  semantic_variables = initialized_model.variables
  del placeholder_batch, initialized_model

  optimizer = flax.optim.Adam(params.train.lr_init).create(semantic_variables)
  train_state = utils.TrainState(optimizer=optimizer)
  del optimizer

  # Load checkpoint
  ckpt_dir = train_utils.checkpoint_dir(params)
  train_state = train_utils.restore_opt_checkpoint(save_dir=ckpt_dir,
                                                   state=train_state)
  semantic_variables = train_state.optimizer.target
  step = train_state.optimizer.state.step
  logging.info("Loaded checkpoint at step=%d", step)
  del ckpt_dir, train_state, step

  return semantic_model, semantic_variables


def _render_video(renderer, batch, output_dir, params):
  """Renders a video and writes it to disk."""
  assert params.render.camera_path_method == "spiral"
  logging.info("Rendering video with %d frames using camera path %s",
               params.render.num_video_frames,
               params.render.camera_path_method)
  video_rays = render_utils.spiral(batch.target_view.rays,
                                   params.render.num_video_frames)
  rendered_video_rays = renderer.render_video(video_rays)
  render_utils.write_rendered_video_rays(
      directory=output_dir,
      name=params.render.camera_path_method,
      rendered_video_rays=rendered_video_rays,
      min_depth=params.render.min_depth,
      max_depth=params.render.max_depth,
      fps=params.render.video_fps)


def _render_epipolar_plane_images(renderer, batch, output_dir, params):
  """Renders EPIs and writes them to disk."""
  for row_idx in params.render.epipolar_row_idxs:
    logging.info("Rendering EPI for row=%d", row_idx)
    rendered_image_rays = renderer.render_epipolar_plane_image(
        batch.target_view.rays,
        num_rows=params.render.epipolar_num_rows,
        row_idx=row_idx)
    render_utils.write_epipolar_plane_image(
        directory=(output_dir / "epipolar"),
        name=f"{row_idx:03d}",
        reference_views=batch.target_view,
        rendered_rays=rendered_image_rays,
        min_depth=params.render.min_depth,
        max_depth=params.render.max_depth,
        row_idx=row_idx)
