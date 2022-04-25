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

"""Semantic Evaluation library."""

import functools
import time
from typing import Callable, Optional

from absl import logging
import flax
import jax
from jax import numpy as jnp

import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic import datasets
from jax3d.projects.nesf.nerfstatic.models import models
from jax3d.projects.nesf.nerfstatic.nerf import utils
from jax3d.projects.nesf.nerfstatic.utils import config as nerf_config
from jax3d.projects.nesf.nerfstatic.utils import eval_utils
from jax3d.projects.nesf.nerfstatic.utils import img_log_utils
from jax3d.projects.nesf.nerfstatic.utils import jax_process_zero
from jax3d.projects.nesf.nerfstatic.utils import render_utils
from jax3d.projects.nesf.nerfstatic.utils import semantic_utils
from jax3d.projects.nesf.nerfstatic.utils import train_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import PRNGKey, Tree, f32  # pylint: disable=g-multiple-import


# Type annotations for use in this file.
NerfSigmaGrids = f32["1 x y z c"]
NerfVariables = Tree[jnp.ndarray]
CheckpointRenderFn2d = Callable[[PRNGKey,
                                 types.Rays,
                                 NerfVariables,
                                 NerfSigmaGrids],
                                types.RenderedRays]
CheckpointRenderFn3d = Callable[[PRNGKey,
                                 types.SamplePoints,
                                 NerfVariables,
                                 NerfSigmaGrids],
                                f32["n k"]]


def evaluate(
    params: nerf_config.ConfigParams,
):
  """Semantic evaluation loop.

  Responsible for,
  - Rendering images,
  - Computing loss metrics,
  - Publishing TensorBoard metrics,

  Args:
    params: All training parameters (e.g. train, model, dataset, loss params).

  """

  max_steps = train_utils.get_max_steps(params)
  if not max_steps:
    return

  if params.train.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")

  rng = jax3d.RandomState(params.train.random_seed)

  logging.info("Loading the datasets.")
  # Load datasets.
  #
  # WARNING: It is CRITICAL that each host receive the same test set with the
  # same images in the same order as all other hosts in a multi-host
  # computation. If this is not the case, everything downstream will break.
  #
  train_dataset = datasets.get_dataset(
      split="train",
      args=params.datasets,
      model_args=params.models,
      example_type=datasets.ExampleType.IMAGE,
      is_novel_scenes=False)
  test_dataset = datasets.get_dataset(
      split="test",
      args=params.datasets,
      model_args=params.models,
      example_type=datasets.ExampleType.IMAGE,
      is_novel_scenes=False)
  novel_dataset_train = datasets.get_dataset(
      split="train",
      args=params.datasets,
      model_args=params.models,
      example_type=datasets.ExampleType.IMAGE,
      is_novel_scenes=True)
  novel_dataset_test = datasets.get_dataset(
      split="test",
      args=params.datasets,
      model_args=params.models,
      example_type=datasets.ExampleType.IMAGE,
      is_novel_scenes=True)

  placeholder_batch = test_dataset.peek()[1]
  placeholder_batch = jax.tree_map(lambda t: t[0, 0, ...], placeholder_batch)

  # Initialize & load per-scene NeRF models.
  logging.info("Initializing pretrained NeRF models.")
  recovered_nerf_state = semantic_utils.load_all_nerf_variables(
      save_dir=params.train.nerf_model_ckpt,
      train_dataset=test_dataset,
      novel_dataset=novel_dataset_test,
      recompute_sigma_grid_opts=(
          semantic_utils.RecomputeSigmaGridOptions.from_params(params.train)))

  # Initialize semantic model.
  logging.info("Initializing semantic model.")
  initialized_vol_sem_model = models.construct_volumetric_semantic_model(
      # We will initialize a model here, but overwrite that model from a
      # checkpoint further down.
      rng=jax3d.RandomState(0),
      num_scenes=-1,
      placeholder_batch=placeholder_batch,
      args=params.models,
      nerf_model=recovered_nerf_state.model,
      nerf_sigma_grid=recovered_nerf_state.train_sigma_grids[0],
      nerf_variables=recovered_nerf_state.train_variables[0])
  vol_sem_model = initialized_vol_sem_model.model
  semantic_variables = initialized_vol_sem_model.variables
  del initialized_vol_sem_model

  logging.info("Intializing optimizer.")
  optimizer = flax.optim.Adam(params.train.lr_init).create(semantic_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer

  last_step = 0

  # If evaluating a single checkpoint, we will not write TensorBoard metrics.
  assert not params.evaluate.eval_once
  train_summary_writer = jax_process_zero.SummaryWriter(
      params.train.train_dir / "eval_train")
  test_summary_writer = jax_process_zero.SummaryWriter(
      params.train.train_dir / "eval_test")
  novel_train_summary_writer = jax_process_zero.SummaryWriter(
      params.train.train_dir / "eval_novel_train")
  novel_test_summary_writer = jax_process_zero.SummaryWriter(
      params.train.train_dir / "eval_novel_test")

  while True:
    # Load latest checkpoint. Ignore unused dataset state.
    save_dir = train_utils.checkpoint_dir(params)

    logging.info("Loading latest checkpoint from %s", save_dir)
    state = train_utils.restore_opt_checkpoint(save_dir=save_dir, state=state)
    step = int(state.optimizer.state.step)

    # Only evaluate newer checkpoints.
    if step <= last_step:
      time.sleep(10)
      continue

    ############################################################################
    # Compile rendering functions. We compile in the variables to the sematnic
    # model to reduce the amount of host-to-device IO per call.

    # Compile 2D prediction function. Return value is of shape [d D ...] where,
    #   d - number of local devices
    #   D - number of total devices
    predict_pfn_2d = jax.pmap(
        functools.partial(render_utils.predict_fn_2d,
                          semantic_variables=state.optimizer.target,
                          semantic_model=vol_sem_model),
        axis_name="batch",
        in_axes=(None, 0, None, None),
    )
    checkpoint_render_fn_2d = functools.partial(
        utils.predict_2d_semantic,
        render_pfn=predict_pfn_2d,
        chunk=params.evaluate.chunk)

    # Compile 3D prediction function. Return value is of shape [d D n k] where,
    #   d - number of local devices
    #   D - number of total devices
    #   n - number of points in sample_points
    #   k - number of semantic categories
    predict_pfn_3d = jax.pmap(
        functools.partial(eval_utils.predict_fn_3d,
                          semantic_variables=state.optimizer.target,
                          semantic_model=vol_sem_model),
        axis_name="batch",
        in_axes=(None, 0, None, None))
    checkpoint_render_fn_3d = functools.partial(
        utils.predict_3d_semantic,
        render_pfn=predict_pfn_3d,
        chunk=params.evaluate.chunk,
        num_semantic_classes=params.models.num_semantic_classes)
    ############################################################################

    shared_params = {
        "checkpoint_render_fn_2d": checkpoint_render_fn_2d,
        "checkpoint_render_fn_3d": checkpoint_render_fn_3d,
        "step": step,
        "rng": rng,
        "num_semantic_classes": params.models.num_semantic_classes,
        "num_images": params.evaluate.eval_num_images,
        "root_out_dir": params.train.train_dir / "test_preds",
        "write_predictions_to_disk": params.evaluate.write_predictions_to_disk,
    }

    if params.evaluate.enable_eval_train:
      logging.info("Evaluating training frames.")
      _evaluate_dataset(
          name="eval_train",
          dataset=train_dataset,
          summary_writer=train_summary_writer,
          all_nerf_variables=recovered_nerf_state.train_variables,
          all_nerf_sigma_grids=recovered_nerf_state.train_sigma_grids,
          num_log_images=params.evaluate.eval_num_log_images,
          **shared_params)

    if params.evaluate.enable_eval_test:
      logging.info("Evaluating test frames.")
      # The following logic assumes that the train dataset and eval dataset draw
      # frames from the same scenes. If this is not the case, the following code
      # will produce invalid renders.
      assert params.datasets.train_scenes == params.datasets.eval_scenes
      _evaluate_dataset(
          name="eval_test",
          dataset=test_dataset,
          summary_writer=test_summary_writer,
          all_nerf_variables=recovered_nerf_state.train_variables,
          all_nerf_sigma_grids=recovered_nerf_state.train_sigma_grids,
          num_log_images=params.evaluate.eval_num_log_images,
          **shared_params)

    if params.evaluate.enable_eval_novel_train:
      logging.info("Evaluating novel train scenes.")
      _evaluate_dataset(
          name="eval_novel_train",
          dataset=novel_dataset_train,
          summary_writer=novel_train_summary_writer,
          all_nerf_variables=recovered_nerf_state.novel_variables,
          all_nerf_sigma_grids=recovered_nerf_state.novel_sigma_grids,
          num_log_images=params.evaluate.eval_num_log_images,
          **shared_params)

    if params.evaluate.enable_eval_novel_test:
      logging.info("Evaluating novel test scenes.")
      _evaluate_dataset(
          name="eval_novel_test",
          dataset=novel_dataset_test,
          summary_writer=novel_test_summary_writer,
          all_nerf_variables=recovered_nerf_state.novel_variables,
          all_nerf_sigma_grids=recovered_nerf_state.novel_sigma_grids,
          num_log_images=params.evaluate.eval_num_log_images,
          **shared_params)

    if int(step) >= max_steps:
      break

    last_step = step


def _evaluate_dataset(
    name: str,
    dataset: datasets.DatasetIterable,
    summary_writer: jax_process_zero.SummaryWriter,
    checkpoint_render_fn_2d: CheckpointRenderFn2d,
    checkpoint_render_fn_3d: CheckpointRenderFn3d,
    step: int,
    rng: jax3d.RandomState,
    all_nerf_variables: NerfVariables,
    all_nerf_sigma_grids: NerfSigmaGrids,
    num_images: Optional[int],
    num_log_images: int,
    num_semantic_classes: int,
    root_out_dir: jax3d.Path,
    write_predictions_to_disk: bool,
):
  """Evaluates a single model checkpoint on an entire dataset."""

  logging.info("Evaluating checkpoint at step=%d on dataset=%s.", step, name)
  image_log = img_log_utils.PredictionImageLog()
  confusion_matrix_2d = jnp.zeros(2 * [num_semantic_classes])
  confusion_matrix_3d = jnp.zeros(2 * [num_semantic_classes])

  for image_idx, (_, batch) in enumerate(iter(dataset)):
    if num_images and image_idx >= num_images:
      break

    logging.info("Rendering image_id=%s", image_idx)
    batch, image_ids = batch.pop_image_id_stateless()

    # scene_id associated with pixel (0,0). It is assumed that all pixels in
    # this frame share the same scene_id.
    scene_id = batch.target_view.rays.scene_id[0, 0, 0]

    # nerf_variables and nerf_sigma_grid's Tensors have shape [1, 1, ...].
    # TODO(svora): Clean up select and stack for non vmap case.
    nerf_variables = semantic_utils.select_and_stack([scene_id],
                                                     all_nerf_variables,
                                                     num_devices=1)
    nerf_sigma_grid = semantic_utils.select_and_stack([scene_id],
                                                      all_nerf_sigma_grids,
                                                      num_devices=1)
    # Render image.
    rendered_rays = checkpoint_render_fn_2d(
        rng=rng,
        rays=batch.target_view.rays,
        nerf_variables=jax.tree_map(lambda x: x[0, 0], nerf_variables),
        nerf_sigma_grid=jax.tree_map(lambda x: x[0, 0], nerf_sigma_grid),
    )

    # Make 3D predictions.
    semantic_logits_3d = None
    if num_semantic_classes > 0:
      semantic_logits_3d = checkpoint_render_fn_3d(
          rng=rng,
          sample_points=(
              eval_utils.create_sample_points_for_3d_semantics(
                  batch.target_view)),
          nerf_variables=jax.tree_map(lambda x: x[0, 0], nerf_variables),
          nerf_sigma_grid=jax.tree_map(lambda x: x[0, 0], nerf_sigma_grid),
      )

    # Keep images for later.
    image_log.append_images(
        image_key=eval_utils.get_image_key_from_image_ids(image_ids),
        rgb=rendered_rays.rgb,
        rgb_ground_truth=batch.target_view.rgb,
        semantic_logits=rendered_rays.semantic,
        semantic_ground_truth=batch.target_view.semantics)

    if num_semantic_classes > 0:
      # Update 2D confusion matrix.
      curr_conf_mat_2d = utils.compute_conf_mat_from_preds(
          logits=rendered_rays.semantic,
          labels=batch.target_view.semantics,
          num_classes=num_semantic_classes)
      confusion_matrix_2d = confusion_matrix_2d + curr_conf_mat_2d

      # Update 3D confusion matrix.
      curr_conf_mat_3d = utils.compute_conf_mat_from_preds(
          logits=semantic_logits_3d,
          labels=batch.target_view.point_cloud.semantics,
          num_classes=num_semantic_classes)
      confusion_matrix_3d = confusion_matrix_3d + curr_conf_mat_3d

  iou_2d = utils.compute_iou_from_con_mat(confusion_matrix_2d)
  iou_3d = utils.compute_iou_from_con_mat(confusion_matrix_3d)

  # Write scalar and text metrics to TensorBoard
  logging.info("Publishing metrics to TensorBoard")
  if num_semantic_classes:
    summary_writer.scalar("metrics/mean_iou_2d", iou_2d.mean_iou, step)
    summary_writer.scalar("metrics/mean_iou_3d", iou_3d.mean_iou, step)
    summary_writer.text(
        f"[{name}] metrics/conf_mat_2d",
        eval_utils.markdown(utils.fmt_confmat(
            confusion_matrix_2d, dataset.semantic_labels)),
        step=step)
    summary_writer.text(
        f"[{name}] metrics/conf_mat_3d",
        eval_utils.markdown(utils.fmt_confmat(
            confusion_matrix_3d, dataset.semantic_labels)),
        step=step)

  # Write image logs to TensorBoard. Only up to 'num_log_images' entries will
  # appear in TensorBoard; the rest are written to disk.
  summary_writer.image(f"[{name}] rgb",
                       image_log.rgb[0:num_log_images],
                       max_outputs=num_log_images,
                       step=step)
  summary_writer.image(f"[{name}] rgb_ground_truth",
                       image_log.rgb_ground_truth[0:num_log_images],
                       max_outputs=num_log_images,
                       step=step)
  if num_semantic_classes:
    summary_writer.image(
        f"[{name}] semantic",
        utils.get_color_coded_semantics_image(
            image_log.semantic[0:num_log_images, ..., 0]),
        max_outputs=num_log_images,
        step=step)
    summary_writer.image(
        f"[{name}] semantic_ground_truth",
        utils.get_color_coded_semantics_image(
            image_log.semantic_ground_truth[0:num_log_images, ..., 0]),
        max_outputs=num_log_images,
        step=step)

  # Write images to disk.
  if write_predictions_to_disk:
    logging.info("Writing predictions to disk.")
    out_dir = root_out_dir / name / f"step-{step:07d}"
    image_log.write_images_to_disk(out_dir)
