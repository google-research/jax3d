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

"""Evaluation library.

By default, continually evaluates the latest model checkpoints found in
--train_dir until the final checkpoint according to --train_steps is reached.
For each checkpoint, this binary does the following,

  1. Renders all camera poses in the "test" split.
  2. Evaluates metrics (== PSNR, SSIM) on rendered images. Metrics are
     published to TensorBoard and written to disk.
  3. Publishes rendered images to CNS path,
       ${TRAIN_DIR}/test_preds/step-${GLOBAL_STEP}
  4. Randomly select --eval_num_log_images frames which are published to
     Tensorboard. The same frames are used for all checkpoints.
"""
import dataclasses
import functools
import itertools
import time
from typing import Any, Callable, List, Optional

from absl import logging
import chex
from etils import etqdm as tqdm
import flax
from flax import traverse_util
import jax
from jax import numpy as jnp

import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic import datasets
from jax3d.projects.nesf.nerfstatic import serialization
from jax3d.projects.nesf.nerfstatic.metrics import metrics
from jax3d.projects.nesf.nerfstatic.models import model_utils
from jax3d.projects.nesf.nerfstatic.models import models
from jax3d.projects.nesf.nerfstatic.nerf import utils
from jax3d.projects.nesf.nerfstatic.utils import config as nerf_config
from jax3d.projects.nesf.nerfstatic.utils import eval_utils
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.nerfstatic.utils import img_utils
from jax3d.projects.nesf.nerfstatic.utils import jax_process_zero
from jax3d.projects.nesf.nerfstatic.utils import train_utils
from jax3d.projects.nesf.utils.typing import f32


@chex.dataclass
class PredictionImageLog:
  """Dataclass to store images for logging to Tensorboard.

  For each component we store a list of images, which are stacked right before
  logging to tensorboard.
  All components should have values in [0, 1].
  """
  # Predicted images.
  rgb: List[f32["h w 3"]] = dataclasses.field(default_factory=list)
  # Predicted foreground.
  rgb_foreground: List[f32["h w 3"]] = dataclasses.field(default_factory=list)
  # Disparity maps.
  disparity: List[f32["h w 1"]] = dataclasses.field(default_factory=list)
  # Opacity of the image.
  opacity: List[f32["h w 1"]] = dataclasses.field(default_factory=list)
  # Ground truth images.
  rgb_ground_truth: List[f32["h w 3"]] = dataclasses.field(default_factory=list)
  # Input images.
  input_images: List[f32["h w 3"]] = dataclasses.field(default_factory=list)
  # Color coded semantic images.
  semantic: List[f32["h w 3"]] = dataclasses.field(default_factory=list)
  # Color coded semantic foreground.
  semantic_foreground: List[f32["h w 3"]] = dataclasses.field(
      default_factory=list)
  # Color coded ground truth semantic images.
  semantic_ground_truth: List[f32["h w 3"]] = dataclasses.field(
      default_factory=list)
  # RGB diff images: rgb - rgb_ground_truth
  rgb_diff: List[f32["h w 3"]] = dataclasses.field(default_factory=list)

  def append_images(
      self,
      disparity: f32["h w 1"],
      input_images: f32["h w 3"],
      opacity: f32["h w 1"],
      rgb: f32["h w 3"],
      rgb_foreground: f32["h w 0|3"],
      rgb_ground_truth: f32["h w 3"],
      semantic: f32["h w num_channels"],
      semantic_foreground: f32["h w 0|num_channels"],
      semantic_ground_truth: f32["h w 1"],
  ):
    """Append a set of images to the log."""
    self.rgb.append(rgb)
    if rgb_foreground.shape[-1] != 0:
      self.rgb_foreground.append(
          img_utils.apply_canvas(rgb_foreground, opacity))
    self.disparity.append(disparity)
    self.opacity.append(opacity)
    self.rgb_ground_truth.append(rgb_ground_truth)
    if input_images is not None:
      self.input_images.append(input_images)
    if semantic is not None and semantic.shape[-1]:
      semantic = jnp.argmax(semantic, axis=-1)
      self.semantic.append(
          utils.get_color_coded_semantics_image(semantics=semantic))
      self.semantic_ground_truth.append(
          utils.get_color_coded_semantics_image(
              semantics=jnp.squeeze(semantic_ground_truth, axis=-1)))
    if semantic_foreground is not None and semantic_foreground.shape[-1] != 0:
      semantic_foreground = jnp.argmax(semantic_foreground, axis=-1)
      semantic_foreground = utils.get_color_coded_semantics_image(
          semantics=semantic_foreground)
      # To allow proper compositing.
      semantic_foreground = semantic_foreground * opacity
      self.semantic_foreground.append(
          img_utils.apply_canvas(semantic_foreground, opacity))
    rgb_diff = jnp.clip(0.5 + rgb - rgb_ground_truth, 0, 1)
    self.rgb_diff.append(rgb_diff)


def _stack_input_images(images):
  """Horizontally stacks images from the -4'th axis."""
  images = jnp.split(images, images.shape[-4], axis=-4)
  images = jnp.concatenate(images, axis=-2)
  return jnp.squeeze(images, axis=-4)


# Rendering is forced to be deterministic even if training was randomized, as
# this eliminates "speckle" artifacts.
def _render_fn(
    variables,
    rays,
    model,
):
  """Renders a batch of rays.

  Args:
    variables: Model's variables
    rays: types.Rays. Each ndarray has leading dimension of length
      'batch_size_per_device'.
    model: Model to use for rendering.

  Returns:
    Nested collection of arrays, each of shape,
      [num_total_devices, batch_size_per_device, ...]
  """
  return jax.lax.all_gather(
      model.apply(
          variables,
          rays=rays,
          randomized_sampling=False,
          deterministic=True),
      axis_name="batch")


# Rendering is forced to be deterministic even if training was randomized, as
# this eliminates "speckle" artifacts.
def _predict_3d_fn(
    variables,
    points,
    model,
):
  """Renders a batch of rays.

  Args:
    variables: Model's variables
    points: Sample points for querying 3D semantics.
    model: Model to use for rendering.

  Returns:
    Nested collection of arrays, each of shape,
      [num_total_devices, batch_size_per_device, ...]
  """
  return jax.lax.all_gather(
      model.apply(
          variables,
          rays=None,
          randomized_sampling=False,
          deterministic=True,
          points=points),
      axis_name="batch")


def evaluate(
    params: nerf_config.ConfigParams,
):
  """Run evaluation until we hit the final checkpoint.

  Note the current version of evaluate calculates the confusion matrix
  for all images, sums them up before computing the mean and per-class IoU.

  Args:
    params: nerf_config.ConfigParams used for calling the eval.
  """
  max_steps = train_utils.get_max_steps(params)
  if not max_steps:
    return

  if params.train.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")

  logging.info("Using chunk of %s with %s devices.", params.evaluate.chunk,
               jax.device_count())

  # Load datasets.
  #
  # WARNING: It is CRITICAL that each host receive the same test set with the
  # same images in the same order as all other hosts in a multi-host
  # computation. If this is not the case, everything downstream will break.
  #
  test_dataset = datasets.get_dataset(
      split="test",
      args=params.datasets,
      model_args=params.models,
      example_type=datasets.ExampleType.IMAGE,
      is_novel_scenes=False)
  train_dataset = datasets.get_dataset(
      split="train",
      args=params.datasets,
      model_args=params.models,
      example_type=datasets.ExampleType.IMAGE,
      is_novel_scenes=False)

  # TODO(tutmann): wrap run specific params inside RunParams dataclass.
  # See epot's comment on cl/373799373.

  initialized_model = models.get_model(
      # We will initialize a model here, but overwrite that model from a
      # checkpoint further down.
      rng=jax3d.RandomState(0),
      args=params.models,
      num_scenes=train_utils.get_num_scenes(params),
      placeholder_batch=test_dataset.peek()[1],  # Batch from (DsState, Batch)
  )
  model = initialized_model.model
  optimizer = flax.optim.Adam(params.train.lr_init).create(
      initialized_model.variables)
  state = utils.TrainState(optimizer=optimizer)
  del initialized_model, optimizer

  # pmap over only over the rays.
  #
  # Args:
  #   variables: Model's variables. Copied to each device.
  #   rays: types.Rays. Each ndarray is of shape,
  #     [num_local_devices, batch_size_per_device, ...]
  #
  # Returns:
  #   result: Nested collection of arrays, each of shape,
  #     [num_local_devices, num_total_devices, batch_size_per_device, ...].
  #     Note that result[0] == ... == result[num_local_devices-1]
  render_pfn = jax.pmap(
      functools.partial(_render_fn, model=model),
      in_axes=(None, 0),
      donate_argnums=1,
      axis_name="batch",
  )

  predict_3d_pfn = jax.pmap(
      functools.partial(_predict_3d_fn, model=model),
      in_axes=(None, 0),
      donate_argnums=1,
      axis_name="batch",
  )

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(metrics.compute_ssim, max_val=1.), backend="cpu")

  last_step = 0

  # Root directory for all rendered images.
  root_out_dir = params.train.train_dir / "test_preds"

  # Log Gin config to disk.
  if jax.process_index() == 0:
    nerf_config.export_gin_config(root_out_dir / "eval.gin")
    gin_utils.log_params_to_disk(root_out_dir / "eval_params.py", params)

  # If evaluating a single checkpoint, we will not write TensorBoard metrics.
  if not params.evaluate.eval_once:
    test_summary_writer = jax_process_zero.SummaryWriter(
        params.train.train_dir / "eval_test")
    test_summary_writer.text(
        "params",
        # Use ``` for markdown.
        f"```python\n{params.to_gin_operative_repr()}\n```",  # pytype: disable=attribute-error
        step=0,
    )

    train_summary_writer = jax_process_zero.SummaryWriter(
        params.train.train_dir / "eval_train")
    train_summary_writer.text(
        "params",
        # Use ``` for markdown.
        f"```python\n{params.to_gin_operative_repr()}\n```",  # pytype: disable=attribute-error
        step=0,
    )
  else:
    test_summary_writer = None
    train_summary_writer = None

  while True:
    # Load latest checkpoint. Ignore unused dataset state.
    save_dir = train_utils.checkpoint_dir(params)
    state = train_utils.restore_opt_checkpoint(
        save_dir=save_dir,
        state=state)

    step = int(state.optimizer.state.step)

    # Only evaluate newer checkpoints.
    if step <= last_step:
      time.sleep(10)
      continue

    checkpoint_render_fn = functools.partial(
        utils.render_image,
        render_fn=functools.partial(render_pfn, state.optimizer.target),
        normalize_disp=params.evaluate.normalize_disp,
        chunk=params.evaluate.chunk)

    checkpoint_predict_3d_fn = functools.partial(
        utils.predict_3d_semanticnerf,
        render_pfn=functools.partial(predict_3d_pfn, state.optimizer.target),
        chunk=params.evaluate.chunk,
        num_semantic_classes=params.models.num_semantic_classes)

    shared_params = {
        "checkpoint_render_fn": checkpoint_render_fn,
        "checkpoint_predict_3d_fn": checkpoint_predict_3d_fn,
        "num_semantic_classes": params.models.num_semantic_classes,
        "ssim_fn": ssim_fn,
        "eval_once": params.evaluate.eval_once,
        "step": step,
        "save_output": params.evaluate.save_output,
        "root_out_dir": root_out_dir,
        "model_params": params.models,
        "train_state": state,
        "sigma_grid_dir": params.evaluate.sigma_grid_dir,
    }

    # Should sigma grid be written to disk for this checkpoint?
    save_sigma_to_disk = (int(step) == max_steps and
                          params.models.preserve_sigma_grid and
                          params.evaluate.sigma_grid_dir)

    evaluate_dataset(
        name="eval_test",
        dataset=test_dataset,
        num_images=params.evaluate.eval_num_images,
        num_log_images=params.evaluate.eval_num_log_images,
        summary_writer=test_summary_writer,
        save_sigma_grid=save_sigma_to_disk,
        **shared_params,
        )
    evaluate_dataset(
        name="eval_train",
        dataset=train_dataset,
        num_images=params.evaluate.eval_num_log_train_images,
        num_log_images=params.evaluate.eval_num_log_train_images,
        summary_writer=train_summary_writer,
        **shared_params,
        )

    if not params.evaluate.eval_once:
      # Log weight histograms only to train summary logs.
      opt_state = state.optimizer.state_dict()
      for var_name, weight in traverse_util.flatten_dict(opt_state).items():
        train_summary_writer.histogram("/".join(var_name), weight, step)
    else:
      break

    if int(step) >= max_steps:
      break
    last_step = step


def evaluate_dataset(
    name: str,
    dataset: datasets.DatasetIterable,
    checkpoint_render_fn: Callable[..., Any],
    checkpoint_predict_3d_fn: Callable[..., Any],
    num_images: int,
    num_log_images: int,
    num_semantic_classes: int,
    ssim_fn: Callable[..., Any],
    eval_once: bool,
    step: int,
    root_out_dir: jax3d.Path,
    save_output: bool,
    summary_writer: jax_process_zero.SummaryWriter,
    model_params: model_utils.ModelParams,
    train_state: utils.TrainState,
    save_sigma_grid: bool = False,
    sigma_grid_dir: Optional[jax3d.Path] = None,
    ):
  """Executes a single evaluation for a given dataset object."""
  out_dir = root_out_dir / name / f"step-{step:07d}"
  if save_output:
    out_dir.mkdir(parents=True, exist_ok=True)

  psnrs = []
  ssims = []
  confusion_matrix = jnp.zeros(2 * [num_semantic_classes])
  confusion_matrix_3d = jnp.zeros(2 * [num_semantic_classes])
  image_log = PredictionImageLog()
  accumulated_render_time_per_step_sec = 0.0
  accumulated_ray_count_per_step = 0

  dataset_iterator = iter(dataset)
  if num_images is not None:
    logging.info("Evaluating on a subset of the eval images: %d", num_images)
    dataset_iterator = itertools.islice(dataset_iterator, num_images)

  for idx, (_, batch) in tqdm.tqdm(
      enumerate(dataset_iterator), desc=f"{name} image"):
    batch, image_id = batch.pop_image_id_stateless()
    # Note: The following logic correctly ensures that the batch is
    # partitioned across all hosts in a multi-host evaluation setup. Only
    # (1 / jax.num_hosts()) percent of batch["rays"] will be processed by
    # the accelerators on this host.
    render_start_sec = time.time()
    prediction = checkpoint_render_fn(
        rays=batch.target_view.rays)
    accumulated_render_time_per_step_sec += time.time() - render_start_sec
    accumulated_ray_count_per_step += prediction.rgb[..., 0].size

    if jax.process_index() != 0:  # Only record via host 0.
      continue

    # Write sigma grid and variables to disk.
    if save_sigma_grid and idx == 0:
      logging.info("Saving sigma grid to disk at step=%d: %s",
                   step, sigma_grid_dir)

      # Select device=0, ray=0. This should be a scalar. It is assumed that
      # all rays share the same scene_idx.
      #
      # NOTE: scene_idx is NOT the same as scene_id. scene_idx is an integer
      # counting from 0. scene_id is just a string.
      scene_idx = int(batch.target_view.rays.scene_id[0, 0, 0])
      if not jnp.all(batch.target_view.rays.scene_id == scene_idx):
        unique_scene_ids = jnp.unique(batch.target_view.rays.scene_id)
        raise ValueError(
            f"Expected all rays to share the same scene_id. Found the "
            f"following instead: {unique_scene_ids}")
      nerf_state = serialization.NerfState(params=model_params,
                                           variables=train_state,
                                           density_grid=prediction.sigma_grid)
      scene_id = dataset.all_metadata[scene_idx].scene_name

      nerf_saver = serialization.NerfSaver(sigma_grid_dir)
      nerf_saver.save_state(scene_id, nerf_state)

      del scene_id, nerf_state, nerf_saver

    # Update image log to be reported to TensorBoard.
    # An image log example is one whose random index is smaller than
    # --eval_num_log_images.
    if not eval_once and idx < num_log_images:
      input_images = None
      image_log.append_images(
          disparity=prediction.disparity,
          input_images=input_images,
          opacity=prediction.opacity,
          rgb=prediction.rgb,
          rgb_foreground=prediction.foreground_rgb,
          rgb_ground_truth=batch.target_view.rgb,
          semantic=prediction.semantic,
          semantic_foreground=prediction.foreground_semantic,
          semantic_ground_truth=batch.target_view.semantics,
      )

    # Make 3D predictions.
    semantic_logits_3d = None
    if num_semantic_classes > 0:
      semantic_logits_3d = checkpoint_predict_3d_fn(
          sample_points=(
              eval_utils.create_sample_points_for_3d_semantics(
                  batch.target_view)),
      )

    if num_semantic_classes > 0:
      # Update 3D confusion matrix.
      curr_conf_mat_3d = utils.compute_conf_mat_from_preds(
          logits=semantic_logits_3d,
          labels=batch.target_view.point_cloud.semantics,
          num_classes=num_semantic_classes)
      confusion_matrix_3d = confusion_matrix_3d + curr_conf_mat_3d

    # Compute accuracy metrics.
    mse = ((prediction.rgb - batch.target_view.rgb)**2).mean()
    psnr = metrics.compute_psnr(mse)
    ssim = ssim_fn(prediction.rgb, batch.target_view.rgb)
    logging.info("PSNR = %.4f, SSIM = %.4f", psnr, ssim)
    psnrs.append(float(psnr))
    ssims.append(float(ssim))
    if num_semantic_classes > 0:
      curr_conf_mat = utils.compute_conf_mat_from_preds(
          logits=prediction.semantic,
          labels=batch.target_view.semantics,
          num_classes=num_semantic_classes)
      confusion_matrix = confusion_matrix + curr_conf_mat

    # Write rendered images to disk.
    if save_output:
      utils.save_img(prediction.rgb, out_dir / f"rgb_{image_id}.png")
      if num_semantic_classes:
        semantic_prediction = jnp.argmax(prediction.semantic, axis=-1)
        utils.save_img(semantic_prediction,
                       out_dir / f"segmentation_{image_id}.png")
      utils.save_img(prediction.disparity[..., 0],
                     out_dir / f"disp_{image_id}.png")
  del batch, prediction, idx  # pylint: disable=undefined-loop-variable

  iou = utils.compute_iou_from_con_mat(confusion_matrix)
  iou_3d = utils.compute_iou_from_con_mat(confusion_matrix_3d)

  # Write metrics and image logs to TensorBoard.
  if not eval_once:
    for key, value in image_log.items():
      if not value:
        continue
      images = jnp.stack(value, axis=0)
      summary_writer.image(
          f"[{name}] {key}",
          images,
          max_outputs=len(images),
          step=step)

    summary_writer.scalar("metrics/psnr", jnp.mean(jnp.array(psnrs)),
                          step)
    summary_writer.scalar("metrics/ssim", jnp.mean(jnp.array(ssims)),
                          step)
    if num_semantic_classes:
      summary_writer.scalar("metrics/mean_iou", iou.mean_iou, step)
      summary_writer.scalar("metrics/mean_iou_3d", iou_3d.mean_iou, step)
      logging.info("2D mIoU: %.4f 3D mIoU: %.4f", iou.mean_iou, iou_3d.mean_iou)
    summary_writer.scalar(
        "stats/rays_per_sec",
        accumulated_ray_count_per_step / accumulated_render_time_per_step_sec,
        step)
    # Log per-class IoU
    label_names = dataset.semantic_labels
    for i in range(num_semantic_classes):
      summary_writer.scalar(f"metrics_iou_class/{label_names[i]}",
                            iou.per_class_iou[i], step)

  # Write metrics to disk.
  if save_output and (jax.process_index() == 0):
    out_dir.joinpath("psnr.txt").write_text(str(jnp.mean(jnp.array(psnrs))))
    out_dir.joinpath("ssim.txt").write_text(str(jnp.mean(jnp.array(ssims))))
    out_dir.joinpath("mean_iou.txt").write_text(str(iou.mean_iou))
