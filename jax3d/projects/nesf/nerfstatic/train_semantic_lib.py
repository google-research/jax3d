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

"""Semantic Training library."""

import functools
from typing import Optional, Tuple

from absl import logging
import chex
import flax
import jax
from jax import numpy as jnp

import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic import datasets
from jax3d.projects.nesf.nerfstatic.losses import losses
from jax3d.projects.nesf.nerfstatic.models import models
from jax3d.projects.nesf.nerfstatic.models import volumetric_semantic_model
from jax3d.projects.nesf.nerfstatic.nerf import utils
from jax3d.projects.nesf.nerfstatic.utils import config as nerf_config
from jax3d.projects.nesf.nerfstatic.utils import jax_process_zero
from jax3d.projects.nesf.nerfstatic.utils import semantic_utils
from jax3d.projects.nesf.nerfstatic.utils import train_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import PRNGKey, Tree, f32, i32  # pylint: disable=g-multiple-import


def train(
    params: nerf_config.ConfigParams,
    xm_wid: Optional[int] = None,
):
  """Semantic train loop.

  Args:
    params: All training parameters (e.g. train, model, dataset, loss params).
    xm_wid: XManager worker id. Only needed when running under XManager.
  """

  del xm_wid

  rng = jax3d.RandomState(params.train.random_seed)

  # Initialize dataset. Ensure that the model and the dataaset agree on the
  # number of semantic classes. It is assumed that all scenes share the same
  # set of semantic classes.
  logging.info("Loading dataset.")
  dataset, placeholder_batch = _construct_dataset(params)
  train_utils.assert_num_semantic_classes(
      params.models.num_semantic_classes, dataset.all_metadata[0].labels)

  # Initialize & load per-scene NeRF models.
  logging.info("Initializing pretrained NeRF models.")
  recovered_nerf_state = semantic_utils.load_all_nerf_variables(
      save_dir=params.train.nerf_model_ckpt,
      train_dataset=dataset,
      novel_dataset=None,
      recompute_sigma_grid_opts=(
          semantic_utils.RecomputeSigmaGridOptions.from_params(params.train)))

  logging.info("nerf scene_ids: %s", recovered_nerf_state.train_scene_ids)

  # Initialize semantic model.
  logging.info("Initializing semantic model.")
  initialized_vol_sem_model = models.construct_volumetric_semantic_model(
      rng=rng,
      num_scenes=-1,
      placeholder_batch=placeholder_batch,
      args=params.models,
      nerf_model=recovered_nerf_state.model,
      nerf_sigma_grid=recovered_nerf_state.train_sigma_grids[0],
      nerf_variables=recovered_nerf_state.train_variables[0])
  vol_sem_model = initialized_vol_sem_model.model
  semantic_variables = initialized_vol_sem_model.variables
  del initialized_vol_sem_model

  # Write HParams to disk.
  summary_writer = jax_process_zero.SummaryWriter(params.train.train_dir /
                                                  "train")
  summary_writer.text(
      "model_config",
      # Avoid markdown interpretation.
      f"<pre>{vol_sem_model}</pre>",
      step=0,
  )
  summary_writer.text(
      "params",
      # Use ``` for markdown.
      f"```python\n{params.to_gin_operative_repr()}\n```",  # pytype: disable=attribute-error
      step=0,
  )

  # Initialize optimizer.
  optimizer = flax.optim.Adam().create(semantic_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer

  # Function for generating learning rate.
  learning_rate_fn = functools.partial(
      utils.learning_rate_decay,
      lr_init=params.train.lr_init,
      lr_final=params.train.lr_final,
      max_steps=train_utils.get_max_steps(params),
      lr_delay_steps=params.train.lr_delay_steps,
      lr_delay_mult=params.train.lr_delay_mult)

  # Restore model checkpoint from disk.
  logging.info("Restoring optimizer state.")
  checkpoint_dir = train_utils.checkpoint_dir(params)
  state = train_utils.restore_opt_checkpoint(
      save_dir=checkpoint_dir, state=state)
  start_step = state.optimizer.state.step

  # Replicate state across local devices.
  state = flax.jax_utils.replicate(state)

  # Build distributed computation.
  logging.info("Compiling train_pstep.")
  train_pstep = jax.pmap(
      functools.partial(_apply_train_step, vol_sem_model, params),
      axis_name="batch",
      in_axes=(0, 0, 0, 0, 0, None, 0),
  )

  logging.info("Starting training now.")
  for step, (_, batch) in zip(
      range(start_step, params.train.train_steps+1), iter(dataset)):
    # Given a device (axis=0) and scene (axis=1), all values for
    # property "scene_id" are identical. Choose the first one.
    batch_scene_ids = jax.tree_map(lambda x: x[:, :, 0, 0],
                                   batch.target_view.rays.scene_id)
    batch_scene_ids = batch_scene_ids.flatten()

    # Select NeRF variables corresponding to chosen scenes.
    select_nerf_variables = semantic_utils.select_and_stack(
        batch_scene_ids, recovered_nerf_state.train_variables)
    select_nerf_sigma_grids = semantic_utils.select_and_stack(
        batch_scene_ids, recovered_nerf_state.train_sigma_grids)
    select_nerf_scene_ids = semantic_utils.select_and_stack(
        batch_scene_ids, recovered_nerf_state.train_scene_ids)

    # Update random state.
    keys = rng.next(jax.local_device_count())

    train_pstep_outputs = train_pstep(keys, state, select_nerf_variables,
                                      select_nerf_sigma_grids,
                                      batch.target_view, learning_rate_fn(step),
                                      select_nerf_scene_ids)
    state = train_pstep_outputs.train_state
    stats = train_pstep_outputs.stats

    # Write model checkpoint to disk. Model state is "de-replicated" before
    # writing to ensure that only one copy of each variable is stored on disk.
    if train_utils.should_execute_now(step, params.train.save_every):
      logging.info("Writing checkpoint to disk.")
      state_to_save = train_utils.de_replicate(state)
      train_utils.save_checkpoints_for_process(
          model_state=state_to_save,
          ds_state=None,
          step=step,
          save_dir=checkpoint_dir,
          overwrite=params.train.overwrite_checkpoints)

    # All quantities have shape [num_local_devices, num_scenes_per_batch].
    # Logic below averages them together.
    if train_utils.should_execute_now(step, params.train.print_every):
      logging.info("Step: %d", step)
      logging.info("Processing unique scene_ids: %s",
                   jnp.unique(batch_scene_ids))
      logging.info("Semantic Loss: %s",
                   stats.semantic_model.semantic_loss.loss)
      logging.info("Smoothness Regularization Loss: %s",
                   stats.semantic_model.smoothness_regularization_loss.loss)
      logging.info("mIOU: %s", stats.semantic_model.mean_iou)
      logging.info("batch scene ids: %s", batch_scene_ids)
      logging.info("select nerf scene ids: %s", select_nerf_scene_ids)
      logging.info(
          ("Percentage of rays where batch's scene_id matches NeRF's scene_id: "
           "%s"),
          stats.semantic_model.percent_matching_scene_ids)
      train_utils.log_rays_per_scene_counts(batch.target_view.rays.scene_id)
      train_utils.log_rays_per_semantic_class_counts(
          batch.target_view.semantics, dataset.semantic_labels)
      if params.train.log_per_scene:
        train_utils.log_loss_miou_per_scene(
            select_nerf_scene_ids, stats, summary_writer, step)

      summary_writer.scalar(
          "loss/total",
          stats.semantic_model.total,
          step)
      summary_writer.scalar(
          "loss/semantic",
          jnp.mean(stats.semantic_model.semantic_loss.loss),
          step)
      summary_writer.scalar(
          "loss/smoothness_regularization",
          jnp.mean(stats.semantic_model.smoothness_regularization_loss.loss),
          step)
      summary_writer.scalar(
          "metrics/mean_iou_2d",
          jnp.mean(stats.semantic_model.mean_iou),
          step)
      summary_writer.scalar(
          "debug/percent_matching_scene_ids",
          jnp.mean(stats.semantic_model.percent_matching_scene_ids),
          step)


################################################################################
# Internal helper functions.
################################################################################


def _construct_dataset(
    params: nerf_config.ConfigParams,
) -> Tuple[datasets.DatasetIterable, types.Batch]:
  """Constructs a dataset and placeholder batch.

  Args:
    params: Hyperparameters.

  Returns:
    dataset: Dataset itself.
    placeholder_batch: Placeholder batch used for initializing a
      VolumetricSemanticModel.
  """

  dataset = datasets.get_dataset(
      split="train",
      args=params.datasets,
      model_args=params.models,
      example_type=datasets.ExampleType.RAY,
      ds_state=None,
      is_novel_scenes=False,
  )

  # Select device 0, scene 0 from the placeholder batch. Later code will
  # use pmap and vmap, respectively, across these two axes.
  _, placeholder_batch = dataset.peek()
  placeholder_batch = jax.tree_map(lambda t: t[0, 0, ...], placeholder_batch)

  return dataset, placeholder_batch


def _apply_forward_pass_single_scene(
    rng: PRNGKey,
    params: nerf_config.ConfigParams,
    nerf_variables: Tree[jnp.ndarray],
    nerf_sigma_grid: f32["x y z c"],
    semantic_model: volumetric_semantic_model.VolumetricSemanticModel,
    semantic_variables: Tree[jnp.ndarray],
    batch: types.Views,
    nerf_scene_id: i32[""]) -> types.SemanticModelStats:
  """Calculate the loss for all rays corresponding to a single scene.

  Args:
    rng: Random key.
    params: All Train, Loss, Dataset and Model parameters.
    nerf_variables: Variables for NerfMLP describing a single scene.
    nerf_sigma_grid: Density grid corresponding to the same scene.
    semantic_model: Architecture of the SemanticModel we want to learn.
    semantic_variables: Variables corresponding to semantic_model.
    batch: Batch of rays from the same scene.
    nerf_scene_id: Scene id of current scene's nerf variables.

  Returns:
    types.SemanticModelStats containing the metrics and losses.
  """

  rng_names = ["params", "sampling", "data_augmentation"]
  rng, rng_smoothness, *rng_keys = jax.random.split(rng, len(rng_names) + 2)

  is_smoothness_regularization_enabled = (
      params.train.semantic_smoothness_regularization_num_points_per_device > 0)

  # Construct points for smoothness regularization.
  smoothness_regularization_sample_points = None
  if is_smoothness_regularization_enabled:
    smoothness_regularization_sample_points = (
        train_utils.create_sample_points_for_smoothness_regularization(
            scene_id=nerf_scene_id,
            num_points=params.train.semantic_smoothness_regularization_num_points_per_device,  # pylint: disable=line-too-long
            stddev=params.train.semantic_smoothness_regularization_stddev,
            rng=rng_smoothness))

  # Apply model.
  result = semantic_model.apply(
      semantic_variables,
      rngs=dict(zip(rng_names, rng_keys)),
      rays=batch.rays,
      sigma_grid=nerf_sigma_grid,
      randomized_sampling=True,
      is_train=True,
      nerf_model_weights=nerf_variables,
      points=smoothness_regularization_sample_points)

  if is_smoothness_regularization_enabled:
    rendered_rays, semantic_3d_predictions = result
  else:
    rendered_rays, semantic_3d_predictions = result, None
  assert isinstance(rendered_rays, types.RenderedRays)

  # Compute cross entropy loss
  softmax_cross_entropy_loss = losses.softmax_cross_entropy_loss(
      logits=rendered_rays.semantic, labels=batch["semantics"])

  semantic_loss_term = types.LossTerm(
      loss=softmax_cross_entropy_loss, weight=params.train.semantic_weight)

  # Compute the mean_iou
  mean_iou = utils.compute_iou_from_preds(
      logits=rendered_rays.semantic,
      labels=batch["semantics"],
      num_classes=params.models.num_semantic_classes).mean_iou

  # Compute smoothness regularization
  if is_smoothness_regularization_enabled:
    chex.assert_shape(
        semantic_3d_predictions,
        (2,
         params.train.semantic_smoothness_regularization_num_points_per_device,
         params.models.num_semantic_classes))
    smoothness_regularization_loss = losses.l1_smoothness_regularization(
        semantic_3d_predictions[0],  # activations at original points
        semantic_3d_predictions[1])  # activations at perturbed points
  else:
    smoothness_regularization_loss = jnp.array(0.0)

  smoothness_regularization_term = types.LossTerm(
      loss=smoothness_regularization_loss,
      weight=params.train.semantic_smoothness_regularization_weight)

  # Ensure that NeRF and batch agree on scene_id. This quantity should always
  # be 1.0.
  percent_matching_scene_ids = jnp.mean(
      jnp.equal(batch.rays.scene_id, nerf_scene_id))

  return types.SemanticModelStats(
      semantic_loss=semantic_loss_term,
      smoothness_regularization_loss=smoothness_regularization_term,
      mean_iou=mean_iou,
      percent_matching_scene_ids=percent_matching_scene_ids)


# Use vmap to map over scenes.
_apply_forward_pass_per_device = jax.vmap(
    _apply_forward_pass_single_scene, in_axes=(None, None, 0, 0,
                                               None, None, 0, 0))


def _apply_train_step(
    semantic_model: volumetric_semantic_model.VolumetricSemanticModel,
    params: nerf_config.ConfigParams,
    rng: jax3d.RandomState,
    state: utils.TrainState,
    select_nerf_variables: Tree[jnp.ndarray],
    select_nerf_sigma_grids: f32["n x y z c"],
    select_batch: types.Views,
    learning_rate: float,
    select_nerf_scene_ids: i32["n"],
) -> train_utils.TrainStepOutput:
  """Apply a single training step.

  Update semantic_variables.

  Args:
    semantic_model: Semantic model architecture.
    params: All Train, Loss, Dataset and Model parameters.
    rng: Random seed.
    state: State of the model/optimizer.
    select_nerf_variables: Variables for NerfMLPs.
    select_nerf_sigma_grids: Density grids derived from NerfMLPs.
    select_batch: Camera rays.
    learning_rate: Learning rate for decoders for this step.
    select_nerf_scene_ids: Scene ids for this device.

  Returns:
    Updated train state and statistics.
  """

  def loss_fn(semantic_variables):
    # semantic_model_stats will have arrays of shape
    # [num_scenes_per_device, ...] thanks to vmap().
    semantic_model_stats = _apply_forward_pass_per_device(
        rng, params, select_nerf_variables, select_nerf_sigma_grids,
        semantic_model, semantic_variables, select_batch, select_nerf_scene_ids)

    # L2 regularization. This quantity is the same across all devices.
    regularization_weight_l2 = losses.l2_regularization(semantic_variables)
    assert not regularization_weight_l2.shape, regularization_weight_l2.shape
    regularization_term = types.LossTerm(
        loss=regularization_weight_l2, weight=params.train.weight_decay_mult)

    stats = utils.SemanticStats(semantic_model=semantic_model_stats,
                                regularization=regularization_term)

    # Average loss across all scenes on this device.
    loss = stats.total
    assert not loss.shape, loss.shape

    return loss, stats

  (_, stats), grads = jax.value_and_grad(
      loss_fn, has_aux=True)(state.optimizer.target)
  grads = jax.lax.pmean(grads, axis_name="batch")
  new_optimizer = state.optimizer.apply_gradient(
      grads, learning_rate=learning_rate)
  new_state = state.replace(optimizer=new_optimizer)
  return train_utils.TrainStepOutput(train_state=new_state, stats=stats)


