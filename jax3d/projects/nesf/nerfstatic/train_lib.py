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

"""NeRF Training library."""

import functools
import gc
import time
from typing import Optional, Tuple

from absl import logging
from etils import etqdm as tqdm
import flax
from flax import linen as nn
import jax
import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic import datasets
from jax3d.projects.nesf.nerfstatic.losses import losses
from jax3d.projects.nesf.nerfstatic.metrics import metrics
from jax3d.projects.nesf.nerfstatic.models import models
from jax3d.projects.nesf.nerfstatic.nerf import utils
from jax3d.projects.nesf.nerfstatic.utils import config as nerf_config
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.nerfstatic.utils import jax_process_zero
from jax3d.projects.nesf.nerfstatic.utils import train_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import PRNGKey, f32  # pylint: disable=g-multiple-import
import numpy as np


def train_step(
    model: nn.Module,
    params: nerf_config.ConfigParams,
    rng: PRNGKey,
    state: utils.TrainState,
    batch: types.Batch,
    lr: float,
) -> train_utils.TrainStepOutput:
  """One optimization step.

  Args:
    model: The linen model.
    params: All Train, Loss, Dataset and Model parameters.
    rng: Random key.
    state: State of the model/optimizer.
    batch: Mini-batch of data for training.
    lr: Learning rate for decoders for this step.

  Returns:
    Updated train state and statistics.
  """

  def loss_fn(
      variables: flax.core.scope.FrozenVariableDict,
      rng: PRNGKey,
  ) -> Tuple[utils.ReconstructionStats, f32[""]]:
    rng_names = ["sampling", "dropout", "random_rotation"]
    rng, *rng_keys = jax.random.split(rng, len(rng_names) + 1)
    ret = model.apply(
        variables,
        rngs=dict(zip(rng_names, rng_keys)),
        rays=batch.target_view.rays,
        randomized_sampling=params.models.randomized,
        deterministic=False)[0]
    assert isinstance(ret, types.RenderResult)
    if (params.models.enable_mipnerf and
        params.models.ray_regularization_loss_mult):
      reconstruction_loss = losses.charbonnier_loss(
          ret.fine.rgb, batch.target_view.rgb[..., :3], epsilon=0.001)
    else:
      reconstruction_loss = losses.l2_loss(ret.fine.rgb,
                                           batch.target_view.rgb[..., :3])
    reconstruction_loss_term = types.LossTerm(
        loss=reconstruction_loss, weight=params.train.reconst_weight)
    psnr = metrics.compute_psnr(reconstruction_loss)
    # Compute cross entropy loss
    softmax_cross_entropy_loss = losses.softmax_cross_entropy_loss(
        logits=ret.fine.semantic, labels=batch.target_view.semantics,
        mask=batch.target_view.semantic_mask)
    # Compute the mean_iou
    mean_iou = utils.compute_iou_from_preds(
        logits=ret.fine.semantic,
        labels=batch.target_view.semantics,
        num_classes=params.models.num_semantic_classes).mean_iou
    semantic_loss_term = types.LossTerm(
        loss=softmax_cross_entropy_loss, weight=params.train.semantic_weight)
    fine_model_stats = types.ReconstructionModelStats(
        reconstruction_loss=reconstruction_loss_term,
        semantic_loss=semantic_loss_term,
        psnr=psnr, mean_iou=mean_iou)

    if ret.coarse is not None:
      # If there are both coarse and fine predictions, we compute the loss for
      # the coarse prediction (ret[0]) as well.
      if (params.models.enable_mipnerf and
          params.models.ray_regularization_loss_mult):
        reconstruction_loss_c = losses.charbonnier_loss(
            ret.coarse.rgb, batch.target_view.rgb[..., :3], epsilon=0.001)
      else:
        reconstruction_loss_c = losses.l2_loss(ret.coarse.rgb,
                                               batch.target_view.rgb[..., :3])
      psnr_c = metrics.compute_psnr(reconstruction_loss_c)
    else:
      reconstruction_loss_c = 0.
      psnr_c = 0.

    reconstruction_loss_c_term = types.LossTerm(
        loss=reconstruction_loss_c, weight=params.train.reconst_weight)
    coarse_model_stats = types.ReconstructionModelStats(
        reconstruction_loss=reconstruction_loss_c_term,
        psnr=psnr_c)

    if (params.train.weight_decay_mult > 0 and
        params.train.scenes_regularization_weight > 0):
      raise ValueError("Only one of weight_decay_mult and "
                       "scenes_regularization_weight can be > 0.")
    if params.train.scenes_regularization_weight > 0:
      regularization_term = losses.scenes_l1_regularization(variables)
      regularization_term = types.LossTerm(
          loss=regularization_term,
          weight=params.train.scenes_regularization_weight)
    else:
      regularization_weight_l2 = losses.l2_regularization(variables)
      regularization_term = types.LossTerm(
          loss=regularization_weight_l2, weight=params.train.weight_decay_mult)

    if (params.models.ray_regularization_loss_mult > 0 and
        not params.models.enable_mipnerf):
      raise ValueError("Ray regularization loss enabled only in mipnerf mode."
                       "Set enable_mipnerf=True.")
    if params.models.ray_regularization_loss_mult > 0:
      ray_regularization_loss = losses.ray_interval_regularization(
          z_values=ret.fine.ray_z_vals,
          weights=ret.fine.contribution)
      ray_regularization_term = types.LossTerm(
          loss=ray_regularization_loss,
          weight=params.models.ray_regularization_loss_mult)
    else:
      ray_regularization_term = types.LossTerm(loss=0., weight=0.)

    stats = utils.ReconstructionStats(
        coarse_model=coarse_model_stats,
        fine_model=fine_model_stats,
        regularization=regularization_term,
        ray_regularization=ray_regularization_term)
    return stats.total, stats

  (_, stats), grad = (
      jax.value_and_grad(loss_fn, has_aux=True)(state.optimizer.target, rng))
  del rng  # Do not use without splitting.
  grad = jax.lax.pmean(grad, axis_name="batch")
  stats = jax.lax.pmean(stats, axis_name="batch")

  # Clip gradients after synchronization.
  grad = train_utils.clip_values_by_global_norm(
      grad, max_norm=params.train.clip_grads_by_norm)

  new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
  new_state = state.replace(optimizer=new_optimizer)
  return train_utils.TrainStepOutput(train_state=new_state, stats=stats)


def train(
    params: nerf_config.ConfigParams,
    xm_wid: Optional[int] = None,
):
  """Train loop.

  Args:
    params: All training parameters (e.g. train, model, dataset, loss params).
    xm_wid: XManager worker id. Only needed when running under XManager.
  """
  max_steps = train_utils.get_max_steps(params)
  if not max_steps:
    return
  rng = jax3d.RandomState(params.train.random_seed)
  # Shift the numpy random seed by process_index() to shuffle data loaded by
  # different hosts.
  # TODO(tutmann): Remove numpy.
  np.random.seed(20201473 + jax.process_index())

  if params.train.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")

  gin_path = params.train.train_dir / "train.gin"
  param_path = params.train.train_dir / "train_params.py"

  save_dir = train_utils.checkpoint_dir(params)
  ds_state = train_utils.restore_ds_checkpoint_for_process(save_dir=save_dir)

  logging.info("Starting to initialize dataset.")
  dataset = datasets.get_dataset(
      split="train",
      args=params.datasets,
      model_args=params.models,
      example_type=datasets.ExampleType.RAY,
      ds_state=ds_state,
      is_novel_scenes=False,
  )
  logging.info("Finished initializing dataset.")

  _, placeholder_batch = dataset.peek()
  initialized_model = models.get_model(
      rng=rng.fork(),
      args=params.models,
      num_scenes=train_utils.get_num_scenes(params),
      placeholder_batch=placeholder_batch,
  )
  model = initialized_model.model

  summary_writer = jax_process_zero.SummaryWriter(
      params.train.train_dir / "train")

  summary_writer.text(
      "model_config",
      # Avoid markdown interpretation.
      f"<pre>{model}</pre>",
      step=0,
  )
  summary_writer.text(
      "params",
      # Use ``` for markdown.
      f"```python\n{params.to_gin_operative_repr()}\n```",  # pytype: disable=attribute-error
      step=0,
  )

  optimizer = flax.optim.Adam().create(initialized_model.variables)
  state = utils.TrainState(optimizer=optimizer)
  del initialized_model, optimizer

  learning_rate_fn = functools.partial(
      utils.learning_rate_decay,
      lr_init=params.train.lr_init,
      lr_final=params.train.lr_final,
      max_steps=max_steps,
      lr_delay_steps=params.train.lr_delay_steps,
      lr_delay_mult=params.train.lr_delay_mult)

  train_pstep = jax.pmap(
      functools.partial(train_step, model, params),
      axis_name="batch",
      in_axes=(0, 0, 0, None),
      donate_argnums=(0, 1, 2))

  # Log Gin config to disk.
  if jax.process_index() == 0:
    nerf_config.export_gin_config(gin_path)
    gin_utils.log_params_to_disk(param_path, params)

  state = train_utils.restore_opt_checkpoint(
      save_dir=save_dir,
      state=state)

  logging.info("restored_checkpoint: %s",
               train_utils.params_to_str(state.optimizer.target))

  # Resume training a the step of the last checkpoint.
  init_step = state.optimizer.state.step + 1

  # TODO(epot): Remove num_semantic_classes to only use semantic_labels.
  if dataset.semantic_labels and params.models.num_semantic_classes:
    assert len(dataset.semantic_labels) == params.models.num_semantic_classes, (
        f"{dataset.semantic_labels} vs. {params.models.num_semantic_classes}")

  state = flax.jax_utils.replicate(state)

  n_local_devices = jax.local_device_count()
  # Make random seed separate across hosts.
  rng.fold_in(jax.process_index())
  gc.disable()  # Disable automatic garbage collection for efficiency.

  logging.info("Starting train loop. Using the following devices: %s",
               jax.devices())
  t_loop_start = time.time()
  for step, batch in tqdm.tqdm(
      zip(range(init_step, max_steps + 1), dataset), desc="train step"):
    ds_state, batch = batch
    lr = learning_rate_fn(step)

    # Create a new independent RandomState for the current step
    rng_step = rng.fold_in_stateless(step)
    keys = rng_step.next(n_local_devices)  # For pmapping RNG keys.

    train_pstep_outputs = train_pstep(
        keys,
        state,
        batch,
        lr
    )
    del keys, state, batch
    state = train_pstep_outputs.train_state
    stats = train_pstep_outputs.stats
    del train_pstep_outputs

    if step % params.train.gc_every == 0:
      gc.collect()

    # Log training summaries. This is put behind a process_index check because
    # in multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if step % params.train.print_every == 0:
      summary_writer.scalar("loss/total", stats.total, step)
      summary_writer.scalar("loss/reconstruction",
                            stats.fine_model.reconstruction_loss.value, step)
      if params.models.num_semantic_classes:
        summary_writer.scalar("loss/semantic",
                              stats.fine_model.semantic_loss.value, step)
      summary_writer.scalar("loss/combined",
                            stats.fine_model.total, step)
      summary_writer.scalar("metrics/psnr",
                            np.mean(stats.fine_model.psnr), step)
      if params.models.num_semantic_classes:
        summary_writer.scalar("metrics/mean_iou",
                              np.mean(stats.fine_model.mean_iou), step)
      # When num_fine_samples is zero, we have no coarse model for 2-stage
      # sampling.
      if params.models.num_fine_samples:
        summary_writer.scalar("loss/reconstruction/coarse",
                              stats.coarse_model.reconstruction_loss.value,
                              step)
        summary_writer.scalar("loss/combined/coarse",
                              stats.coarse_model.total, step)
        summary_writer.scalar("metrics/psnr/coarse",
                              np.mean(stats.coarse_model.psnr), step)
      if params.train.scenes_regularization_weight > 0:
        summary_writer.scalar("regularization", stats.regularization.value,
                              step)
      summary_writer.scalar("learning_rate", lr, step)
      now = time.time()
      steps_per_sec = params.train.print_every / (now - t_loop_start)
      t_loop_start = now
      del now
      rays_per_sec = params.datasets.batch_size.total * steps_per_sec
      summary_writer.scalar("stats/steps_per_sec", steps_per_sec, step)
      summary_writer.scalar("stats/rays_per_sec", rays_per_sec, step)
    if step % params.train.save_every == 0:
      state_to_save = train_utils.de_replicate(state)
      train_utils.save_checkpoints_for_process(
          model_state=state_to_save,
          ds_state=ds_state,
          step=step,
          save_dir=save_dir,
          overwrite=params.train.overwrite_checkpoints,
      )

  if max_steps % params.train.save_every != 0:
    state_to_save = train_utils.de_replicate(state)
    train_utils.save_checkpoints_for_process(
        model_state=state_to_save,
        ds_state=ds_state,
        step=max_steps,
        save_dir=save_dir,
        overwrite=params.train.overwrite_checkpoints,
    )
