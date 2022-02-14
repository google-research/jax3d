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

"""Utilities for Training."""

import enum
import os
from typing import List, Optional, Union

from absl import logging
import chex
import flax
from flax import serialization
from flax import traverse_util
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic.datasets import dataset
from jax3d.projects.nesf.nerfstatic.nerf import utils
from jax3d.projects.nesf.nerfstatic.utils import config as nerf_config
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import PRNGKey, f32, i32  # pylint: disable=g-multiple-import
import numpy as np


# Prefixes for checkpoints written to disk. CKPT_PREFIX_OPT is for optimizer
# state, which is the same across all hosts. CKPT_PREFIX_DS is for dataset
# state, which is different per device.
_CKPT_PREFIX_OPT = "checkpoint_opt_"
_CKPT_PREFIX_DS = "checkpoint_ds{process_id:03d}_"

# How many checkpoints to keep.
_KEEP = 10
_KEEP_EVERY_N_STEPS = 100000


@chex.dataclass
class TrainStepOutput:
  """Output of train_step()."""

  # The updated training state after one training step.
  train_state: utils.TrainState

  stats: Union[utils.ReconstructionStats, utils.SemanticStats]


def save_checkpoints_for_process(*,
                                 model_state: utils.TrainState,
                                 ds_state: Optional[dataset.DsState],
                                 step: int,
                                 save_dir: jax3d.Path,
                                 # TODO(b/198251059): Remove this once fixed.
                                 overwrite: bool = False,
                                 ) -> None:
  """Save checkpoints for both model state and dataset iterator state.

  For the host 0, write out the model state to a checkpoint in checkpoint dir
  with prefix CKPT_PREFIX_OPT. For all hosts, write out the respective
  dataset iterator state (data_state) with prefix
  f"{CKPT_PREFIX_STATE}{jax.process_index()}_".

  Args:
    model_state: Dataclass object, with stored optimizer state.
    ds_state: Dataset state, as yielded by the `DatasetIterable`.
    step: Training step.
    save_dir: Directory to save checkpoints.
    overwrite: Whether to overwrite existing checkpoints. If this flag is set to
      False, an error is thrown if one attempts to write a checkpoint that is
      not strictly newer than the last one (i.e., if the checkpoint to be
      written is either overriding an existing one, or if a newer checkpoint
      already exists in the directory).

  """

  if jax.process_index() == 0:  # Can't use jax_process_zero (circular import).
    checkpoints.save_checkpoint(
        save_dir, model_state, int(step),
        keep=_KEEP, keep_every_n_steps=_KEEP_EVERY_N_STEPS,
        prefix=_CKPT_PREFIX_OPT, overwrite=overwrite)

  checkpoints.save_checkpoint(
      save_dir,
      # Original ds state is 128-bits which is not supported by serialization
      # so we normalize to `bytes` it here.
      # Note: If DsState was already normalized previously, this is a no-op.
      target=dataset.to_ds_state_bytes(ds_state),
      step=int(step),
      keep=_KEEP, keep_every_n_steps=_KEEP_EVERY_N_STEPS,
      prefix=_CKPT_PREFIX_DS.format(process_id=jax.process_index()),
      overwrite=overwrite,
  )


def restore_ds_checkpoint_for_process(
    *,
    save_dir: jax3d.Path,
)-> Optional[dataset.DsState]:
  """Restore checkpoint for dataset iterator state according to process id.

  Args:
    save_dir: Directory to save checkpoints.

  Returns:
    Dataset iterator state.

  """
  ds_state_ckpt = checkpoints.restore_checkpoint(
      os.fspath(save_dir),
      target=None,
      prefix=_CKPT_PREFIX_DS.format(process_id=jax.process_index()),
  )

  # Handle case where a ckpt was found.
  if ds_state_ckpt:
    # Restore the shape of the ds_state
    target = dataset.to_ds_state_bytes(np.random.PCG64().state["state"])
    ds_state_ckpt = serialization.from_state_dict(
        target=target,
        state=ds_state_ckpt,
    )

    # Restore the ds state values
    ds_state_ckpt = dataset.to_ds_state_int(ds_state_ckpt)
  return ds_state_ckpt


def restore_opt_checkpoint(*,
                           save_dir: jax3d.Path,
                           state: utils.TrainState,
                           ) -> utils.TrainState:
  """Restore checkpoint for model optimizer state.

  Args:
    save_dir: Directory to save checkpoints.
    state: Dataclass object, with stored optimizer state.

  Returns:
    Restored model state.

  """
  state = checkpoints.restore_checkpoint(
      os.fspath(save_dir), state, prefix=_CKPT_PREFIX_OPT)
  return state


def de_replicate(state):
  """Undo flax.jax_utils.replicate()."""

  def first(x):
    assert x.shape, x.shape
    assert x.shape[0], x.shape
    return x[0]

  return jax.device_get(jax.tree_map(first, state))


class TrainMode(enum.Enum):
  """Mode of training."""
  TRAIN = enum.auto()  # Optimize scene params AND decoders.
  SEMANTIC = enum.auto()  # Optimize semantic params only


def checkpoint_dir(
    params: nerf_config.ConfigParams) -> jax3d.Path:
  """Return the 'train' directory."""
  d = jax3d.Path(params.train.train_dir)
  return d / "train_checkpoints"


def get_max_steps(params: nerf_config.ConfigParams):
  """Return the train steps."""
  return params.train.train_steps


def get_num_scenes(params: nerf_config.ConfigParams):
  """Returns the number of scenes."""
  scenes = params.datasets.train_scenes
  num = dataset.num_scenes(scenes)
  logging.info("get_num_scenes: %d", num)
  return num


def params_to_str(params: flax.core.FrozenDict, filter_fn=None) -> str:
  """Match params with filter_fn and return them as loggable string."""
  params = params.unfreeze()
  flat_dict = traverse_util.flatten_dict(params)
  result = []
  for key, value in flat_dict.items():
    path = "/" + "/".join(key)
    if filter_fn is None or filter_fn(path, value):
      result.append(f"{path} {value.shape}")
  return "\n".join(result)


def _l2_norm(tree):
  """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
  leaves, _ = jax.tree_flatten(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def clip_values_by_global_norm(pytree, max_norm: Optional[f32]):
  """Clips values in a pytree by global L2 norm."""
  if max_norm is None:
    return pytree
  norm = _l2_norm(pytree)
  normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
  return jax.tree_map(normalize, pytree)


def should_execute_now(step: int, step_every: Optional[int]) -> bool:
  """Returns true every step_every steps."""
  if not step_every:
    return False
  return step % step_every == 0


def assert_num_semantic_classes(expected_num_semantic_labels: int,
                                semantic_labels: List[str]):
  # If expected_num_semantic_labels isn't set, then training will ignore
  if not expected_num_semantic_labels:
    return

  # Allow user to specify MORE semantic classes than the dataset actually has,
  # but not fewer. This makes end2end_test.py a little easier to write.
  if len(semantic_labels) > expected_num_semantic_labels:
    raise ValueError(
        f"Expected {expected_num_semantic_labels} semantic labels, but found "
        f"{len(semantic_labels)}: {semantic_labels}")


def log_rays_per_scene_counts(batch_scene_id):
  unique, counts = jnp.unique(batch_scene_id, return_counts=True)
  lines = ["Number of rays allocated to each scene in this batch:"]
  for value, count in zip(unique, counts):
    lines.append(f"\t {value:03d} | {count:>8}")
  logging.info("\n".join(lines))


def log_rays_per_semantic_class_counts(semantics: i32["..."],
                                       labels: List[str]):
  """Log the number of rays assigned to each semantic class."""
  unique, counts = jnp.unique(semantics, return_counts=True)

  # Convert to basic Python types.
  unique = unique.tolist()
  counts = counts.tolist()

  # Construct an int-to-string mapping. If labels is empty, then create a fake
  # label for every class.
  if labels:
    labels = {i: name for (i, name) in enumerate(labels)}
  else:
    labels = {i: f"class{i}" for i in unique}

  # Validate int-to-string mapping.
  for value in unique:
    if value not in labels:
      raise ValueError(
          f"Unable to map semantic class ids to names. \n"
          f"    semantic_class_ids={unique}\n"
          f"    labels={labels}")

  lines = ["Number of rays allocated to each semantic class in this batch:"]
  for value, count in zip(unique, counts):
    label = labels[value]
    lines.append(f"\t {value:03d} {label:<8} | {count:>8}")
  logging.info("\n".join(lines))


def log_loss_miou_per_scene(scene_ids, stats, summary_writer, step):
  """Calculate per scene loss and miou for all scenes in batch."""

  all_losses = stats.semantic_model.semantic_loss.loss
  all_mious = stats.semantic_model.mean_iou

  # Get unique ids
  unique_scene_ids = jnp.unique(scene_ids)
  # For each id, calculate the avg loss & avg mIOU
  for scene_id in unique_scene_ids:
    matched_ids = jnp.where(scene_ids == scene_id, 1, 0)
    counts = jnp.sum(matched_ids)
    # For scene id, calculate the avg loss
    total_scene_loss = jnp.sum(all_losses*matched_ids)
    scene_loss = total_scene_loss / counts
    # For scene id, calculate the avg mIOU
    total_scene_miou = jnp.sum(all_mious*matched_ids)
    scene_miou = total_scene_miou / counts

    summary_writer.scalar(
        f"scene_metrics/loss_scene_{scene_id}",
        scene_loss,
        step)
    summary_writer.scalar(
        f"scene_metrics/miou_scene_{scene_id}",
        scene_miou,
        step)


def create_sample_points_for_smoothness_regularization(
    scene_id: i32[""],
    num_points: int,
    stddev: float,
    rng: PRNGKey,
) -> types.SamplePoints:
  """Constructs sample points fro smoothness regulariztion.

  Args:
    scene_id: identifier for which scene to use.
    num_points: number of 3D points to sample
    stddev: standard deviation for normally-distributed noise when perturbing
      points.
    rng: random number generator seed.

  Returns:
    result: sample points to calculate semantic logits at. The 'points' field is
      guaranteed to have shape [2, num_points, 3]. result[0] contains the
      original points; result[1] contains perturbed points.
  """
  rng_original, rng_perturbations = jax.random.split(rng, 2)

  # Construct original points inside the bounding box.
  original_points = jax.random.uniform(rng_original, (2, num_points, 3),
                                       minval=-1., maxval=1.)

  # Apply normally-sampled perturbations.
  perturbations = jax.random.normal(rng_perturbations, (2, num_points, 3))
  perturbed_points = original_points + stddev * perturbations

  # Ensure perturbed points are still in the [-1, 1]^3 box.
  perturbed_points = jnp.maximum(perturbed_points, -1.)
  perturbed_points = jnp.minimum(perturbed_points, 1.)

  scene_id = jnp.full((2, 1), scene_id, jnp.int32)
  dummy_direction = jnp.zeros((2, 3), jnp.float32)

  return types.SamplePoints(scene_id=scene_id,
                            position=perturbed_points,
                            direction=dummy_direction)
