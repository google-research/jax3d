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

"""Trainer for GLO NeRF Model."""

import dataclasses
import functools
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from etils.array_types import PRNGKey
import flax
from flax import jax_utils
from flax.core import unfreeze
from flax.metrics import tensorboard
import gin
import jax
import jax.numpy as jnp
from jax3d.projects.generative.nerf import trainer
from jax3d.projects.generative.nerf.glo_nerf import eval as evaluation
from jax3d.projects.generative.nerf.glo_nerf import loss
from jax3d.projects.generative.nerf.glo_nerf import models
import numpy as np
import optax


@flax.struct.dataclass
class TransformerNeRFTrainState(trainer.TrainState):
  """Training state for the NeRF Transformer Model."""
  model_parameters: models.ModelParameters
  latent_table: np.ndarray
  optimizer_state: optax.OptState
  rng: PRNGKey

  def to_serializable(self) -> "TransformerNeRFTrainState":
    """Transforms the state values into a form suitable for serialization."""
    return self.replace(
        model_parameters=jax_utils.unreplicate(self.model_parameters),
        optimizer_state=jax_utils.unreplicate(self.optimizer_state),
    )

  def from_serializable(self) -> "TransformerNeRFTrainState":
    """Transforms deserialized values into a form suitable for training."""
    state = self
    state = state.replace(
        latent_table=np.copy(state.latent_table),
        model_parameters=jax_utils.replicate(state.model_parameters),
        optimizer_state=jax_utils.replicate(state.optimizer_state),
    )
    return state


@gin.configurable()
@dataclasses.dataclass(frozen=True)
class TransformerNeRFTrainer(trainer.Trainer):
  """Trainer implementation for the NeRF Transformer Model."""
  learning_rate_start: float = 5e-4
  learning_rate_end: float = 1e-4
  latent_learning_rate_start: float = 5e-3
  latent_learning_rate_end: float = 5e-3

  num_latent_tokens: int = 128
  latent_token_dim: int = 128

  summary_identities: int = 8

  dataset_reader_class: Any = gin.REQUIRED

  @property
  @functools.lru_cache()
  def model(self) -> models.Model:
    """An instance of the model class."""
    return models.Model()

  @property
  def optimizer(self) -> optax.GradientTransformation:
    network_weight_lr_schedule = optax.exponential_decay(
        self.learning_rate_start, self.max_steps,
        self.learning_rate_end / self.learning_rate_start)
    return optax.adam(network_weight_lr_schedule)

  @property
  @functools.lru_cache()
  def data_reader(self) -> "dataset_reader_class":
    return self.dataset_reader_class()

  def init_state(self) -> TransformerNeRFTrainState:
    """Initializes training state."""
    latent_table = np.zeros((self.data_reader.identity_count,
                             self.num_latent_tokens, self.latent_token_dim))

    rng = jax.random.PRNGKey(self.random_seed)
    train_rng, model_init_rng = jax.random.split(rng)
    model_parameters = self.model.initialize_parameters(model_init_rng,
                                                        self.num_latent_tokens,
                                                        self.latent_token_dim)
    model_parameters = unfreeze(model_parameters)

    optimizer_state = self.optimizer.init(model_parameters)

    train_state = TransformerNeRFTrainState(  # pylint: disable=unexpected-keyword-arg
        step=0,
        model_parameters=model_parameters,
        latent_table=latent_table,
        optimizer_state=optimizer_state,
        rng=train_rng,
    )
    return train_state

  def init_data_loader(self) -> Iterable[Any]:
    """Initializes dataset loaders."""
    return iter(self.data_reader)

  def build_train_step(self) -> Callable[[Any], Any]:
    """Build the JIT-ed parallel train step function."""

    def per_device_train_step(model_parameters, latents, optimizer_state, data,
                              rng, step):
      """The train step logic for a single device."""

      def _loss_fn(params, data, rng, step):
        model_parameters, latents = params
        inputs = models.ModelInputs(latent_tokens=latents)
        return loss.transformer_nerf_loss_fn(model_parameters, inputs, data,
                                             rng, step)

      latent_ids, latent_vals = latents

      grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
      (total_loss, loss_terms), grad = grad_fn((model_parameters, latent_vals),
                                               data, rng, step)
      loss_terms["Total"] = total_loss

      model_grad, latent_grad = grad
      model_grad = jax.lax.pmean(model_grad, axis_name="replicas")
      updates, optimizer_state = self.optimizer.update(model_grad,
                                                       optimizer_state,
                                                       model_parameters)
      model_parameters = optax.apply_updates(model_parameters, updates)

      latent_updates = (latent_ids, latent_grad)
      latent_updates = jax.lax.all_gather(latent_updates, axis_name="replicas")

      return model_parameters, optimizer_state, latent_updates, loss_terms

    return jax.pmap(per_device_train_step, "replicas")

  def train_step(
      self, train_state: TransformerNeRFTrainState, inputs: Dict[str, Any],
      scratch: Optional[Dict[str, Any]]
  ) -> Tuple[TransformerNeRFTrainState, Dict[str, Any], Dict[str, Any]]:
    """Performs a single training step.

    Args:
      train_state: A state object tracking the current progress of training.
      inputs: Some data containing the input to the training step.
      scratch: Optional scratch state object for re-using values between steps.

    Returns:
      train_state: The updated state after the effect of the step is applied.
      summary_data: Optional data to be passed later to 'write_summaries'.
      scratch: Updated scratch state object.
    """
    if scratch is None:
      scratch = {"parallel_train_step": self.build_train_step()}
    batch_indices = inputs["identity_index"]
    batch_latents = train_state.latent_table[batch_indices]
    latents = (batch_indices, batch_latents)

    next_rng, current_rng = jax.random.split(train_state.rng)
    current_rngs = jax.random.split(current_rng, jax.local_device_count())

    (new_model_parameters, new_optimizer_state, latent_updates,
     loss_terms) = scratch["parallel_train_step"](
         train_state.model_parameters, latents, train_state.optimizer_state,
         inputs, current_rngs, jax_utils.replicate(train_state.step))  # pytype: disable=wrong-arg-count  # trace-all-classes

    loss_terms = jax.tree_map(np.array, loss_terms)

    # Updates are gathered from all devices across all hosts, so the return
    # value from pmap will contain duplicates for each local device. As such,
    # we take only the first.
    latent_updates = jax.tree_map((lambda x: np.array(x[0])), latent_updates)
    latent_ids, latent_grad = latent_updates
    latent_learning_rate = (
        self.learning_rate_start *
        (self.learning_rate_end / self.learning_rate_start)
        **(float(train_state.step) / self.max_steps))
    steps = -latent_learning_rate * latent_grad
    train_state.latent_table[latent_ids] += steps

    new_train_state = train_state.replace(
        model_parameters=new_model_parameters,
        optimizer_state=new_optimizer_state,
        latent_table=train_state.latent_table,
        rng=next_rng,
        step=train_state.step + 1)

    return new_train_state, loss_terms, scratch

  def eval_step(self, train_state: TransformerNeRFTrainState,
                summary_writer: tensorboard.SummaryWriter,
                scratch: Optional[Any]) -> Any:
    """Performs evaluation on a model checkpoint.

    Args:
      train_state: A state object tracking the current progress of training.
      summary_writer: Summary writer object for writing logs.
      scratch: Optional scratch state object for re-using values between steps.

    Returns:
      Updated scratch state object.
    """
    if scratch is None:
      # Override identity_batch_size to avoid OOM error on small eval topology.
      summary_data_reader = dataclasses.replace(
          self.data_reader,
          identity_batch_size=evaluation.EVAL_IDS_PER_BATCH)

      scratch = {
          "data_iterator":
              summary_data_reader.test_view_iter(),
          "summary_data":
              summary_data_reader.load_summary_data(self.summary_identities),
          "psnr_function":
              jax.pmap(evaluation.compute_batch_psnr),
          "image_renderer":
              self.model.create_image_renderer(),
      }

    step = train_state.step
    model_parameters = jax_utils.unreplicate(train_state.model_parameters)
    latent_table = train_state.latent_table

    # Copy summary_data so we can modify a local copy.
    summary_data = dict(scratch["summary_data"])
    batch_latents = latent_table[summary_data["identity_index"]]
    summary_data["latents"] = batch_latents

    # Order summary outputs fastest-to-slowest, such that preempted jobs have
    # more opportunities for output.

    # Write input images.
    columns = []
    for identity_images in summary_data["image"]:
      columns.append(jnp.concatenate(identity_images, axis=0))
    ground_truth_images = jnp.concatenate(columns, axis=1)
    summary_writer.image("Input", ground_truth_images, step=step)
    summary_writer.flush()

    # Write multi ID-view reconstructions.
    summary_image_grids = evaluation.render_id_view_grid(
        scratch["image_renderer"], summary_data, model_parameters, step)
    for name in summary_image_grids:
      # Prevent pixel values of 256 to make tensorboard happy.
      image = jnp.clip(summary_image_grids[name], 0.0, 0.99999)
      summary_writer.image(
          evaluation.RENDER_RESULT_TO_LABEL[name], image, step=step)
    summary_writer.flush()

    # Compute and write residual Image.
    residual_image = jnp.clip(
        jnp.abs(ground_truth_images / 255.0 - summary_image_grids["gamma_rgb"]),
        0.0, 1.0)
    residual_image = jnp.mean(residual_image, axis=2)
    residual_image = (residual_image * 255).astype("uint8")
    summary_writer.image("Reconstruction Residual", residual_image, step=step)

    # Write evaluation data PSNR.
    psnr = evaluation.compute_eval_psnr(model_parameters, latent_table,
                                        scratch["data_iterator"],
                                        scratch["psnr_function"], step)
    summary_writer.scalar("Eval PSNR", psnr, step=step)
    summary_writer.flush()

    return scratch

  def write_summaries(self, summary_writer: tensorboard.SummaryWriter,
                      summary_data: Sequence[Dict[str, np.ndarray]], step: int):
    if not summary_data:
      return

    for key in summary_data[0].keys():
      summary_writer.scalar(
          "Running Average Losses/" + key,
          np.mean(list(step_data[key] for step_data in summary_data)),
          step=step)
