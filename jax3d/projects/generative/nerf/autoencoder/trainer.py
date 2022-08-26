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

"""Trainer for 2D autoencoder transformer model."""

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
from jax3d.projects.generative.nerf.autoencoder import eval as evaluation
from jax3d.projects.generative.nerf.autoencoder import loss
from jax3d.projects.generative.nerf.autoencoder import models
import numpy as np
import optax


@flax.struct.dataclass
class TransformerTrainState(trainer.TrainState):
  """Training state for the NeRF Transformer Model."""
  model_parameters: models.ModelParameters
  optimizer_state: optax.OptState
  rng: PRNGKey

  def to_serializable(self) -> "TransformerTrainState":
    """Transforms the state values into a form suitable for serialization."""
    return self.replace(
        model_parameters=jax_utils.unreplicate(self.model_parameters),
        optimizer_state=jax_utils.unreplicate(self.optimizer_state),
    )

  def from_serializable(self) -> "TransformerTrainState":
    """Transforms deserialized values into a form suitable for training."""
    state = self
    state = state.replace(
        model_parameters=jax_utils.replicate(state.model_parameters),
        optimizer_state=jax_utils.replicate(state.optimizer_state),
    )
    return state


@gin.configurable()
@dataclasses.dataclass(frozen=True)
class TransformerTrainer(trainer.Trainer):
  """Trainer implementation for the NeRF Transformer Model."""
  learning_rate_start: float = 5e-4
  learning_rate_end: float = 1e-4

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

  def init_state(self) -> TransformerTrainState:
    """Initializes training state."""
    rng = jax.random.PRNGKey(self.random_seed)
    train_rng, model_init_rng = jax.random.split(rng)
    res = self.data_reader.resolution
    model_parameters = self.model.initialize_parameters(model_init_rng,
                                                        image_size=(res, res))
    model_parameters = unfreeze(model_parameters)

    optimizer_state = self.optimizer.init(model_parameters)

    train_state = TransformerTrainState(  # pylint: disable=unexpected-keyword-arg
        step=0,
        model_parameters=model_parameters,
        optimizer_state=optimizer_state,
        rng=train_rng,
    )
    return train_state

  def init_data_loader(self) -> Iterable[Any]:
    """Initializes dataset loaders."""
    return iter(self.data_reader)

  def build_train_step(self) -> Callable[[Any], Any]:
    """Build the JIT-ed parallel train step function."""

    def per_device_train_step(model_parameters, optimizer_state, data,
                              rng, step):
      """The train step logic for a single device."""

      def _loss_fn(model_parameters, data, rng, step):
        return loss.transformer_loss_fn(model_parameters, data, rng, step)

      grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
      (total_loss, loss_terms), grad = grad_fn(model_parameters,
                                               data, rng, step)
      loss_terms["Total"] = total_loss

      grad = jax.lax.pmean(grad, axis_name="replicas")
      updates, optimizer_state = self.optimizer.update(grad,
                                                       optimizer_state,
                                                       model_parameters)
      model_parameters = optax.apply_updates(model_parameters, updates)

      return model_parameters, optimizer_state, loss_terms

    return jax.pmap(per_device_train_step, "replicas")

  def train_step(
      self, train_state: TransformerTrainState, inputs: Dict[str, Any],
      scratch: Optional[Dict[str, Any]]
  ) -> Tuple[TransformerTrainState, Dict[str, Any], Dict[str, Any]]:
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
      unrep_params = jax_utils.unreplicate(train_state.model_parameters)
      decoder_params = unrep_params["params"]["decoder"]
      leaves = jax.tree_leaves(decoder_params)
      param_count = sum(leaf.size for leaf in leaves)

      scratch = {
          "parallel_train_step": self.build_train_step(),
          "param_count": param_count,
      }

    next_rng, current_rng = jax.random.split(train_state.rng)
    current_rngs = jax.random.split(current_rng, jax.local_device_count())

    (new_model_parameters, new_optimizer_state,
     loss_terms) = scratch["parallel_train_step"](
         train_state.model_parameters, train_state.optimizer_state,
         inputs, current_rngs, jax_utils.replicate(train_state.step))  # pytype: disable=wrong-arg-count  # trace-all-classes

    loss_terms = jax.tree_map(np.array, loss_terms)
    loss_terms["param_count"] = scratch["param_count"]

    new_train_state = train_state.replace(
        model_parameters=new_model_parameters,
        optimizer_state=new_optimizer_state,
        rng=next_rng,
        step=train_state.step + 1)

    return new_train_state, loss_terms, scratch

  def eval_step(self, train_state: TransformerTrainState,
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
          batch_size=evaluation.EVAL_IDS_PER_BATCH,
          split="test")

      scratch = {
          "data_iterator": iter(summary_data_reader),
          "summary_data":
              summary_data_reader.load_summary_data(self.summary_identities),
          "render_function":
              jax.pmap(evaluation.render_frames, static_broadcasted_argnums=2),
          "psnr_function":
              jax.pmap(evaluation.psnr, static_broadcasted_argnums=2),
      }

    step = train_state.step
    model_parameters = train_state.model_parameters

    summary_data = scratch["summary_data"]

    # Write input images.
    columns = []
    for identity_images in summary_data["image_data"]["image"]:
      columns.append(jnp.concatenate(identity_images, axis=0))
    ground_truth_images = jnp.concatenate(columns, axis=1)
    summary_writer.image("Input", ground_truth_images, step=step)
    summary_writer.flush()

    rendered = scratch["render_function"](model_parameters, summary_data, step)
    rendered = list(rendered[i] for i in range(rendered.shape[0]))
    rendered = jnp.concatenate(rendered, axis=1)
    summary_writer.image("Rendered", rendered, step=step)
    summary_writer.flush()

    psnrs = []
    for _ in range(evaluation.EVAL_BATCHES_PER_CHECKPOINT):
      eval_batch = next(scratch["data_iterator"])
      eval_psnr = scratch["psnr_function"](model_parameters, eval_batch, step)
      psnrs.append(eval_psnr)
    psnrs = jnp.stack(psnrs)
    summary_writer.scalar("Eval PSNR", jnp.mean(psnrs), step=step)
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
