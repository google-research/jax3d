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

"""An extendable trainer executable in JAX."""

import dataclasses
import subprocess
import time
from typing import Any, Iterable, Optional, Sequence, Tuple

from absl import logging
from etils import epath
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
from jax3d.projects.generative.nerf import gpu_utilization


@flax.struct.dataclass
class TrainState:
  """Extendable base class for tracking training state."""
  step: int

  def to_serializable(self) -> "TrainState":
    """Transforms the state values into a form suitable for serialization.

    This function can be overridden to perform operations like dereplication to
    prepare the state for serialization.

    Returns:
      A TrainState object ready to be serialized.
    """
    return self

  def from_serializable(self) -> "TrainState":
    """Transforms deserialized values into a form suitable for training.

    This function can be overridden to perform operations like replication or
    construction of dataclasses from dictionaries to prepare the deserialized
    state for training.

    Returns:
      A TrainState object ready for training.
    """
    return self


@dataclasses.dataclass(frozen=True)
class LogState:
  """State object for tracking changes between log events."""
  last_log_step: int
  last_log_timestamp: float
  gpu_monitor: Optional[gpu_utilization.GPUMonitor]


@gin.configurable(
    name_or_fn="BaseTrainer", denylist=["experiment_name", "working_dir"])
@dataclasses.dataclass(frozen=True)
class Trainer:
  """Extendable model trainer.

  The (common) trainer is responsible for:
   - checkpoint saving & loading
   - writing summaries
   - going through training loop iterations
   - reporting utlization / time-per-step
  """
  experiment_name: str
  working_dir: str
  random_seed: Optional[int] = 1234

  max_steps: int = 1000000
  save_every: int = 1000
  log_every: int = 1000
  num_kept_checkpoints: int = 2

  # Allows loading a pre-trained checkpoint at model initialization, but does
  # not override loading/saving after that.
  pre_trained_checkpoint: str = ""

  @property
  def experiment_dir(self) -> epath.Path:
    return epath.Path(self.working_dir) / self.experiment_name

  @property
  def summary_dir(self) -> epath.Path:
    return self.experiment_dir / "summaries"

  @property
  def checkpoint_dir(self) -> epath.Path:
    return self.experiment_dir / "checkpoints"

  def log_device_config(self) -> None:
    """Logs device configuration for a given host machine."""
    logging.info("Starting process %d. There are %d processes.",
                 jax.process_index(), jax.process_count())
    logging.info("Found %d accelerator devices: %s.", jax.local_device_count(),
                 str(jax.local_devices()))
    logging.info("Found %d total devices: %s.", jax.device_count(),
                 str(jax.devices()))

  def initialize_experiment_directory(self) -> None:
    """Creates and populates the experiment directory.

    This writes initial folders and files for configs, checkpoints, and logs.
    """
    if jax.process_index() == 0:
      logging.info("experiment_dir = %s", self.experiment_dir)
      if not self.experiment_dir.exists():
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

      with (self.experiment_dir / "train_config.gin").open("w") as config_file:
        config_file.write(gin.config_str())

      logging.info("summary_dir = %s", self.summary_dir)
      if not self.summary_dir.exists():
        self.summary_dir.mkdir(parents=True, exist_ok=True)

      logging.info("checkpoint_dir = %s", self.checkpoint_dir)
      if not self.checkpoint_dir.exists():
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

  def train_step(self, train_state: TrainState, inputs: Any,
                 scratch: Optional[Any]) -> Tuple[TrainState, Any, Any]:
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
    del inputs
    return train_state.replace(step=train_state.step + 1), None, scratch

  def init_state(self) -> TrainState:
    """Initializes training state."""
    return TrainState(step=0)

  def init_data_loader(self) -> Iterable[Any]:
    """Initializes dataset loaders."""
    while True:
      yield None

  def write_summaries(self, summary_writer: tensorboard.SummaryWriter,
                      summary_data: Sequence[Any], step: int) -> None:
    """Writes training summaries."""

  def save_checkpoint(self, train_state: TrainState) -> None:
    """Saves training checkpoints."""
    train_state = train_state.to_serializable()
    filename = checkpoints.save_checkpoint(
        self.checkpoint_dir,
        train_state,
        train_state.step,
        keep=self.num_kept_checkpoints)
    logging.info("Wrote checkpoint to %s", filename)

  def load_checkpoint(self, init_state: TrainState) -> TrainState:
    """Loads most recent training checkpoint if it exists."""
    checkpoint_dir = self.checkpoint_dir
    if (checkpoints.latest_checkpoint(self.checkpoint_dir) is None and
        self.pre_trained_checkpoint):
      logging.info("Using pre-trained checkpoint %s",
                   self.pre_trained_checkpoint)
      checkpoint_dir = self.pre_trained_checkpoint
    new_state = checkpoints.restore_checkpoint(checkpoint_dir, init_state)

    if new_state.step != init_state.step:
      logging.info("Restored from checkpoint at step %i", new_state.step)
    return new_state.from_serializable()

  def train(self) -> None:
    """Iterates through training."""
    self.log_device_config()
    self.initialize_experiment_directory()

    checkpoint_state = self.load_checkpoint(self.init_state())
    state = checkpoint_state

    gpu_monitor = None
    if jax.process_index() == 0:
      summary_writer = self.create_summary_writer()
      summary_writer.text("Config", gin.config_str(), 0)

      if jax.default_backend() == "gpu":
        logging.info("Creating GPU monitor.")
        nvidia_smi_path = subprocess.check_output(["which", "nvidia-smi"
                                                  ]).decode("UTF-8").strip()
        gpu_monitor = gpu_utilization.GPUMonitor(
            nvidia_smi_path=nvidia_smi_path)

      log_state = LogState(state.step, time.perf_counter(), gpu_monitor)

    data_loader = self.init_data_loader()

    scratch = None
    collected_summary_data = []
    while state.step < self.max_steps:
      inputs = next(data_loader)
      state, summary_data, scratch = self.train_step(state, inputs, scratch)
      collected_summary_data.append(summary_data)

      if state.step <= checkpoint_state.step:
        # We leave it up to the deriving class to increment the step so as to
        # allow for fusing multiple steps inside a JIT call. This check errors
        # if the implementer forgets to increment which would cause an infinite
        # loop.
        raise RuntimeError("Step counter was not incremented between steps. "
                           "Please add an increment to avoid an infinite loop.")

      if jax.process_index() > 0:
        continue

      if (state.step - checkpoint_state.step) >= self.save_every:
        self.save_checkpoint(state)
        checkpoint_state = state

      if (state.step - log_state.last_log_step) >= self.log_every:
        logging.info("Training Step %d / %d", state.step, self.max_steps)
        self.write_summaries(summary_writer, collected_summary_data, state.step)
        collected_summary_data = []
        log_state = self.write_utilization_summary(summary_writer, state.step,
                                                   log_state)
        summary_writer.flush()

    if gpu_monitor is not None:
      logging.info("Terminating GPU monitor.")
      gpu_monitor.stop()

  def eval_step(self, train_state: TrainState,
                summary_writer: tensorboard.SummaryWriter,
                scratch: Optional[Any]) -> Any:
    """Performs evaluation on a model checkpoint.

    Args:
      train_state: A state object tracking the current progress of training.
      summary_writer: Summary writer object for writing logs.
      scratch: Optional scratch state object for re-using values between steps.
        The value of this object should not be depended on, as it will be erased
        when a job is interrupted.

    Returns:
      Updated scratch state object.
    """
    del train_state, summary_writer
    return scratch

  def eval(self) -> None:
    """Performs continuous evaluation of the model."""
    previous_state = self.init_state()

    summary_writer = self.create_summary_writer()

    logging.info("Starting eval loop.")
    scratch = None
    while True:
      state = self.load_checkpoint(previous_state)
      if state.step == previous_state.step:
        logging.info("No new checkpoints (%d <= %d).", state.step,
                     previous_state.step)
        time.sleep(10)
        continue

      scratch = self.eval_step(state, summary_writer, scratch)
      previous_state = state

      if state.step >= self.max_steps - 1:
        logging.info("Finished evaluation at %d steps.", state.step)
        break

  def create_summary_writer(self) -> tensorboard.SummaryWriter:
    """Creates summary writer."""
    logging.info("Creating summary writer.")
    return tensorboard.SummaryWriter(self.summary_dir)

  def write_utilization_summary(self, summary_writer: tensorboard.SummaryWriter,
                                step: int, log_state: LogState) -> LogState:
    """Writes compute-device utilization summary."""
    now = time.perf_counter()
    t_diff = now - log_state.last_log_timestamp
    step_diff = step - log_state.last_log_step
    summary_writer.scalar(
        "Debug/Steps per second", step_diff / t_diff, step=step)

    if log_state.gpu_monitor:
      utilization = log_state.gpu_monitor.get_utilization()
      if utilization is not None:
        for i, ui in enumerate(utilization):
          summary_writer.scalar(f"Debug/GPU {i} Utilization (%)", ui, step=step)

    return dataclasses.replace(
        log_state, last_log_step=step, last_log_timestamp=now)
