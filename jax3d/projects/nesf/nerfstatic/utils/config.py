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

"""Utility functions."""

import dataclasses
from typing import Optional, Sequence, Tuple

from absl import flags
from absl import logging
import gin
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic import datasets as datasets_lib
from jax3d.projects.nesf.nerfstatic.models import model_utils
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.utils.typing import PathLike

# pylint: disable=logging-format-interpolation

flags.DEFINE_multi_string("gin_file", None,
                          "List of paths to the config files.")
flags.DEFINE_multi_string("gin_bindings", [],
                          "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class TrainParams:
  """Train Params."""
  # TODO(svora): Move train_dir to ConfigParams / alternate configurable class.
  train_dir: j3d.Path = j3d.utils.PathField()  # pytype: disable=annotation-type-mismatch
  lr_init: float = 5e-4  # The initial learning rate.
  lr_final: float = 1e-5  # The final learning rate.
  lr_delay_steps: int = 0  # The number of steps at the beginning of training
                           # to reduce the learning rate by lr_delay_mult
  lr_delay_mult: float = 1.  # A multiplier on the learnign rate when the step
                             # is < lr_delay_steps
  # If not None, gradients are clipped globally by this maximum norm.
  clip_grads_by_norm: Optional[float] = None
  train_steps: int = 500000  # The number of steps for training.

  save_every: int = 10000  # The number of steps to save a checkpoint.
  print_every: int = 100  # The number of steps between reports to tensorboard.
  gc_every: int = 10000  # The number of steps to run python garbage collection.
  log_per_scene: bool = False  # Whether to log per scene metrics.
  random_seed: int = 20200823  # Random seed to use.
  weight_decay_mult: int = 0  # The multiplier on weight decay.
  reconst_weight: float = 1.0  # Multiplier weight for reconstruction task.
  semantic_weight: float = 0.1  # Multiplier weight for semantics task.
  scenes_regularization_weight: float = 0.0  # Multiplier weight for scene
                                             # parameter reg. loss.
  # TODO(b/198251059): Remove this once fixed.
  overwrite_checkpoints: bool = False  # Whether to overwrite outdated ckpts.
  mode: str = "TRAIN"  # Train Mode to configure binaries.
  nerf_model_ckpt: Optional[str] = None  # Directory with NeRF models for
                                         # restoring. Only used in SEMANTIC
                                         # mode. See NerfSaver for details.

  # If True, recompute sigma grid instead of loading the one on disk.
  nerf_model_recompute_sigma_grid: bool = False

  # Resolution for the recomputed sigma grid.
  nerf_model_recompute_sigma_grid_shape: Tuple[int, int, int] = (32, 32, 32)

  # If true, apply f(x) = 1-exp(-relu(x)) to sigma.
  nerf_model_recompute_sigma_grid_convert_sigma_to_density: bool = False

  # If true, concatenate attenuated RGB to recomputed sigma grid.
  nerf_model_recompute_sigma_grid_include_rgb: bool = False

  # Configures smoothness regularization for semantic logits in 3D space.
  semantic_smoothness_regularization_num_points_per_device: int = 0
  semantic_smoothness_regularization_stddev: float = 0.0
  semantic_smoothness_regularization_weight: float = 0.0


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class EvalParams:
  """Eval Params."""
  eval_once: bool = False  # If True eval loop will only run once.
  save_output: bool = True  # Save predicted images to disk if True.
  # If set, run evaluation only on this many images.
  eval_num_images: Optional[int] = None
  eval_num_log_images: int = 4  # Maximum number of images to log per checkpoint
                                # during model evaluation.
  eval_num_log_train_images: int = 2  # Maximum number of train images to log
                                      # per checkpoint during model evaluation.
  chunk: int = 8192  # The size of chunks for evaluation inferences, set to the
                     # value that fits your GPU/TPU memory.
  normalize_disp: bool = False  # If True, disparity maps are normalized to
                                # [0, 1]. Otherwise, they are in absolute scene
                                # units.
  sigma_grid_dir: Optional[PathLike] = None  # Where to write sigma grid and
                                             # variables at the end of training.
  write_predictions_to_disk: bool = False  # If True, predictons are written to
                                           # disk as PNGs.

  # Which dataset splits to process at each checkpoint.
  enable_eval_train: bool = True
  enable_eval_test: bool = True
  enable_eval_novel_train: bool = True
  enable_eval_novel_test: bool = True


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class RenderParams:
  """Render Params."""
  # Subdirectory to write to.
  label: Optional[str] = None

  # How many camera rays can be rendered at a time.
  max_rays_per_render_call: int = 32768

  # Dataset config.
  dataset_split: str = "eval_train"

  # Each video and set of EPIs is based on a starting frame. How many of those
  # would you like to consider?
  num_start_frames: int = 1

  # Minimum and maximum depth for depth maps.
  min_depth: float = 0.0
  max_depth: float = 10.0

  # Parameters for rendering a video.
  enable_video: bool = True
  camera_path_method: str = "spiral"
  num_video_frames: int = 96
  video_fps: float = 24

  # Parameters for rendering epipolar plane images.
  enable_epipolar: bool = True
  epipolar_row_idxs: Optional[Sequence[int]] = (
      24, 48, 96, 128, 152, 176, 200, 224, 248)
  epipolar_num_rows: int = 256

  # Set to True to render videos from SparseConvNet predictions
  render_sparseconvnet: bool = False
  # CNS path to the predictions from the SparseConvNet model.
  sparseconvnet_predictions_path: str = ""


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class ConfigParams:
  """Config Params."""
  models: model_utils.ModelParams = gin_utils.ConfigField(
      model_utils.ModelParams, required=True
  )
  datasets: datasets_lib.DatasetParams = dataclasses.field(
      default_factory=datasets_lib.DatasetParams)
  train: TrainParams = dataclasses.field(default_factory=TrainParams)
  evaluate: EvalParams = dataclasses.field(default_factory=EvalParams)
  render: RenderParams = dataclasses.field(default_factory=RenderParams)


def root_config_from_flags() -> ConfigParams:
  """Calls `gin.parse_config_files_and_bindings`."""
  logging.info(
      f"parsing gin config_files: {FLAGS.gin_file} and bindings: "
      f"{FLAGS.gin_bindings}"
  )
  gin.parse_config_files_and_bindings(
      config_files=FLAGS.gin_file,
      bindings=FLAGS.gin_bindings,
  )
  # Ensure all scope names are valid
  gin_utils.validate_scope_names()
  params = ConfigParams()
  logging.info(
      f"parsed the following params:\n{params.to_gin_operative_repr()}"  # pytype: disable=attribute-error
  )
  return params


def export_gin_config(path: PathLike) -> None:
  """Export the gin config used.

  Export the FLAGS.gin_file and FLAGS.gin_bindings to the destination path.
  Values not specified in gin won't be saved on disk (default params value
  used).

  Args:
    path: Path to which save the .gin config.

  Raises:
    RuntimeError: If gin config is not locked.
  """
  if not gin.config_is_locked():
    raise RuntimeError(
        "log_gin_config_to_disk should only be called after config is "
        "finalized"
    )

  path = j3d.Path(path)
  path.parent.mkdir(exist_ok=True, parents=True)
  path.write_text(gin.config_str())


def import_params_from_gin_config(path: PathLike) -> ConfigParams:
  """Resore the params saved through .

  Warning: It's the caller responsibility to call `gin.clear_config()`
  before calling this method to avoid eventual conflict with other
  methods.

  Only gin bindings are restored, not other default `ConfigParams` values.

  Args:
    path: Gin path to which save the config

  Returns:
    params: The restored params
  """
  # Restore the bindings from the saved config
  gin.parse_config_files_and_bindings(config_files=[path], bindings=[])
  return ConfigParams()
