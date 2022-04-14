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

"""This file holds all parameters that are shared across all models."""

import dataclasses
from typing import Optional, Tuple

from flax import linen as nn
import jax.numpy as jnp
import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic.models import grid_interpolator
from jax3d.projects.nesf.nerfstatic.models import mlp
from jax3d.projects.nesf.nerfstatic.models import nerf_utils
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import ActivationFn
from jax3d.projects.nesf.utils.typing import f32
import numpy as np


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class ModelParams:
  """Base Model Params."""
  num_rgb_channels: int = 3  # the number of RGB channels.
  num_sigma_channels: int = 1  # the number of density channels.
  randomized: bool = True  # use randomized stratified sampling.
  min_deg_point: int = 0  # Minimum degree of positional encoding for points.
  max_deg_point: int = 10  # Maximum degree of positional encoding for points.
  max_deg_point_sem: int = 10  # Maximum degree of positional encoding for sem.
  deg_view: int = 4  # Degree of positional encoding for viewdirs.
  num_coarse_samples: int = 64  # the number of samples on each ray
                                # for the coarse model.
  num_fine_samples: int = 128  # the number of samples on
                               # each ray for the fine model. If set to zero,
                               # the coarse model is disabled and the fine model
                               # is evaluated on the coarse samples only.
  use_viewdirs: bool = True  # use view directions as a condition.
  lindisp: bool = False  # sampling linearly in disparity rather than depth.
  net_activation: str = "relu"  # activation function used within the MLP.
  rgb_activation: str = "sigmoid"  # activation function used to produce RGB.
  sigma_activation: str = "relu"  # activation function used to produce density.
  background: types.BackgroundType = jax3d.utils.EnumField(
      types.BackgroundType.WHITE    # pytype: disable=annotation-type-mismatch
  )
  noise_std: float = 0.0  # Noise to add to sigma during training.
  # Semantic Model Params.
  net_depth_semantic: int = 2  # depth for the semantic MLP.
  net_width_semantic: int = 128  # width of the semantic MLP.
  # Initialization of variables is identical whether semantics is enabled
  # or not.
  num_semantic_classes: int = 0  # number of semantic classes.

  interpolation_type: types.InterpolationType = jax3d.utils.EnumField(
      types.InterpolationType.TRILINEAR)  # pytype: disable=annotation-type-mismatch

  grid_features: int = 32  # Number of features stored in the latent grid.
                           # This setting is only used for --model=psf and for
                           # the semantics when psf_mode is True.

  enable_sigma_semantic: bool = True  # If True, passes density field to sem.
  sigma_grid_size: Tuple[int, int, int] = (32, 32, 32)
  unet_feature_size: Tuple[int, int, int, int] = (32, 64, 128, 256)
  unet_depth: int = 2  # Maximum 4.
  unet_activation_fn: ActivationFn = nn.relu
  # If specified, use given static near and far planes instead of computing them
  # from the scene bounding box.
  static_near_far: Optional[Tuple[float, float]] = None
  # By default this class will drop all aux data. This is usally the right thing
  # to do. Only enable for debugging or if you know what you are doing.
  preserve_aux: bool = False
  # By default this class will drop contribution values. This is usally the
  # right thing to do.
  # Only enable for debugging or if you know what you are doing.
  preserve_contribution: bool = False
  # Write sigma_grid of shape [X, Y, Z, 1] to disk for each scene in the
  # dataset. See self.sigma_grid_size for (X, Y, Z). Only written once at
  # the end of training.
  preserve_sigma_grid: bool = False
  # If set, VolumetricSemanticModel will apply f(x) = max(threshold, x) to
  # sigma_grid.
  threshold: Optional[int] = None
  # If set while threshold is also set, VolumetricSemanticModel will apply
  # f(x) = where(threshold > threshold, 1, 0) to sigma_grid.
  binarization: bool = False
  # If set, VolumetricSemanticModel will apply random rotations about the
  # z-axis before processing each scene. The scene box will be reduced by
  # a factor of 1/sqrt(2). This requires regenerating sigma grids at each
  # train step.
  apply_random_scene_rotations: bool = False
  # Maximum amount of rotation allowed by random scene rotations.
  random_scene_rotation_max_angle: float = 2 * np.pi
  # If true, MipNeRF is used instead of NeRF.
  enable_mipnerf: bool = False
  # Regularization for the ray distortion. Only used if enable_mipnerf is True.
  ray_regularization_loss_mult: float = 0.0


def _check_fn_boundaries(fn: ActivationFn,
                         min_value: float = -jnp.inf,
                         max_value: float = jnp.inf) -> None:
  x = jnp.exp(jnp.linspace(-90, 90, 1024))
  x = jnp.concatenate([-x[::-1], x], 0)
  x = fn(x)
  if jnp.any(x < min_value) or jnp.any(x > max_value):
    raise ValueError(f"fn `{fn}` produces values outside of "
                     f"[{min_value}, {max_value}]")


def get_activation(name: str) -> ActivationFn:
  """A helper to get an activation by name."""
  if name == "identity":
    return lambda x: x
  return getattr(nn, name)


def get_rgb_activation(args: ModelParams) -> ActivationFn:
  """A helper to get the rgb activation from args."""
  rgb_activation = get_activation(args.rgb_activation)
  _check_fn_boundaries(fn=rgb_activation, min_value=0, max_value=1)
  return rgb_activation


def get_sigma_activation(args: ModelParams) -> ActivationFn:
  """A helper to get the sigma activation from args."""
  sigma_activation = get_activation(args.sigma_activation)
  _check_fn_boundaries(fn=sigma_activation, min_value=0)
  return sigma_activation


def get_net_activation(args: ModelParams) -> ActivationFn:
  """A helper to get the net activation from args."""
  return get_activation(args.net_activation)


def generate_grid(
    num_scenes: int,
    grid_size: Tuple[int, int, int],
) -> f32["num_scenes *grid_size 3"]:
  """Generates 3D positional grid akin to np.meshgrid().

  Constructs a flattened version of an array of shape (num_scenes, X, Y, Z, 3)
  where (X, Y, Z) = grid_shape. The dimensions corresponding to X, Y, Z have
  been merged together. The 3-dimensional vector stored at (_, x, y, z)
  contains the coordinates corresponding to point (x, y, z). For example,

  ```
  shape = (2, 3, 4)
  x = generate_grid(1, shape)
  x[0]
  DeviceArray([[-1.        , -1.        , -1.        ],
               [-1.        , -1.        , -0.33333328],
               [-1.        , -1.        ,  0.33333337],
               [-1.        , -1.        ,  1.        ],
               [-1.        ,  0.        , -1.        ],
               [-1.        ,  0.        , -0.33333328],
               [-1.        ,  0.        ,  0.33333337],
               [-1.        ,  0.        ,  1.        ],
               [-1.        ,  1.        , -1.        ],
               [-1.        ,  1.        , -0.33333328],
               [-1.        ,  1.        ,  0.33333337],
               [-1.        ,  1.        ,  1.        ],
               [ 1.        , -1.        , -1.        ],
               [ 1.        , -1.        , -0.33333328],
               [ 1.        , -1.        ,  0.33333337],
               [ 1.        , -1.        ,  1.        ],
               [ 1.        ,  0.        , -1.        ],
               [ 1.        ,  0.        , -0.33333328],
               [ 1.        ,  0.        ,  0.33333337],
               [ 1.        ,  0.        ,  1.        ],
               [ 1.        ,  1.        , -1.        ],
               [ 1.        ,  1.        , -0.33333328],
               [ 1.        ,  1.        ,  0.33333337],
               [ 1.        ,  1.        ,  1.        ]], dtype=float32)
  ```

  Args:
    num_scenes: Number of scenes.
    grid_size: Latent grid size.

  Returns:
    3D array of coordinates. The 3-D vector at index (s, i) corresponds to the
    i-th coordinate for scene s.
  """
  x_grid, y_grid, z_grid = grid_size
  grid_total = int(x_grid * y_grid * z_grid)
  x_lin = np.linspace(-1, 1, x_grid, endpoint=True)
  y_lin = np.linspace(-1, 1, y_grid, endpoint=True)
  z_lin = np.linspace(-1, 1, z_grid, endpoint=True)

  # Use "ij" indexing rather than the default "xy" indexing. The default
  # indexing swaps the first two dimensions. See numpy documentation for
  # details.
  x_, y_, z_ = np.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
  x_y_z_ = np.stack((x_, y_, z_), axis=-1)
  x_y_z_points = x_y_z_.reshape((1, grid_total, 3))
  x_y_z_points = jnp.repeat(x_y_z_points, num_scenes, axis=0)
  return x_y_z_points


def generate_sigma_grid(
    num_scenes: int,
    grid_size: Tuple[int, int, int],
    embeddings: Optional[f32["num_grid, d, h, w, num_features"]],
    grid: Optional[grid_interpolator.GridInterpolator],
    num_posencs: int,
    sigma_decoder: mlp.MLP) -> f32["num_scenes *grid_size 1"]:
  """Generate sigma grid.

  Args:
    num_scenes: Number of scenes.
    grid_size: Latent grid size.
    embeddings: Embeddings for the latent grid.
    grid: interpolator
    num_posencs: Number of posencs.
    sigma_decoder: MLP sigma decoder.

  Returns:
    4D jnp array, with 3D sigma density field for each of num_scenes.

  """
  x_y_z_points = generate_grid(num_scenes, grid_size)
  if embeddings is not None and grid is not None:
    scene_id_sigma_grid = jnp.expand_dims(np.asarray(range(num_scenes)),
                                          axis=1)
    scene_id_sigma_grid = jnp.broadcast_to(scene_id_sigma_grid[..., None, :],
                                           x_y_z_points.shape[:-1] + (1,))
    latent_sigma_grid = grid(embeddings, scene_id_sigma_grid, x_y_z_points)
    grid_points = nerf_utils.posenc(x_y_z_points, num_posencs)
    latent_sigma_grid = jnp.concatenate([latent_sigma_grid, grid_points],
                                        axis=-1)
  else:
    latent_sigma_grid = nerf_utils.posenc(x_y_z_points, num_posencs)
  sigma_grid_output = sigma_decoder(latent_sigma_grid)
  sigma_grid = sigma_grid_output.predictions
  sigma_grid = jnp.reshape(sigma_grid, (num_scenes, *grid_size, 1))
  return sigma_grid
