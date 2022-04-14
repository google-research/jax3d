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

"""Different model implementation plus a general port for all the models."""

import dataclasses
from typing import Optional, Tuple
from absl import logging
import flax.linen as nn
import jax.numpy as jnp
import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic.models import grid_interpolator
from jax3d.projects.nesf.nerfstatic.models import mlp
from jax3d.projects.nesf.nerfstatic.models import model_utils
from jax3d.projects.nesf.nerfstatic.models import nerf_renderer
from jax3d.projects.nesf.nerfstatic.models import semantic_model
from jax3d.projects.nesf.nerfstatic.models import unet3d
from jax3d.projects.nesf.nerfstatic.models import vanilla_nerf_mlp
from jax3d.projects.nesf.nerfstatic.models import volumetric_semantic_model
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import f32, Tree  # pylint: disable=g-multiple-import


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class NerfParams(model_utils.ModelParams):
  """Nerf Params."""
  net_depth: int = 8  # depth of the first part of MLP.
  net_width: int = 256  # width of the first part of MLP.
  net_depth_condition: int = 1  # depth of the second part of MLP.
  net_width_condition: int = 128  # width of the second part of MLP.
  skip_layer: int = 4  # add a skip connection to the output
                       # vector of every skip_layer layers.
  num_scene_features: int = 0  # Number of per scene features.

  @classmethod
  def from_dict(cls, contents):
    """Instantiate a NerfParams instance from a dictionary."""
    result = cls(**contents)

    # Convert non-serializable fields back to their native types.
    # Note: Enum fields don't need to be manually converted.
    result.unet_activation_fn = getattr(nn, result.unet_activation_fn)

    return result

  def to_dict(self):
    """Serialize NerfParams as a dictionary."""
    result = dataclasses.asdict(self)

    # Convert non-serializable fields to a serializable format.
    result["background"] = self.background.name
    result["interpolation_type"] = self.interpolation_type.name
    result["unet_activation_fn"] = self.unet_activation_fn.__name__

    return result


def get_model(rng: jax3d.RandomState,
              args: model_utils.ModelParams,
              num_scenes: int,
              placeholder_batch: jnp.ndarray,
              ) -> types.InitializedModel:
  """A helper function that wraps around a 'model zoo'.

  Produces initialized models.

  Args:
    rng: Random number generator seed. Used for initializating model
      parameters.
    args: model_utils.ModelParams defined in models.py.
    num_scenes: the number of scenes to model.
    placeholder_batch: Placeholder batch to initialize the models.

  Returns:
    Initialized model.
  """
  logging.info("Initializing model=%s", type(args).__name__)
  # TODO(b/192059669): Move model construction into each model's file.
  model_dict = {
      NerfParams: construct_nerf,
  }
  # Set target_view placeholder batch shape because it is different from the
  # shape that the dataset yields.
  placeholder_batch = jax3d.zeros_like(
      types.Batch.as_types(target_batch_shape=(1, 1),
                           enable_base_radii=args.enable_mipnerf))  # B N
  return model_dict[type(args)](rng, num_scenes, placeholder_batch, args)  # pytype: disable=wrong-arg-types


def create_semantic_model(
    args: model_utils.ModelParams,
    name: str,
) -> semantic_model.SemanticModel:
  """Helper function to create the semantic model."""
  semantic_net_params = mlp.MlpParams(
      depth=args.net_depth_semantic,
      width=args.net_width_semantic,
      activation=model_utils.get_net_activation(args),
      num_outputs=args.num_semantic_classes)
  unet_params = unet3d.UNetParams(
      depth=args.unet_depth,
      feature_size=args.unet_feature_size,
      activation_fn=args.unet_activation_fn,
      output_dims=args.grid_features)

  assert args.interpolation_type == types.InterpolationType.TRILINEAR, (
      "Unknown interpolation type. Supported value: TRILINEAR "
      f"Selected: {args.interpolation_type}")
  interpolation = grid_interpolator.TrilinearInterpolation()
  grid = grid_interpolator.GridInterpolator(
      interpolation=interpolation)
  return semantic_model.SemanticModel(
      interpolator=grid, decoder_params=semantic_net_params,
      num_posencs=args.max_deg_point_sem,
      enable_sigma_semantic=args.enable_sigma_semantic,
      unet_params=unet_params, name=name)


def construct_nerf(
    rng: jax3d.RandomState,
    num_scenes: int,
    placeholder_batch: types.Batch,
    args: NerfParams,
) -> types.InitializedModel:
  """Construct a Neural Radiance Field.

  Args:
    rng: the random state.
    num_scenes: unused.
    placeholder_batch: An example of a batch of data.
    args: Hyperparameters of nerf.

  Returns:
    Initialized model.
  """

  rgb_activation = model_utils.get_rgb_activation(args)
  sigma_activation = model_utils.get_sigma_activation(args)
  net_activation = model_utils.get_net_activation(args)

  def create_mlp(enable_sigma_semantic: bool,
                 sigma_grid_size: Optional[Tuple[int, int, int]],
                 name: str):
    main_net_params = mlp.MlpParams(
        depth=args.net_depth,
        width=args.net_width,
        activation=net_activation,
        skip_layer=args.skip_layer,
        num_outputs=1)
    viewdir_net_params = mlp.MlpParams(
        depth=args.net_depth_condition,
        width=args.net_width_condition,
        activation=net_activation,
        num_outputs=3)
    return vanilla_nerf_mlp.VanillaNerfMLP(
        num_posencs=args.max_deg_point,
        viewdir_num_posencs=args.deg_view,
        use_viewdirs=args.use_viewdirs,
        net_params=main_net_params,
        viewdir_net_params=viewdir_net_params,
        num_scenes=num_scenes,
        num_scene_features=args.num_scene_features,
        enable_sigma_semantic=enable_sigma_semantic,
        sigma_grid_size=sigma_grid_size,
        enable_mipnerf=args.enable_mipnerf, name=name)

  if args.num_semantic_classes > 0:
    semantic_mlp = create_semantic_model(
        args, name="semantic_mlp")
  else:
    semantic_mlp = None

  coarse_mlp = create_mlp(
      enable_sigma_semantic=args.enable_sigma_semantic,
      sigma_grid_size=args.sigma_grid_size,
      name="coarse_mlp")
  fine_mlp = create_mlp(
      enable_sigma_semantic=args.enable_sigma_semantic,
      sigma_grid_size=args.sigma_grid_size,
      name="fine_mlp")

  model = nerf_renderer.NerfRenderer(
      coarse_sample_store=coarse_mlp,
      fine_sample_store=fine_mlp,
      semantic_sample_store=semantic_mlp,
      rgb_activation=rgb_activation,
      sigma_activation=sigma_activation,
      num_coarse_samples=args.num_coarse_samples,
      num_fine_samples=args.num_fine_samples,
      lindisp=args.lindisp,
      background_params=args.background,
      noise_std=args.noise_std,
      static_near_far=args.static_near_far,
      preserve_aux=args.preserve_aux,
      preserve_contribution=args.preserve_contribution,
      preserve_sigma_grid=args.preserve_sigma_grid,
      enable_mipnerf=args.enable_mipnerf,
      enable_ray_regularization=args.ray_regularization_loss_mult > 0)

  init_variables = model.init(
      rngs={"params": rng.next(), "sampling": rng.next()},
      rays=placeholder_batch.target_view.rays,
      randomized_sampling=args.randomized)

  return types.InitializedModel(model=model, variables=init_variables)


# TODO(anyone): Fold nerf_sigma_grid, nerf_model_weights into
# placeholder_batch.
def construct_volumetric_semantic_model(
    rng: jax3d.RandomState,
    num_scenes: int,
    placeholder_batch: types.Batch,
    args: model_utils.ModelParams,
    nerf_model: vanilla_nerf_mlp.VanillaNerfMLP,
    nerf_variables: Tree[jnp.ndarray],
    nerf_sigma_grid: f32["..."],
    ) -> types.InitializedModel:
  """Constructs a VolumetricSemanticModel instance.

  Args:
    rng: Random seed for model initialization.
    num_scenes: Unused.
    placeholder_batch: Batch used for initializing model shape.
    args: Model HParams.
    nerf_model: NeRF-MLP architecture with per-scene density field.
    nerf_variables: NeRF-MLP model parameters.
    nerf_sigma_grid: Density grid derived from nerf_model and
      nerf_model_weights.

  Returns:
    Initialized VolumetricSemanticModel instance.
  """
  del num_scenes

  interpolation = grid_interpolator.TrilinearInterpolation()
  grid = grid_interpolator.GridInterpolator(interpolation=interpolation)
  semantic_net_params = mlp.MlpParams(
      depth=args.net_depth_semantic,
      width=args.net_width_semantic,
      activation=model_utils.get_net_activation(args),
      num_outputs=args.num_semantic_classes)
  unet_params = unet3d.UNetParams(
      depth=args.unet_depth,
      feature_size=args.unet_feature_size,
      activation_fn=args.unet_activation_fn,
      output_dims=args.grid_features)
  model = volumetric_semantic_model.VolumetricSemanticModel(
      nerf_model=nerf_model,
      interpolator=grid,
      semantic_decoder_params=semantic_net_params,
      unet_params=unet_params,
      num_posencs=args.max_deg_point_sem,
      num_samples=args.num_fine_samples,
      lindisp=args.lindisp,
      static_near_far=args.static_near_far,
      threshold=args.threshold,
      binarization=args.binarization,
      apply_random_scene_rotations=args.apply_random_scene_rotations,
      random_scene_rotation_max_angle=args.random_scene_rotation_max_angle,
  )
  variables = model.init(
      {"params": rng.next(),
       "sampling": rng.next(),
       "data_augmentation": rng.next()},
      placeholder_batch.target_view.rays,
      randomized_sampling=True,
      is_train=True,
      sigma_grid=nerf_sigma_grid,
      nerf_model_weights=nerf_variables)
  return types.InitializedModel(model=model, variables=variables)
