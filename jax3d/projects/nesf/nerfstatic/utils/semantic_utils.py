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

"""Utilities for Semantic Training."""

import functools
from typing import List, Optional, Tuple

from absl import logging
import chex
import jax
from jax import numpy as jnp
from jax3d.projects.nesf.nerfstatic import datasets
from jax3d.projects.nesf.nerfstatic import serialization
from jax3d.projects.nesf.nerfstatic.models import mlp
from jax3d.projects.nesf.nerfstatic.models import model_utils
from jax3d.projects.nesf.nerfstatic.models import models
from jax3d.projects.nesf.nerfstatic.models import vanilla_nerf_mlp
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils import tree_utils
from jax3d.projects.nesf.utils.typing import Tree, f32  # pylint: disable=g-multiple-import
import numpy as np


@chex.dataclass
class RecoveredNeRFModel:
  """Contains the MLP, variables and sigma field."""

  # NeRF model architecture.
  model: vanilla_nerf_mlp.VanillaNerfMLP

  # Variables and sigma grids for training scenes.
  train_variables: List[Tree[jnp.ndarray]]
  train_sigma_grids: List[f32['1 d h w c']]
  train_scene_ids: Optional[List[f32]]

  # Variables and sigma grids for novel scenes.
  novel_variables: Optional[List[Tree[jnp.ndarray]]]
  novel_sigma_grids: Optional[List[f32['1 d h w c']]]
  novel_scene_ids: Optional[List[f32]]


@chex.dataclass
class RecomputeSigmaGridOptions:
  """Configuration for recomputed sigma grids."""

  # Spatial dimensions of the new sigma grid.
  sigma_grid_shape: Tuple[int, int, int]

  # Whether or not to apply f(x) = 1-exp(-relu(x)) to sigma.
  convert_sigma_to_density: bool

  # Whether or not to concatenate pointwise RGB outputs to the sigma grid.
  include_rgb: bool

  @classmethod
  def from_params(cls,
                  train_params,
                  ) -> Optional['RecomputeSigmaGridOptions']:
    """Builds RecomputeSigmaGridOptions from TrainParams."""
    if not train_params.nerf_model_recompute_sigma_grid:
      return None
    return RecomputeSigmaGridOptions(
        sigma_grid_shape=train_params.nerf_model_recompute_sigma_grid_shape,
        convert_sigma_to_density=(
            train_params.nerf_model_recompute_sigma_grid_convert_sigma_to_density),  # pylint: disable=line-too-long
        include_rgb=train_params.nerf_model_recompute_sigma_grid_include_rgb)


def load_all_nerf_variables(
    save_dir: str,
    train_dataset: datasets.DatasetIterable,
    novel_dataset: Optional[datasets.DatasetIterable],
    recompute_sigma_grid_opts: Optional[RecomputeSigmaGridOptions],
) -> RecoveredNeRFModel:
  """Loads parameters for all NeRF models on disk.

  Args:
    save_dir: CNS Path where the final checkpointed models are stored.
    train_dataset: Train dataset indicating which scenes to load corresponding
      NeRFs for.
    novel_dataset: Novel dataset indicating which scenes to load corrresponding
      NeRFs for.
    recompute_sigma_grid_opts: Optional configuration for recomputing sigma grid
      on-the-fly. If None, the sigma grid stored on disk is used instead.

  Returns:
    A RecoveredNeRFModel instance with the model, train_variables,
      novel_variables, train_sigma_grids and novel_sigma_grids.
  """
  logging.info('Loading NeRF states from: %s', save_dir)
  nerf_saver = serialization.NerfSaver(save_dir)
  load_state_fn = functools.partial(_load_nerf_state_for_scene,
                                    nerf_saver=nerf_saver)

  model, all_train_variables, all_train_sigma_grids, all_train_scene_ids = (
      _load_nerf_state(
          load_fn=load_state_fn, dataset=train_dataset,
          recompute_sigma_grid_opts=recompute_sigma_grid_opts))

  _, all_novel_variables, all_novel_sigma_grids, all_novel_scene_ids = (
      _load_nerf_state(
          load_fn=load_state_fn, dataset=novel_dataset,
          recompute_sigma_grid_opts=recompute_sigma_grid_opts))

  return RecoveredNeRFModel(
      model=model, train_variables=all_train_variables,
      train_sigma_grids=all_train_sigma_grids,
      train_scene_ids=all_train_scene_ids,
      novel_variables=all_novel_variables,
      novel_sigma_grids=all_novel_sigma_grids,
      novel_scene_ids=all_novel_scene_ids)


################################################################################
# Private helper functions


def _load_nerf_state(
    load_fn,
    dataset: Optional[datasets.DatasetIterable],
    recompute_sigma_grid_opts: Optional[RecomputeSigmaGridOptions]):
  """Helper function to load the model state, variables and sigma grid.

  Args:
    load_fn: See _load_nerf_state_for_scene() for guidance.
    dataset: Dataset to load NeRFs for. NeRFs are only loaded for scenes
      appearing in this dataset.
    recompute_sigma_grid_opts: Optional configuration for recomputing sigma grid
      on-the-fly. If None, the sigma grid stored on disk is used instead.

  Returns:
    model: VanillaNerfMLP instance. NeRF architecture shared by all NeRFs.
    all_variables: List of PyTrees. Each PyTree corresponds to the variables
      for a single NeRF scene.
    all_sigma_grids: Precomputed sigma grids associated with each pretrained
      NeRF. Final ReLU has not yet been applied.
    scene_ids: List of ints. Corresponds to the scene_id associated with each
      pretrained NeRF. scene_id is an increasing counter from 0 and is not
      associated with DatasetMetadata.scene_name.
  """

  model = None
  all_variables = None
  all_sigma_grids = None
  scene_ids = None
  if dataset is not None:

    nerf_states = tree_utils.parallel_map(load_fn, dataset.all_metadata)
    model = nerf_states[0][0]  # All models are assumed to be identical.
    all_variables = [state[1] for state in nerf_states]
    all_sigma_grids = [state[2] for state in nerf_states]

    # Use increasing integers for scene_id rather than the strings associated
    # with each NeRF. This corresponds to this logic in dataset.py,
    # http://jax3d.projects.nesf/nerfstatic/datasets/dataset.py;l=299-306;rcl=402602253
    #
    # scene_names = [state[3] for state in nerf_states]
    num_scenes = len(dataset.all_metadata)
    scene_ids = list(range(num_scenes))

    # Recompute sigma grids. We use a standard for-loop to limit peak memory
    # usage.
    if recompute_sigma_grid_opts is not None:
      logging.info('Recomputing %d sigma_grids with resolution=%s',
                   num_scenes,
                   recompute_sigma_grid_opts.sigma_grid_shape)
      recompute_sigma_grid_fn = functools.partial(
          recompute_sigma_grid,
          model=model,
          spatial_shape=recompute_sigma_grid_opts.sigma_grid_shape,
          convert_sigma_to_density=(
              recompute_sigma_grid_opts.convert_sigma_to_density),
          include_rgb=recompute_sigma_grid_opts.include_rgb)
      all_sigma_grids = [
          jax.device_get(recompute_sigma_grid_fn(variables=variables))
          for variables in all_variables
      ]
      del recompute_sigma_grid_fn

  return model, all_variables, all_sigma_grids, scene_ids


def _construct_nerf_model(
    params: models.NerfParams) -> vanilla_nerf_mlp.VanillaNerfMLP:
  """Helper function to construct a nerf model given the params."""
  net_activation = model_utils.get_net_activation(params)
  main_net_params = mlp.MlpParams(
      depth=params.net_depth,
      width=params.net_width,
      activation=net_activation,
      skip_layer=params.skip_layer,
      num_outputs=1)
  viewdir_net_params = mlp.MlpParams(
      depth=params.net_depth_condition,
      width=params.net_width_condition,
      activation=net_activation,
      num_outputs=3)
  return vanilla_nerf_mlp.VanillaNerfMLP(
      num_posencs=params.max_deg_point,
      viewdir_num_posencs=params.deg_view,
      use_viewdirs=params.use_viewdirs,
      net_params=main_net_params,
      viewdir_net_params=viewdir_net_params,
      num_scenes=1,
      num_scene_features=0,
      enable_sigma_semantic=False,
      name='fine_sample_store')


def _load_nerf_state_for_scene(scene_metadata,
                               *,
                               nerf_saver: serialization.NerfSaver):
  """Load & preprocess state for a single NeRF.

  Args:
    scene_metadata: DatasetMetadata object. Used for determining which NeRF to
      load.
    nerf_saver: NerfSaver object used for loading NeRFs.

  Returns:
    model: VanillaNerfMLP instance.
    variables: Variables for model.
    sigma_grid: Precomputed sigma grid derived from model.
    scene_name: string identifier for this scene.
  """
  assert scene_metadata.scene_name is not None
  logging.info('Loading NeRF state for scene_name=%s',
               scene_metadata.scene_name)

  # Note: scene_name is NOT necessarily the same as scene_id. scene_name is a
  # string. scene_id is an increasing integer index. If you set
  # DatasetParams.train_scenes = '5:8', then scene_names = ['5', '6', '7'] but
  # scene_ids = [0, 1, 2].
  scene_name = scene_metadata.scene_name

  nerf_state = nerf_saver.load_state(scene_metadata.scene_name)

  # Override 'preserve_sigma_grid'. This computation is no longer necessary.
  # in VolumetricSemanticModel.
  nerf_state.params.preserve_sigma_grid = False
  model = _construct_nerf_model(nerf_state.params)
  variables = {
      'params':
          nerf_state.variables['optimizer']['target']['params']
          ['fine_sample_store']
  }
  sigma_grid = jnp.expand_dims(nerf_state.density_grid, axis=0)

  # Spot check sigma grid. Does it match the values output by
  # the model?
  # _verify_sigma_grid(model, variables, sigma_grid)

  # Ensure all variables are in host memory.
  variables = jax.tree_map(jax.device_get, variables)
  sigma_grid = jax.device_get(sigma_grid)

  return model, variables, sigma_grid, scene_name


def _verify_sigma_grid(model, variables, sigma_grid):
  """Spot-check correctness of sigma grid."""
  # shape=[1, 1]
  scene_id = jnp.array([0], dtype=jnp.int32).reshape([1, 1])

  # Choose an XYZ point corresponding to an entry in the sigma grid.
  #
  # shape=[1, n, 3].
  N, X, Y, Z, _ = sigma_grid.shape  # pylint: disable=invalid-name
  assert N == 1
  all_coordinates = (model_utils.generate_grid(N, (X, Y, Z))
                     .reshape((N, X, Y, Z, 3)))

  # The coordinate chosen below is arbitrary.
  i = X // 2
  j = Y // 3
  k = (3*Z) // 4
  positions = jnp.asarray(all_coordinates[0, i, j, k]).reshape([1, 1, 3])

  # shape=[1, 3]
  direction = jnp.zeros((1, 3))

  # Evaluate NeRF.
  points = types.SamplePoints(scene_id=scene_id,
                              position=positions,
                              direction=direction)
  sigma_values = model.apply(variables, points).sigma

  # Ensure values align up to 1e-2 relative error. The actual values may
  # diverge thanks to floating point error in bfloat16.
  np.testing.assert_allclose(sigma_grid[0, i, j, k, 0],
                             sigma_values[0, 0, 0],
                             rtol=1e-2)


def _nested_stack(x):
  return jax.tree_map(lambda *args: jnp.stack(args), *x)


def _device_map(x: Tree[jnp.ndarray],
                num_devices: Optional[int] = None,
                ) -> Tree[jnp.ndarray]:
  """Distribute x's 0th dimension across local devices."""
  # Overwrite num_devices if undefined.
  if num_devices is None:
    num_devices = jax.local_device_count()

  def _device_map_internal(y):
    batch_size, *remaining_dims = y.shape
    if batch_size % num_devices != 0:
      raise ValueError(
          f'Unable to evenly distribute array of shape {x.shape}'
          f' across {num_devices} devices. Change batch size to be a multiple'
          f' of {num_devices} and try again.')
    batch_size_per_device = batch_size // num_devices
    return jnp.reshape(y, (num_devices, batch_size_per_device, *remaining_dims))
  return jax.tree_map(_device_map_internal, x)


def select_and_stack(select_scene_ids, elements, num_devices=None):
  """Similar to gather() but for PyTrees.

  Args:
    select_scene_ids: 1-D array of indexes.
    elements: List of PyTrees, each with the same structure.
    num_devices: Optional int. Number of devices to spoof for device map.

  Returns:
    A single PyTree with the same structure as an entry in elements.
    Each array in this PyTree will have dimensions [K, N/K, ...] where
    K = number of local devices and N = len(select_scene_ids).
  """
  result = [elements[scene_id] for scene_id in select_scene_ids]
  stacked_results = _nested_stack(result)
  return _device_map(stacked_results, num_devices)


def recompute_sigma_grid(model: vanilla_nerf_mlp.VanillaNerfMLP,
                         variables: Tree[jnp.ndarray],
                         spatial_shape: Tuple[int, int, int],
                         convert_sigma_to_density: bool,
                         include_rgb: bool,
                         ) -> f32['1 x y z c']:
  """Recomputes sigma grid using a pretrained model.

  Args:
    model: NeRF MLP architecture.
    variables: variables associated with this pretrained NeRF
    spatial_shape: spatial shape (x, y, z) for the result.
    convert_sigma_to_density: if True, apply f(x) = 1-exp(-relu(x)) to the
      sigma output of the NeRF MLP.
    include_rgb: if True, concatenate RGB output of the NeRF MLP to each point
      in the sigma grid. The RGB output is preprocessed with a sigmoid and is
      attenuated by f(sigma).

  Returns:
    Discretized version of NeRF MLP as an array of shape [1 x y z c]. At each
      spatial index (x, y, z), this array contains the sigma value of the
      NeRF MLP at that point.
  """
  # shape=[1, n, 3]
  sigma_grid_positions = model_utils.generate_grid(num_scenes=1,
                                                   grid_size=spatial_shape)
  sigma_grid_sample_points = types.SamplePoints(
      scene_id=jnp.asarray([[0]]),  # arbitrary value
      position=sigma_grid_positions,
      direction=jnp.zeros((1, 3)))  # arbitrary value

  # shape=[1, n, 1]
  sample_results: types.SampleResults = model.apply(
      variables, sigma_grid_sample_points)
  sigma_grid = sample_results.sigma
  num_feature_dims = 1

  if convert_sigma_to_density:
    sigma_grid = 1. - jnp.exp(-1 * jax.nn.relu(sample_results.sigma))

    # Include RGB if desired.
  if include_rgb:
    # The attenuation multiplier comes from the expression in the volumetric
    # rendering equation, but without the \delta term.
    attenuated_rgb = (jax.nn.sigmoid(sample_results.rgb) *
                      (1. - jnp.exp(-1 * jax.nn.relu(sample_results.sigma))))
    sigma_grid = jnp.concatenate([sigma_grid, attenuated_rgb], axis=-1)
    assert attenuated_rgb.shape[-1] == 3
    num_feature_dims += 3

    # shape=[1, x, y, z, k]
  sigma_grid = jnp.reshape(sigma_grid, [1, *spatial_shape, num_feature_dims])

  return sigma_grid
