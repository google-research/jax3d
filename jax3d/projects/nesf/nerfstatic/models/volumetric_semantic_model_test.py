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

"""Tests for volumetric_semantic_model."""

import jax
from jax import numpy as jnp

import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic.models import grid_interpolator
from jax3d.projects.nesf.nerfstatic.models import mlp
from jax3d.projects.nesf.nerfstatic.models import unet3d
from jax3d.projects.nesf.nerfstatic.models import vanilla_nerf_mlp
from jax3d.projects.nesf.nerfstatic.models import volumetric_semantic_model
from jax3d.projects.nesf.nerfstatic.utils import types


def test_simple():
  num_posencs = 2
  num_semantic_classes = 3
  rng = jax3d.RandomState(0)

  # Prepare NeRF.
  p = types.SamplePoints(
      scene_id=jnp.zeros([2, 1], dtype=jnp.int32),
      position=jax.random.uniform(rng.next(), shape=[2, 5, 3]),
      direction=jax.random.uniform(rng.next(), shape=[2, 3]))

  rays = types.Rays(
      scene_id=jnp.zeros([2, 1], dtype=jnp.int32),
      origin=jax.random.uniform(rng.next(), shape=[2, 3]),
      direction=jax.random.uniform(rng.next(), shape=[2, 3]))

  net_params = mlp.MlpParams(depth=2, width=4, skip_layer=1, num_outputs=1)
  viewdir_net_params = mlp.MlpParams(depth=2, width=4, num_outputs=3)
  nerf_model = vanilla_nerf_mlp.VanillaNerfMLP(
      use_viewdirs=True,
      num_posencs=3,
      viewdir_num_posencs=0,
      net_params=net_params,
      viewdir_net_params=viewdir_net_params)
  nerf_model_weights = nerf_model.init(rng.next(), p)

  # Prepare sigma grid.
  sigma_grid = jax.random.randint(rng.next(), shape=[1, 8, 8, 8, 1],
                                  minval=0., maxval=1.)

  # Prepare VolumetricSemanticModel.
  semantic_params = mlp.MlpParams(
      depth=2, width=4, num_outputs=num_semantic_classes)
  unet_params = unet3d.UNetParams(
      feature_size=(2, 2, 2, 2), output_dims=8, depth=1)
  interpolation = grid_interpolator.TrilinearInterpolation()
  interpolator = grid_interpolator.GridInterpolator(interpolation=interpolation)

  m = volumetric_semantic_model.VolumetricSemanticModel(
      nerf_model=nerf_model, interpolator=interpolator,
      semantic_decoder_params=semantic_params, unet_params=unet_params,
      num_posencs=num_posencs, num_samples=3, lindisp=True,
      apply_random_scene_rotations=True)

  rngs = {'params': rng.next(),
          'sampling': rng.next(),
          'data_augmentation': rng.next()}
  semantic_init_variables = m.init(
      rngs, rays=rays, randomized_sampling=True,
      is_train=True, sigma_grid=sigma_grid,
      nerf_model_weights=nerf_model_weights)

  # 3D points for semantic classification
  #
  # TODO(duckworthd): Find a way to eliminiate the length-1 dimension in the
  # middle.
  points = types.SamplePoints(
      scene_id=jnp.zeros((8, 1), dtype=jnp.int32),
      position=jax.random.uniform(rng.next(), shape=[8, 1, 3]),
      direction=jnp.zeros((8, 3)))

  rendered_rays, semantic_3d_predictions = m.apply(
      semantic_init_variables,
      rngs=rngs,
      rays=rays,
      randomized_sampling=True,
      is_train=True,
      sigma_grid=sigma_grid,
      nerf_model_weights=nerf_model_weights,
      points=points)

  assert rendered_rays.semantic.shape == (2, num_semantic_classes)
  assert semantic_3d_predictions.shape == (8, 1, num_semantic_classes)
