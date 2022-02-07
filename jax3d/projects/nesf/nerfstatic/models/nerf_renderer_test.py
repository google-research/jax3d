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

"""Tests for jax3d.nerfstatic.models.nerf_renderer.

This file checks the following:
1. The model can be loaded and the output shapes match expectation.
2. The gradients through the model are non-trivial.
3. Using a semantic loss only affects the semantic variables.
4. Using a RGB loss only affects the rgb, sigma and voxel variables.
5. Model intiialization is the same with and without semantics and UNet.
6. Model gradients are the same for models with and without semantics.
"""

import jax
from jax import numpy as jnp
import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic.models import nerf_renderer
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import f32  # pylint: disable=g-multiple-import
import pytest


def create_mock_sample_store():
  def mock_sample_store(p: types.SamplePoints,
                        weights: f32['...'] = None,
                        viewdir_weights: f32['...'] = None,
                        determinsitic: bool = True,
                        ) -> types.SampleResults:
    del weights, viewdir_weights, determinsitic  # Unused.
    # In a real example this method would be a NeRF MLP or similar.
    p = p.position
    semantic = jnp.ones(p.shape[:-1] + (0,), dtype=p.dtype)
    sigma_grid = jnp.ones(p.shape[:1] + (8, 8, 8, 1), dtype=p.dtype)
    return types.SampleResults(rgb=jnp.ones_like(p),
                               sigma=jnp.ones_like(p[..., :1]),
                               semantic=semantic,
                               sigma_grid=sigma_grid,
                               sigma_penultimate_embeddings=jnp.ones_like(p))
  return mock_sample_store


def create_mock_semantic_sample_store(num_semantic_classes: int):
  def mock_sample_store(p: types.SamplePoints,
                        sigma_grid: f32['...'] = None,
                        sigma_penultimate_features: f32['...'] = None
                        ) -> types.SampleResults:
    del sigma_grid, sigma_penultimate_features  # Unused.
    # In a real example this method would be a NeRF MLP or similar.
    p = p.position
    return jnp.ones(p.shape[:-1] + (num_semantic_classes,),
                    dtype=p.dtype)
  return mock_sample_store


@pytest.mark.parametrize('randomized_sampling', [True, False])
@pytest.mark.parametrize('num_semantic_classes', [0, 5])
@pytest.mark.parametrize('background', [types.BackgroundType.NONE,
                                        types.BackgroundType.WHITE])
@pytest.mark.parametrize('preserve_sigma', [True, False])
@pytest.mark.parametrize('enable_mipnerf', [True, False])
def test_simple(randomized_sampling: bool,
                num_semantic_classes: int,
                background: types.BackgroundType,
                preserve_sigma: bool,
                enable_mipnerf: bool):
  rng = jax3d.RandomState(0)
  rays = types.Rays(
      scene_id=jnp.zeros([2, 1], dtype=jnp.int32),
      origin=jax.random.uniform(rng.next(), shape=[2, 3]),
      direction=jax.random.uniform(rng.next(), shape=[2, 3]),
      base_radius=jax.random.uniform(rng.next(), shape=[2, 1]))
  sample_store_fn = create_mock_sample_store()
  semantic_sample_store_fn = create_mock_semantic_sample_store(
      num_semantic_classes)
  renderer = nerf_renderer.NerfRenderer(
      coarse_sample_store=sample_store_fn,
      fine_sample_store=sample_store_fn,
      semantic_sample_store=semantic_sample_store_fn,
      num_coarse_samples=7,
      num_fine_samples=17,
      lindisp=False,
      background_params=background,
      preserve_sigma_grid=preserve_sigma,
      enable_mipnerf=enable_mipnerf)

  rngs = {'params': rng.next(),
          'sampling': rng.next()}
  init_variables = renderer.init(rngs, rays=rays,
                                 randomized_sampling=randomized_sampling)
  output = renderer.apply(init_variables, rngs=rngs, rays=rays,
                          randomized_sampling=randomized_sampling)
  assert output.coarse.rgb.shape == (2, 3)
  assert output.coarse.disparity.shape == (2,)
  assert output.coarse.opacity.shape == (2,)
  assert output.fine.rgb.shape == (2, 3)
  assert output.fine.disparity.shape == (2,)
  assert output.fine.opacity.shape == (2,)
  assert output.fine.semantic.shape == (2, num_semantic_classes)
  assert output.coarse.foreground_rgb.shape == (2, 3)
  assert output.fine.foreground_rgb.shape == (2, 3)
  assert output.fine.foreground_semantic.shape == (2, num_semantic_classes)

  if preserve_sigma and num_semantic_classes > 0:
    assert output.fine.sigma_grid.shape == (2, 8, 8, 8, 1)

