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

"""Tests for jax3d.nerfstatic.models.semantic_model."""

import jax
from jax import numpy as jnp
import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic.models import grid_interpolator
from jax3d.projects.nesf.nerfstatic.models import mlp
from jax3d.projects.nesf.nerfstatic.models import semantic_model
from jax3d.projects.nesf.nerfstatic.models import unet3d
from jax3d.projects.nesf.nerfstatic.utils import types
import pytest


@pytest.mark.parametrize('enable_sigma_semantic', [True, False])
@pytest.mark.parametrize('num_semantic_classes', [0, 3])
def test_simple(enable_sigma_semantic: bool, num_semantic_classes: int):
  num_posencs = 10
  rng = jax3d.RandomState(0)
  p = types.SamplePoints(
      scene_id=jnp.zeros([2, 1], dtype=jnp.int32),
      position=jax.random.uniform(rng.next(), shape=[2, 5, 3]),
      direction=jax.random.uniform(rng.next(), shape=[2, 3]))
  sigma_grid = jax.random.uniform(rng.next(), shape=[2, 8, 8, 8, 1])
  penultimate_features = jax.random.uniform(rng.next(), shape=[2, 5, 2])
  semantic_params = mlp.MlpParams(
      depth=2, width=4, num_outputs=num_semantic_classes)
  unet_params = unet3d.UNetParams(
      feature_size=(2, 2, 2, 2), output_dims=8)
  interpolation = grid_interpolator.TrilinearInterpolation()
  interpolator = grid_interpolator.GridInterpolator(interpolation=interpolation)
  m = semantic_model.SemanticModel(
      interpolator=interpolator,
      decoder_params=semantic_params,
      num_posencs=num_posencs,
      unet_params=unet_params, enable_sigma_semantic=enable_sigma_semantic)
  semantic_init_variables = m.init(
      rng.next(), points=p, sigma_grid=sigma_grid,
      sigma_penultimate_features=penultimate_features)

  output = m.apply(
      semantic_init_variables,
      points=p,
      sigma_grid=sigma_grid,
      sigma_penultimate_features=jax.lax.stop_gradient(penultimate_features))

  assert output.shape == (2, 5, num_semantic_classes)
