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

"""Tests for jax3d.nerfstatic.models.vanilla_nerf_mlp."""

import jax
from jax import numpy as jnp
import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic.models import mlp
from jax3d.projects.nesf.nerfstatic.models import vanilla_nerf_mlp
from jax3d.projects.nesf.nerfstatic.utils import types
import pytest


@pytest.mark.parametrize('use_viewdirs', [True, False])
@pytest.mark.parametrize('enable_mipnerf', [True, False])
def test_simple(use_viewdirs: bool, enable_mipnerf: bool):
  rng = jax3d.RandomState(0)
  p = types.SamplePoints(
      scene_id=jnp.zeros([2, 1], dtype=jnp.int32),
      position=jax.random.uniform(rng.next(), shape=[2, 5, 3]),
      direction=jax.random.uniform(rng.next(), shape=[2, 3]),
      covariance=jax.random.uniform(rng.next(), shape=[2, 5, 3]))
  net_params = mlp.MlpParams(depth=8, width=11, skip_layer=4, num_outputs=1)
  viewdir_net_params = mlp.MlpParams(depth=8, width=13, num_outputs=3)
  m = vanilla_nerf_mlp.VanillaNerfMLP(use_viewdirs=use_viewdirs,
                                      num_posencs=10,
                                      viewdir_num_posencs=4,
                                      net_params=net_params,
                                      viewdir_net_params=viewdir_net_params,
                                      enable_mipnerf=enable_mipnerf)
  init_variables = m.init(rng.next(), p)
  output = m.apply(init_variables, p)
  assert output.rgb.shape == (2, 5, 3)
  assert output.sigma.shape == (2, 5, 1)
  assert output.semantic.shape == (2, 5, 0)


@pytest.mark.parametrize('use_viewdirs', [True, False])
@pytest.mark.parametrize('enable_mipnerf', [True, False])
def test_sigma_grid(use_viewdirs: bool, enable_mipnerf: bool):
  rng = jax3d.RandomState(0)
  p = types.SamplePoints(
      scene_id=jnp.zeros([2, 1], dtype=jnp.int32),
      position=jax.random.uniform(rng.next(), shape=[2, 5, 3]),
      direction=jax.random.uniform(rng.next(), shape=[2, 3]),
      covariance=jax.random.uniform(rng.next(), shape=[2, 5, 3]))
  net_params = mlp.MlpParams(depth=8, width=11, skip_layer=4, num_outputs=1)
  viewdir_net_params = mlp.MlpParams(depth=8, width=13, num_outputs=3)
  m = vanilla_nerf_mlp.VanillaNerfMLP(use_viewdirs=use_viewdirs,
                                      num_scenes=1,
                                      num_posencs=10,
                                      viewdir_num_posencs=4,
                                      net_params=net_params,
                                      viewdir_net_params=viewdir_net_params,
                                      enable_sigma_semantic=True,
                                      sigma_grid_size=(2, 2, 2),
                                      enable_mipnerf=enable_mipnerf)
  init_variables = m.init(rng.next(), p)
  output = m.apply(init_variables, p)
  assert output.rgb.shape == (2, 5, 3)
  assert output.sigma.shape == (2, 5, 1)
  assert output.semantic.shape == (2, 5, 0)
  assert output.sigma_grid.shape == (1, 2, 2, 2, 1)
