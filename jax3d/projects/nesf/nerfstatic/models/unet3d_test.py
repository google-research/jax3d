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

"""Tests for jax3d.projects.nesf.nerfstatic.models.unet3d."""

from flax import linen as nn
import jax
from jax3d.projects.nesf.nerfstatic.models import unet3d


def test_simple():
  rng_key = jax.random.PRNGKey(0)
  init_key, p_key = jax.random.split(rng_key, 2)
  p = jax.random.uniform(p_key, shape=[2, 16, 16, 16, 5])
  params = unet3d.UNetParams(
      feature_size=(2, 4, 8, 16), activation_fn=nn.relu, output_dims=5, depth=1)
  m = unet3d.UNet3D(params=params)
  init_variables = m.init(init_key, p)
  output = m.apply(init_variables, p)
  assert output.shape == p.shape
