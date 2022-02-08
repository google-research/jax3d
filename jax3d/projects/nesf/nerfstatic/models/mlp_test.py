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

"""Tests for jax3d.projects.nesf.nerfstatic.models.mlp."""

import jax
from jax3d.projects.nesf.nerfstatic.models import mlp


def test_simple():
  rng_key = jax.random.PRNGKey(0)
  init_key, p_key = jax.random.split(rng_key, 2)
  p = jax.random.uniform(p_key, shape=[2, 3, 4, 5])
  params = mlp.MlpParams(depth=3, width=10, num_outputs=20)
  m = mlp.MLP(params=params)
  init_variables = m.init(init_key, p)
  output = m.apply(init_variables, p).predictions
  assert output.shape == (2, 3, 4, 20)
