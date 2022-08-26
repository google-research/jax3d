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

"""Tests for transforms."""
from absl.testing import absltest
import jax
import jax.numpy as jnp

from jax3d.projects.generative.nerf import transforms


class TransformsTest(absltest.TestCase):

  def test_6d_conversion_to_rotation_matrix(self):
    key = jax.random.PRNGKey(0)
    random_eulers = jax.random.uniform(key, (3,))
    rotation_matrix = transforms.euler_to_rotation_matrix(random_eulers)
    rotation_six_dim = transforms.rotation_matrix_to_rotation_six_dim(
        rotation_matrix)
    self.assertEqual(rotation_six_dim.shape[-1], 6)
    reconstructed_matrix = transforms.rotation_six_dim_to_rotation_matrix(
        rotation_six_dim)
    jnp.allclose(reconstructed_matrix, rotation_matrix, atol=1e-7)


if __name__ == '__main__':
  absltest.main()
