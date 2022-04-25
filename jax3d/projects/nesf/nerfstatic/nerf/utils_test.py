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

"""Unit tests for utils."""
import functools

from absl.testing import absltest
from jax3d.projects.nesf.nerfstatic.nerf import utils
import numpy as np


class UtilsTest(absltest.TestCase):

  def test_learning_rate_decay(self):
    np.random.seed(0)
    for _ in range(10):
      lr_init = np.exp(np.random.normal() - 3)
      lr_final = lr_init * np.exp(np.random.normal() - 5)
      max_steps = int(np.ceil(100 + 100 * np.exp(np.random.normal())))

      lr_fn = functools.partial(
          utils.learning_rate_decay,
          lr_init=lr_init,
          lr_final=lr_final,
          max_steps=max_steps)

      # Test that the rate at the beginning is the initial rate.
      np.testing.assert_allclose(lr_fn(0), lr_init)

      # Test that the rate at the end is the final rate.
      np.testing.assert_allclose(lr_fn(max_steps), lr_final)

      # Test that the rate at the middle is the geometric mean of the two rates.
      np.testing.assert_allclose(lr_fn(max_steps / 2), np.sqrt(lr_init * lr_final))

      # Test that the rate past the end is the final rate
      np.testing.assert_allclose(lr_fn(max_steps + 100), lr_final)

  def test_delayed_learning_rate_decay(self):
    np.random.seed(0)
    for _ in range(10):
      lr_init = np.exp(np.random.normal() - 3)
      lr_final = lr_init * np.exp(np.random.normal() - 5)
      max_steps = int(np.ceil(100 + 100 * np.exp(np.random.normal())))
      lr_delay_steps = int(np.random.uniform(low=0.1, high=0.4) * max_steps)
      lr_delay_mult = np.exp(np.random.normal() - 3)

      lr_fn = functools.partial(
          utils.learning_rate_decay,
          lr_init=lr_init,
          lr_final=lr_final,
          max_steps=max_steps,
          lr_delay_steps=lr_delay_steps,
          lr_delay_mult=lr_delay_mult)

      # Test that the rate at the beginning is the delayed initial rate.
      np.testing.assert_allclose(lr_fn(0), lr_delay_mult * lr_init)

      # Test that the rate at the end is the final rate.
      np.testing.assert_allclose(lr_fn(max_steps), lr_final)

      # Test that the rate at after the delay is over is the usual rate.
      np.testing.assert_allclose(
          lr_fn(lr_delay_steps),
          utils.learning_rate_decay(lr_delay_steps, lr_init, lr_final,
                                    max_steps))

      # Test that the rate at the middle is the geometric mean of the two rates.
      np.testing.assert_allclose(lr_fn(max_steps / 2), np.sqrt(lr_init * lr_final))

      # Test that the rate past the end is the final rate
      np.testing.assert_allclose(lr_fn(max_steps + 100), lr_final)

  def test_get_color_coded_semantics_image(self):
    indexes = (np.random.rand(11, 13) * 256).astype(np.int32)
    colors = utils.get_color_coded_semantics_image(indexes)
    self.assertEqual(colors.shape, (11, 13, 3))


if __name__ == '__main__':
  absltest.main()
