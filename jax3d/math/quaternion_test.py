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

"""Unit tests for quaternions."""

import functools
import math
import unittest

from jax import random
import jax.numpy as jnp
from jax3d.math import quaternion
import pytest


TEST_BATCH_SIZE = 128


class QuaternionTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = 42
    self._key = random.PRNGKey(self._seed)

  def test_identity(self):
    identity = quaternion.identity()
    self.assertLen(identity, 4)
    self.assertEqual(identity.tolist(), [0.0, 0.0, 0.0, 1.0])

  @pytest.mark.parametrize(('single', (4,)), ('batched', (TEST_BATCH_SIZE, 4)))
  @pytest.mark.parametrize('shape', )
  def test_real_imaginary_part(self, shape):
    if len(shape) > 1:
      num_quaternions = shape[0]
    else:
      num_quaternions = 1
    random_quat = random.uniform(self._key, shape=shape)
    imaginary = quaternion.im(random_quat)
    real = quaternion.re(random_quat)

    # The first three components are imaginary and the fourth is real.
    self.assertEqual(jnp.prod(jnp.array(imaginary.shape)), num_quaternions * 3)
    self.assertEqual(jnp.prod(jnp.array(real.shape)), num_quaternions)
    self.assertEqual(random_quat[..., :3].tolist(), imaginary[..., :].tolist())
    self.assertEqual(random_quat[..., 3:].tolist(), real[..., :].tolist())

  @pytest.mark.parametrize('batch', [None, TEST_BATCH_SIZE])
  @pytest.mark.parametrize('func', [random.uniform, jnp.ones, jnp.zeros])
  @pytest.mark.parametrize('sign', [-1, 1])
  def test_safe_acos(self, batch, func, sign):
    # We need a seed to generate random numbers.
    if func == random.uniform:
      func = functools.partial(func, key=self._key)

    if batch:
      shape = (batch, 4)
    else:
      shape = (4,)
    t = sign * func(shape=shape)

    output = quaternion.safe_acos(t)

    # All elements must be within the range of the arc-cosine function.
    self.assertTrue(jnp.all(output > 0))
    self.assertTrue(jnp.all(output < math.pi))

  @pytest.mark.parametrize(('single', None), ('batched', TEST_BATCH_SIZE))
  def test_conjugate(self, batch):
    if batch:
      shape = (batch, 4)
    else:
      shape = (4,)
    quat = random.uniform(self._key, shape=shape)
    conjugate = quaternion.conjugate(quat)
    self.assertTrue(jnp.all(-1 * quat[..., :3] == conjugate[..., :3]))
    self.assertTrue(jnp.all(quat[..., 3:] == conjugate[..., 3:]))

  @pytest.mark.parametrize(('single', None), ('batched', TEST_BATCH_SIZE))
  def test_normalize(self, batch):
    eps = 1e-6
    if batch:
      shape = (batch, 4)
    else:
      shape = (4,)
    q = random.uniform(self._key, shape=shape)
    self.assertTrue(jnp.all(jnp.abs(quaternion.norm(q) - 1) > eps))
    q_norm = quaternion.normalize(q)
    self.assertTrue(jnp.all(jnp.abs(quaternion.norm(q_norm) - 1) < eps))
