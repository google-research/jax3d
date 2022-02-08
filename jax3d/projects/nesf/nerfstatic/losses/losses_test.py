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

"""Tests for jax3d.projects.nesf.nerfstatic.losses.losses."""

import jax
from jax import numpy as jnp
from jax3d.projects.nesf.nerfstatic.losses import losses
import numpy as np
import tensorflow as tf


def test_softmax_cross_entropy_loss():
  labels = jnp.array([[0], [1], [2]], dtype=jnp.int32)
  logits = jnp.array([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0], [0.1, 0.8, 2.0]],
                     dtype=jnp.float32)
  loss_value = losses.softmax_cross_entropy_loss(logits=logits, labels=labels)

  predicted_log_logits = jnp.log(jax.nn.softmax(logits, axis=-1))
  log_likelihood = jnp.take_along_axis(predicted_log_logits, labels, axis=-1)
  expected_cross_entropy = -jnp.mean(log_likelihood)
  assert jnp.isclose(expected_cross_entropy, loss_value)


def test_softmax_cross_entropy_loss_against_tf():
  batch_shape = [2, 3, 5]
  num_classes = 7
  label_shape = batch_shape + [1]
  logit_shape = batch_shape + [num_classes]

  labels = np.random.randint(0, num_classes, size=label_shape, dtype=np.int32)
  logits = np.random.random(logit_shape)

  actual_loss = losses.softmax_cross_entropy_loss(logits=logits, labels=labels)
  expected_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=np.squeeze(labels, axis=-1)))
  assert jnp.isclose(actual_loss, np.mean(expected_loss))
