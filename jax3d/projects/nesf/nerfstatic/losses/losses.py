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

"""Losses."""

from typing import Optional
import chex
import flax
import jax
import jax.numpy as jnp
from jax3d.projects.nesf.utils.typing import f32, i32  # pylint: disable=g-multiple-import


def l1_regularization(variables: flax.core.scope.FrozenVariableDict,
                      ) -> f32['']:
  """Computes the L1 regularization loss on the model variables.

  Args:
    variables: Variables to optimize in a nested tree structure.

  Returns:
    Average absolute value of all elements in all variables.
  """
  def tree_sum_fn(fn) -> jnp.ndarray:
    return jax.tree_util.tree_reduce(
        lambda x, y: x + fn(y), variables, initializer=0)

  return (
      tree_sum_fn(lambda z: jnp.sum(jnp.absolute(z))) /
      tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))


def l2_regularization(variables: flax.core.scope.FrozenVariableDict,
                      ) -> f32['']:
  """Computes the L2 regularization loss on the model variables.

  Args:
    variables: Variables to optimize in a nested tree structure.

  Returns:
    Average squared value of all elements in all variables.
  """
  def tree_sum_fn(fn) -> jnp.ndarray:
    return jax.tree_util.tree_reduce(
        lambda x, y: x + fn(y), variables, initializer=0)

  return (
      tree_sum_fn(lambda z: jnp.sum(z**2)) /
      tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))


def scenes_l1_regularization(variables: flax.core.scope.FrozenVariableDict,
                             ) -> f32['']:
  """Computes the L2 regularization loss on the scene parameters.

  Args:
    variables: Variables to optimize in a nested tree structure.

  Returns:
    Average squared value of all elements in all variables.
  """
  # Note: This will fail, if no grid has been defined.
  try:
    v = variables['params']['coarse_sample_store']['scenes']
  except KeyError as err:
    raise RuntimeError(f'scenes_l1_regularization can only be used with '
                       f'"scenes" parameters: {err}')
  return l1_regularization(v)


def l1_smoothness_regularization(original: f32['n k'],
                                 perturbed: f32['n k']) -> f32['']:
  """Computes the average absolute difference between original and perturbed.

  Args:
    original: values at original points.
    perturbed: values at perturbed points.

  Returns:
    Average value of |original - perturbed|
  """
  return jnp.mean(jnp.abs(original - perturbed))


def l2_loss(labels: f32['...'], predictions: f32['...']) -> f32['']:
  """Computes the reconstruction loss between the labels and the predictions.

  Args:
    labels: Array of arbitrary shape. Target values.
    predictions: Array of arbitrary shape. Shape must match 'labels'.
      Predicted values.

  Returns:
    Squared error between labels and predictions, averaged across all elements.
  """
  chex.assert_equal_shape([labels, predictions])
  return ((predictions - labels)**2).mean()


def softmax_cross_entropy_loss(
    logits: Optional[f32['... num_classes']],
    labels: i32['... 1'],
    mask: Optional[i32['... 1']] = None,
    ) -> f32['']:
  """Computes the softmax cross entropy loss between the logits and the labels.

  Args:
    logits: Predicted logits.
    labels: Ground-truth labels.
    mask: Optional semantic mask. If set, only pixels with mask value=1 are
      used for computing the loss. The rest are ignored.

  Returns:
    Negative log likelihood between the logits and the labels per-batch,
      averaged across all elements.
  """
  # If no semantics are predicted, loss is 0.
  if logits is None or logits.shape[-1] == 0:
    return 0.

  chex.assert_equal_shape_prefix([logits, labels], labels.ndim - 1)

  if mask is not None:
    labels = labels * mask

  logp = jax.nn.log_softmax(logits)
  loglik = jnp.take_along_axis(logp, labels, axis=-1)
  return -jnp.mean(loglik)


def ray_interval_regularization(
    z_values: f32['...'],
    weights: f32['...']
    ) -> f32['']:
  """Computes the regularization loss for a ray interval.

  Args:
    z_values: Sampled z values along the ray.
    weights: Ray contribution.

  Returns:
    Computes the double integral w_i w_j |x_i - x_j| d_i d_j
  """

  # The loss incurred between all pairs of intervals
  ux = (z_values[..., 1:] + z_values[..., :-1]) / 2
  dux = jnp.abs(ux[..., :, None] - ux[..., None, :])
  losses_cross = jnp.sum(
      weights * jnp.sum(weights[..., None, :] * dux, axis=-1), axis=-1)

  # The loss incurred within each individual interval with itself.
  losses_self = jnp.sum(weights**2 * (
      z_values[..., 1:] - z_values[..., :-1]), axis=-1) / 3

  return jnp.mean(losses_cross + losses_self)


def charbonnier_loss(labels: f32['...'], predictions: f32['...'], epsilon: f32
                     ) -> f32['']:
  """Computes the reconstruction loss between the labels and the predictions.

  Args:
    labels: Array of arbitrary shape. Target values.
    predictions: Array of arbitrary shape. Shape must match 'labels'.
      Predicted values.
    epsilon: Epsilon value to be added.

  Returns:
    Squared error between labels and predictions, averaged across all elements.
  """
  chex.assert_equal_shape([labels, predictions])
  return jnp.sqrt((predictions - labels)**2 + epsilon**2).mean()
