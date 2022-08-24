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

"""JAX utilities for working with geometric transformations."""

import jax.numpy as jnp


def euler_to_rotation_matrix(euler):
  """Convert [..., 3] Euler angles to [..., 3, 3] rotation matrix."""
  sin_angles = jnp.sin(euler)
  cos_angles = jnp.cos(euler)
  sinx, siny, sinz = tuple(sin_angles[..., i] for i in range(3))
  cosx, cosy, cosz = tuple(cos_angles[..., i] for i in range(3))
  return jnp.stack([
      cosy * cosz,
      (sinx * siny * cosz) - (cosx * sinz),
      (cosx * siny * cosz) + (sinx * sinz),
      cosy * sinz,
      (sinx * siny * sinz) + (cosx * cosz),
      (cosx * siny * sinz) - (sinx * cosz),
      -siny,
      sinx * cosy,
      cosx * cosy,
  ],
                   axis=-1).reshape(euler.shape[:-1] + (3, 3))


def rotation_matrix_to_euler(matrix):
  """Convert [..., 3, 3] rotation matrix to [..., 3] Euler angles.

  Note: does not handle gimbal lock, use with caution.

  Args:
    matrix: [..., 3, 3] rotation matrix.

  Returns:
    [..., 3] Euler angles.
  """
  nzs = lambda x: jnp.sign(jnp.sign(x) + 0.1)
  perturbed = lambda x: nzs(x) * 1e-6 + x
  y_angle = -jnp.arcsin(matrix[..., 2, 0])
  sign = nzs(jnp.cos(y_angle))
  z_angle = jnp.arctan2(matrix[..., 1, 0] * sign,
                        perturbed(matrix[..., 0, 0]) * sign)
  x_angle = jnp.arctan2(matrix[..., 2, 1] * sign,
                        perturbed(matrix[..., 2, 2]) * sign)
  return jnp.stack([x_angle, y_angle, z_angle], axis=-1)


def divide_safe(numerator: jnp.ndarray,
                denominator: jnp.ndarray,
                eps: float = 1e-7) -> jnp.ndarray:
  """Division of jnp.ndarray's with zero denominator safety."""
  denominator_ = jnp.where(denominator < eps, 1.0, denominator)
  return jnp.divide(numerator, denominator_)


def normalize_safe(array: jnp.ndarray,
                   axis: int = -1,
                   eps: float = 1e-7) -> jnp.ndarray:
  """Normalizes an jnp.ndarray with zero denominator safety."""
  return divide_safe(array, jnp.linalg.norm(array, axis=axis, keepdims=True),
                     eps=eps)


def rotation_six_dim_to_rotation_matrix(rotation_six_dim: jnp.ndarray,
                                        eps: float = 1e-7) -> jnp.ndarray:
  """Converts [..., 6] rotation representation to [..., 3, 3] rotation matrix.

  Args:
    rotation_six_dim: [..., 6] 6D continuous rotation representation.
    eps: a small positive float for controlling numerical stability in division.

  Returns:
    [..., 3, 3] rotation matrix.

  Note: for more information, read Zhou, Y., Barnes, C., Lu, J., Yang, J., &
  Li, H. On the Continuity of Rotation Representations in Neural Networks.
  IEEE Conference on Computer Vision and Pattern Recognition, 2019.
  Retrieved from http://arxiv.org/abs/1812.07035
  """
  a1, a2 = rotation_six_dim[..., :3], rotation_six_dim[..., 3:]
  b1 = normalize_safe(a1, eps=eps)
  b2 = a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1
  b2 = normalize_safe(b2, eps=eps)
  b3 = jnp.cross(b1, b2)
  return jnp.stack((b1, b2, b3), axis=-2)


def rotation_matrix_to_rotation_six_dim(matrix: jnp.ndarray) -> jnp.ndarray:
  """Converts [..., 3, 3] rotation matrices [..., 6] rotation representation.

  Args:
    matrix : [..., 3, 3] rotation matrix.

  Returns:
    [..., 6] 6-dimensional continuous rotation representation.


  Note: for more information, read Zhou, Y., Barnes, C., Lu, J., Yang, J., &
  Li, H. On the Continuity of Rotation Representations in Neural Networks.
  IEEE Conference on Computer Vision and Pattern Recognition, 2019.
  Retrieved from http://arxiv.org/abs/1812.07035
  """
  batch_dim = matrix.shape[:-2]
  return matrix[..., :2, :].reshape((*batch_dim, 6))
