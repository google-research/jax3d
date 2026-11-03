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

"""Utilization functions for handling rigid body transforms."""
import jax
from jax import numpy as jnp


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def divide_safe(numerator: jnp.ndarray,
                denominator: jnp.ndarray,
                eps: float = 1e-7) -> jnp.ndarray:
  """Division of jnp.ndarray's with zero denominator safety."""
  denominator_ = jnp.where(denominator < eps, 1.0, denominator)
  return jnp.divide(numerator, denominator_)


@jax.jit
def skew(w: jnp.ndarray) -> jnp.ndarray:
  """Build a skew matrix ("cross product matrix") for vector w.

  Modern Robotics Eqn 3.30.

  Args:
    w: (3,) A 3-vector

  Returns:
    W: (3, 3) A skew matrix such that W @ v == w x v
  """
  w = jnp.reshape(w, (3))
  return jnp.array([[0.0, -w[2], w[1]],
                    [w[2], 0.0, -w[0]],
                    [-w[1], w[0], 0.0]])


def rotation_translation_to_homogeneous_transform(
    rotation: jnp.ndarray, translation: jnp.ndarray) -> jnp.ndarray:
  """Rotation and translation to homogeneous transform.

  Args:
    R: (3, 3) An orthonormal rotation matrix.
    p: (3,) A 3-vector representing an offset.

  Returns:
    X: (4, 4) The homogeneous transformation matrix described by rotating by R
      and translating by p.
  """
  translation = jnp.reshape(translation, (3, 1))
  return jnp.block([[rotation, translation],
                    [jnp.array([[0.0, 0.0, 0.0, 1.0]])]])


def exp_so3(w: jnp.ndarray, theta: float) -> jnp.ndarray:
  """Exponential map from Lie algebra so3 to Lie group SO3.

  Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

  Args:
    w: (3,) An axis of rotation. This is assumed to be a unit-vector.
    theta: An angle of rotation.

  Returns:
    rotation: (3, 3) An orthonormal rotation matrix representing a rotation of
      magnitude theta about axis w.
  """
  w_skew = skew(w)
  return (jnp.eye(3)
          + jnp.sin(theta) * w_skew
          + (1.0 - jnp.cos(theta)) * matmul(w_skew, w_skew))


def exp_se3(screw_axis: jnp.ndarray, theta: float) -> jnp.ndarray:
  """Exponential map from Lie algebra so3 to Lie group SO3.

  Modern Robotics Eqn 3.88.

  Args:
    screw_axis: (6,) A screw axis of motion.
    theta: Magnitude of motion.

  Returns:
    a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
      motion of magnitude theta about S for one second.
  """
  w, v = jnp.split(screw_axis, 2)
  w_skew = skew(w)
  rotation = exp_so3(w_skew, theta)
  translation = matmul((theta * jnp.eye(3) + (1.0 - jnp.cos(theta)) * w_skew
                        + (theta - jnp.sin(theta)) * matmul(w_skew, w_skew)), v)
  return rotation_translation_to_homogeneous_transform(rotation, translation)


def to_homogenous(v):
  return jnp.concatenate([v, jnp.ones_like(v[..., :1])], axis=-1)


def from_homogenous(v):
  return v[..., :3] / v[..., -1:]


def se3_to_rotation_translation(
    se3: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes rotation and translation from 6D smooth manifold."""
  w, v = jnp.split(se3, 2, axis=-1)
  theta = jnp.linalg.norm(w, axis=-1)
  w = w / theta[..., None]
  rot_axis = jnp.concatenate((w, v), axis=-1)
  homo_trans = exp_se3(rot_axis, theta)
  rotation_matrix = homo_trans[..., :3, :3]
  translation_vector = homo_trans[..., :3, -1]
  return rotation_matrix, translation_vector


def hat_inv(skew_sym_matrix: jnp.ndarray) -> jnp.ndarray:
  """Computes the inverse Hat operator of a skew symmetric matrix.

  References:
    https://en.wikipedia.org/wiki/Hat_operator

  Args:
    skew_sym_matrix: a skew symmetric matrix of size 3x3

  Returns:
    a vector of length 3
  """
  x = skew_sym_matrix[..., 2, 1]
  y = skew_sym_matrix[..., 0, 2]
  z = skew_sym_matrix[..., 1, 0]

  v = jnp.stack((x, y, z), axis=-1)
  return v


def _taylor_first(x, nth=10):
  """Taylor expansion of sin(x)/x."""
  ans = jnp.zeros_like(x)
  denom = 1.
  for i in range(nth + 1):
    if i > 0:
      denom = denom * (2 * i) * (2 * i + 1)
    ans = ans + (-1)**i * x**(2 * i) / denom
  return ans


def _taylor_second(x, nth=10):
  """Taylor expansion of (1-cos(x))/x**2."""
  ans = jnp.zeros_like(x)
  denom = 1.
  for i in range(nth + 1):
    denom = denom * (2 * i + 1) * (2 * i + 2)
    ans = ans + (-1)**i * x**(2 * i) / denom
  return ans


def _taylor_third(x, nth=10):
  """Taylor expansion of (x-sin(x))/x**3."""
  ans = jnp.zeros_like(x)
  denom = 1.
  for i in range(nth + 1):
    denom = denom * (2 * i + 2) * (2 * i + 3)
    ans = ans + (-1)**i * x**(2 * i) / denom
  return ans


def rotation_translation_to_se3(rotation_matrix: jnp.ndarray,
                                translation_vector: jnp.ndarray,
                                eps: float = 1e-7) -> jnp.ndarray:
  """Computes the pseudo inverse of a smooth 6D vector of a rigid transform.

  References:
    https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf

  Args:
    rotation_matrix: a 3x3 rotation matrix.
    translation_vector: an array of length 3.
    eps: an epsilon for avoiding division by zero.

  Returns:
    a 6D se3 representation of the given rigid transform.
  """
  trace_rotation = rotation_matrix[..., 0, 0] + rotation_matrix[
      ..., 1, 1] + rotation_matrix[..., 2, 2]
  cos_theta = 0.5 * (trace_rotation - 1)
  sin_theta = jnp.sqrt(1 - cos_theta**2)
  theta = jnp.arccos(cos_theta)
  log_rotation = rotation_matrix - rotation_matrix.T
  log_rotation = log_rotation * (theta / (2 * sin_theta))
  w = hat_inv(log_rotation)
  wx = skew(w)
  identity = jnp.eye(3, dtype=jnp.float32)
  first_coeff = _taylor_first(theta)
  second_coeff = _taylor_second(theta)
  invesrse_v_matrix = identity - 0.5 * wx + (
      1 - first_coeff /
      (2 * second_coeff)) / (theta**2 + eps) * matmul(wx, wx)
  u = divide_safe(invesrse_v_matrix @ translation_vector, theta)
  wu = jnp.concatenate((w, u), axis=-1)
  return wu
