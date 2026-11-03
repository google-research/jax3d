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

import math
import unittest

import jax
from jax import numpy as jnp
from jax import random
from jax3d.math import quaternion
from jax3d.math import rigid_body
import numpy as np
import pytest


TEST_BATCH_SIZE = 128
SAMPLE_POINTS = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1)]


class RigidBodyTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = 42
    self._key = random.PRNGKey(self._seed)

  @staticmethod
  def _process_parameters(batch, vector_size=4):
    if batch:
      shape = (batch, vector_size)
      num_vectors = batch
    else:
      shape = (vector_size,)
      num_vectors = 1

    return shape, num_vectors

  def get_random_vector(self, func, shape):
    if func == random.uniform:
      self._key, _ = random.split(self._key)
      return func(shape=shape, key=self._key)
    else:
      return func(shape=shape)

  @pytest.mark.parametrize('batch', [None, TEST_BATCH_SIZE])
  @pytest.mark.parametrize('func', [random.uniform, jnp.ones])
  @pytest.mark.parametrize('sign', [-1, 1])
  def test_from_homogenous(self, batch, func, sign):
    shape, num_vectors = self._process_parameters(batch, 4)
    vector = sign * self.get_random_vector(func, shape=shape)
    output = rigid_body.from_homogenous(vector)
    self.assertEqual(jnp.prod(jnp.array(output.shape)), num_vectors * 3)
    np.testing.assert_array_equal(output, vector[..., :3] / vector[..., -1:])

  @pytest.mark.parametrize('batch', [None, TEST_BATCH_SIZE])
  @pytest.mark.parametrize('func', [random.uniform, jnp.ones, jnp.zeros])
  @pytest.mark.parametrize('sign', [-1, 1])
  def test_to_homogenous(self, batch, func, sign):
    shape, num_vectors = self._process_parameters(batch, 3)
    vector = sign * self.get_random_vector(func, shape=shape)
    output = rigid_body.to_homogenous(vector)
    self.assertEqual(jnp.prod(jnp.array(output.shape)), num_vectors * 4)
    np.testing.assert_array_equal(output[..., :3], vector)
    np.testing.assert_array_equal(output[..., -1:], 1.0)

  @pytest.mark.parametrize('func', [random.uniform, jnp.ones, jnp.zeros])
  @pytest.mark.parametrize('sign1', [-1, 1])
  @pytest.mark.parametrize('sign2', [-1, 1])
  def test_skew_matrix(self, func, sign1, sign2):
    # The skew function does not support batched operation.
    shape, _ = self._process_parameters(None, 3)
    w = sign1 * self.get_random_vector(func, shape=shape)
    v = sign2 * self.get_random_vector(func, shape=shape)
    skew_matrix = rigid_body.skew(w)

    # Properties of a skew symmetric matrix.
    self.assertEqual(jnp.trace(skew_matrix), 0)
    np.testing.assert_array_equal(-1 * jnp.transpose(skew_matrix), skew_matrix)

    # Does the matrix approximate the actual cross product?
    expected_cross_product = jnp.cross(w, v)
    predicted_cross_product = jnp.matmul(skew_matrix, v)
    np.testing.assert_allclose(
        expected_cross_product, predicted_cross_product, atol=1E-5, rtol=1E-5)

  @pytest.mark.parametrize('func', [random.uniform, jnp.ones])
  @pytest.mark.parametrize('sign1', [-1, 1])
  @pytest.mark.parametrize('sign2', [-1, 1])
  def test_exp_so3(self, func, sign1, sign2):
    shape, num_vectors = self._process_parameters(None, 3)

    # Generate a normalized axis of rotation and the angle of rotation.
    w = sign1 * self.get_random_vector(func, shape=shape)
    w = w / jnp.linalg.norm(w)

    theta = sign2 * self.get_random_vector(func, shape=(num_vectors, 1))
    output = rigid_body.exp_so3(w, theta)

    # Verify orthonormality.
    np.testing.assert_allclose(
        jnp.matmul(jnp.transpose(output), output),
        jnp.eye(3),
        atol=1E-5,
        rtol=1E-5)
    np.testing.assert_allclose(
        jnp.matmul(output, jnp.transpose(output)),
        jnp.eye(3),
        atol=1E-5,
        rtol=1E-5)

  @pytest.mark.parametrize('axis', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  @pytest.mark.parametrize('theta', [x * math.pi / 4 for x in range(8)])
  @pytest.mark.parametrize('sign', [-1, 1])
  @pytest.mark.parametrize('pt_input', SAMPLE_POINTS)
  def test_exp_so3_rotation(self, axis, theta, sign, pt_input):
    axis = jnp.array(axis)
    theta = jnp.array(sign * theta)
    pt_input = jnp.array(pt_input)
    theta = jnp.expand_dims(theta, 0)

    axis = axis / jnp.linalg.norm(axis)
    rotation_matrix = rigid_body.exp_so3(axis, theta)
    predicted_output = jnp.matmul(rotation_matrix, pt_input)

    # Use a quaternion to compute the rotation and use it as a comparison.
    quat = quaternion.from_axis_angle(axis, theta)
    quaternion_output = quaternion.rotate(quat, pt_input)
    np.testing.assert_allclose(
        predicted_output, quaternion_output, atol=1E-5, rtol=1E-5)

  @pytest.mark.parametrize('func', [random.uniform, jnp.ones, jnp.zeros])
  @pytest.mark.parametrize('sign1', [-1, 1])
  @pytest.mark.parametrize('sign2', [-1, 1])
  @pytest.mark.parametrize('sign3', [-1, 1])
  def test_rotation_translation_to_homogeneous_transform(self, func, sign1, sign2, sign3):
    shape, num_vectors = self._process_parameters(None, 3)
    w = sign1 * self.get_random_vector(func, shape=shape)
    w = w / jnp.linalg.norm(w)

    theta = sign2 * self.get_random_vector(func, shape=(num_vectors, 1))
    r = rigid_body.exp_so3(w, theta)

    p = sign3 * self.get_random_vector(func, shape=(num_vectors, 3))
    output = rigid_body.rotation_translation_to_homogeneous_transform(r, p)
    self.assertEqual(output.shape, (4, 4))
    np.testing.assert_array_equal(jnp.squeeze(r), jnp.squeeze(output[0:3, 0:3]))
    np.testing.assert_array_equal(jnp.squeeze(p), jnp.squeeze(output[0:3, 3]))
    np.testing.assert_array_equal(
        jnp.squeeze(jnp.array([0.0, 0.0, 0.0, 1.0])), jnp.squeeze(output[3, :]))

  @pytest.mark.parametrize('func', [random.uniform, jnp.ones, jnp.zeros])
  @pytest.mark.parametrize('sign', [-1, 1])
  @pytest.mark.parametrize('pt', SAMPLE_POINTS)
  def test_exp_se3_only_rotation(self, func, sign, pt):
    shape, _ = self._process_parameters(None, 3)
    pt = jnp.array(pt)
    w = sign * self.get_random_vector(func, shape=shape)
    v = jnp.zeros(shape=shape)
    theta = jnp.linalg.norm(w, axis=-1)
    w = w / theta[..., jnp.newaxis]
    screw_axis = jnp.concatenate([w, v], axis=-1)
    transform = rigid_body.exp_se3(screw_axis, theta)

    quat = quaternion.from_axis_angle(w, theta)
    pt_rotated = quaternion.rotate(quat, pt)

    self.assertEqual(transform.shape, (4, 4))
    pt_rotated_tf = rigid_body.from_homogenous(
        jnp.matmul(transform, rigid_body.to_homogenous(pt)))
    np.testing.assert_allclose(pt_rotated_tf, pt_rotated, atol=1E-5, rtol=1E-5)

  @pytest.mark.parametrize('func', [random.uniform, jnp.ones, jnp.zeros])
  @pytest.mark.parametrize('sign', [-1, 1])
  @pytest.mark.parametrize('pt', SAMPLE_POINTS)
  def test_exp_se3_only_translation(self, func, sign, pt):
    shape, _ = self._process_parameters(None, 3)
    w = jnp.zeros(shape=shape)
    v = sign * self.get_random_vector(func, shape=shape)
    theta = jnp.array(1)
    screw_axis = jnp.concatenate([w, v], axis=-1)
    transform = rigid_body.exp_se3(screw_axis, theta)

    pt = jnp.array(pt)
    pt_translated = pt + v

    self.assertEqual(transform.shape, (4, 4))
    pt_translated_tf = rigid_body.from_homogenous(
        jnp.matmul(transform, rigid_body.to_homogenous(pt)))
    np.testing.assert_allclose(
        pt_translated_tf, pt_translated, atol=1E-5, rtol=1E-5)

  @pytest.mark.parametrize('func', [random.uniform, jnp.ones])
  @pytest.mark.parametrize('sign1', [-1, 1])
  @pytest.mark.parametrize('sign2', [-1, 1])
  @pytest.mark.parametrize('pt', SAMPLE_POINTS)
  def test_exp_se3(self, func, sign1, sign2, pt):
    shape, _ = self._process_parameters(None, 3)
    w = sign1 * self.get_random_vector(func, shape=shape)
    v = sign2 * self.get_random_vector(func, shape=shape)
    theta = jnp.linalg.norm(w)
    w = w / theta[..., jnp.newaxis]
    v = v / theta[..., jnp.newaxis]

    screw_axis = jnp.concatenate([w, v], axis=-1)
    transform = rigid_body.exp_se3(screw_axis, theta)

    # TODO(utsinh): Figure out how this t relates to v and add a test..
    # t = jnp.squeeze(transform[0:3, 3])
    r = jnp.squeeze(transform[0:3, 0:3])
    last_row = jnp.squeeze(transform[3, :])

    # The rotation section of the matrix should be orthonormal.
    self.assertAlmostEqual(jnp.linalg.det(r).tolist(), 1, places=6)
    np.testing.assert_allclose(
        jnp.matmul(jnp.transpose(r), r), jnp.eye(3), atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        jnp.matmul(r, jnp.transpose(r)), jnp.eye(3), atol=1E-5, rtol=1E-5)

    # The last row should be [0, 0, 0, 1].
    np.testing.assert_array_equal(last_row, [0, 0, 0, 1])

    pt = jnp.array(pt)
    q = quaternion.from_axis_angle(w, theta)
    pt_transformed = quaternion.rotate(q, pt) + (v * theta)

    pt_transformed_tf = rigid_body.from_homogenous(
        jnp.matmul(transform, rigid_body.to_homogenous(pt)))
    self.assertEqual(pt_transformed.shape, (3,))
    self.assertEqual(pt_transformed_tf.shape, (3,))
    # TODO(utsinh): Make this work - this is the key assert. There's some
    #  discrepency between how to represent the screw-axis transform as a
    #  quaternion to rotate and a translation.
    # np.testing.assert_allclose(pt_transformed_tf, pt_transformed)

  def test_se3_back_and_forth_conversion(self):
    key = jax.random.PRNGKey(1)
    rotvec = jax.random.uniform(key, (5, 6))
    rotmat, trans = jax.vmap(rigid_body.se3_to_rotation_translation)(rotvec)
    self.assertEqual(rotmat.shape[-2:], (3, 3))
    rotvec_rec = jax.vmap(rigid_body.rotation_translation_to_se3)(rotmat, trans)
    self.assertEqual(rotvec_rec.shape, rotvec.shape)

