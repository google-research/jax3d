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

"""Tests for jax3d.utils.shape_utils."""

from etils.array_types import f32, i32  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
import jax3d.public_api as j3d
from jax3d.utils import shape_utils
import numpy as np
import pytest
import tensorflow as tf


def test_shape_utils():

  ones = [
      # tf.data.Dataset
      tf.data.Dataset.from_generator(lambda: (), output_signature={
          'b': tf.TensorSpec((2,), dtype=tf.int32),
          'c': tf.TensorSpec((3,), dtype=tf.float32),
      }),
      # tf.TensorSpec
      tf.TensorSpec((4,), dtype=tf.int32),
      # tf.Tensor
      tf.constant([1, 1], dtype=tf.float32),
      # jnp.array
      jnp.ones((5,), dtype=jnp.bool_),
      # jax.ShapeDtypeStruct
      jax.ShapeDtypeStruct((6,), dtype=np.int32),
      # Array alias
      f32[1, 2],
      # np.array
      {'a': np.ones((7,), dtype=np.float32)},
      # tf.NoneTensorSpec
      shape_utils._get_none_spec(),
      None,
  ]

  tensor_specs = j3d.tensor_spec_like(ones + [
      # tf.string tensor is only supported in TF, so only test it for
      # tensor_spec_like
      np.array(['abc', 'def', ''], dtype=object),
      np.array(['abc', 'def']),
  ])
  assert tensor_specs == [
      {
          'b': tf.TensorSpec((2,), dtype=tf.int32),
          'c': tf.TensorSpec((3,), dtype=tf.float32),
      },
      tf.TensorSpec((4,), dtype=tf.int32),
      tf.TensorSpec((2,), dtype=tf.float32),
      tf.TensorSpec((5,), dtype=tf.bool),
      tf.TensorSpec((6,), dtype=tf.int32),
      tf.TensorSpec((1, 2), dtype=tf.float32),
      {'a': tf.TensorSpec((7,))},
      shape_utils._get_none_spec(),
      shape_utils._get_none_spec(),
      tf.TensorSpec((3,), dtype=tf.string),
      tf.TensorSpec((2,), dtype=tf.string),
  ]

  zeros = j3d.zeros_like(ones)
  expected_zeros = [
      {
          'b': jnp.zeros((2,), dtype=np.int32),
          'c': jnp.zeros((3,), dtype=np.float32),
      },
      jnp.zeros((4,), dtype=np.int32),
      jnp.zeros((2,), dtype=np.float32),
      jnp.zeros((5,), dtype=np.bool_),
      jnp.zeros((6,), dtype=np.int32),
      jnp.zeros((1, 2), dtype=np.float32),
      {'a': jnp.zeros((7,))},
      None,
      None,
  ]
  j3d.testing.assert_tree_all_equal_spec(zeros, expected_zeros)
  j3d.testing.assert_tree_all(jnp.allclose, zeros, expected_zeros)

  with pytest.raises(TypeError, match='Unknown array-like type'):
    j3d.zeros_like([3])


def test_shape_dtype_like():
  arr_spec = j3d.shape_dtype_like([{'a': jnp.zeros((2, 1), dtype=np.int32)}])
  assert repr(arr_spec) == "[{'a': i32[2 1]}]"


def test_type_like():
  arr_spec = j3d.types_like([{'a': jnp.zeros((2, 1), dtype=np.int32)}])
  assert arr_spec == [{'a': i32[2, 1]}]


def test_standardized():
  tree_struct = {
      'a': jnp.zeros((2, 1), dtype=np.int32),
      'b': 'some non-array value',
  }

  with pytest.raises(TypeError, match='Unknown array-like type'):
    j3d.types_like(tree_struct)

  assert j3d.types_like(tree_struct, skip_non_arrays=True) == {  # pytype: disable=wrong-keyword-args
      'a': i32[2, 1],
      'b': 'some non-array value',
  }
