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

"""Tests for jax3d.utils.shape_validations."""

import einops
from etils.array_types import Array, f32, i32  # pylint: disable=g-multiple-import
import jax.numpy as jnp
import jax3d.public_api as j3d
from jax3d.utils import shape_validation
import pytest


@pytest.fixture
def shapeguard_scope():
  with shape_validation._ShapeTracker.track():
    yield


def check(spec, array):
  shape_validation.assert_match_array_alias(array=array, expected_spec=spec)


@pytest.mark.usefixtures('shapeguard_scope')
def test_assert_type():
  # Scalar int and float are accepted as-is
  check(i32[''], 12)
  check(f32[''], 12.)

  with pytest.raises(ValueError, match='Dtype do not match'):
    check(i32[''], 12.)

  with pytest.raises(ValueError, match='Dtype do not match'):
    check(f32[''], 12)

  # Number of dimensions should match and new dimensions are registered
  check(f32[''], jnp.zeros(()))
  check(f32['b'], jnp.zeros((2,)))
  check(f32['b h w'], jnp.zeros((2, 3, 4)))

  with pytest.raises(ValueError, match='Rank should be the same.'):
    check(f32['b'], jnp.zeros((2, 3)))

  # Recorded dimension values should be constant
  check(f32['w b h'], jnp.zeros((4, 2, 3)))

  with pytest.raises(ValueError, match='Expected b=2.'):
    check(f32['b'], jnp.zeros((3,)))

  with pytest.raises(ValueError, match='Expected w=4.'):
    check(f32['b h w'], jnp.zeros((
        2,
        3,
        10,
    )))

  with pytest.raises(TypeError, match='Expected .* array'):
    check(f32['b h w'], dict())

  # TODO(epot): There should be a cleaner API to check shapes values
  assert shape_validation._ShapeTracker.current().resolve_spec(
      'b h w') == '2 3 4'


@pytest.mark.usefixtures('shapeguard_scope')
def test_assert_type_incomplete():

  # No dtype and no shape
  check(Array, 12)
  check(Array, jnp.zeros((1, 2, 3)))
  with pytest.raises(TypeError, match='Expected .* array'):
    check(Array, {})

  # Shape but no dtype
  check(Array['h w c'], jnp.zeros((1, 2, 3)))

  with pytest.raises(ValueError, match='Rank should be the same.'):
    check(Array['h w c'], jnp.zeros((1, 2)))

  # Dtype but no shape
  check(f32, jnp.zeros((1, 2, 3)))

  with pytest.raises(ValueError, match='Dtype do not match'):
    check(f32, jnp.zeros((1, 2, 3), dtype=jnp.int32))


def test_assert_type_outside_scope():

  with pytest.raises(AssertionError, match='Calling .* from outside .* scope'):
    check(f32[''], jnp.zeros((3,)))


def test_shape_valid():

  @j3d.assert_typing
  def fn(x: f32['b h w c'], y: f32['']) -> f32['b h w']:
    return (x + y).mean(axis=-1)

  # 2 independent function calls can have different dimensions
  fn(jnp.zeros((1, 2, 3, 4)), 4.)
  fn(jnp.zeros((5, 6, 1, 2)), 7.)

  with pytest.raises(ValueError, match='Rank should be the same.'):
    fn(jnp.zeros((5, 6, 1)), 7.)

  with pytest.raises(ValueError, match='Dtype do not match'):
    fn(jnp.zeros((5, 6, 1, 2)), 7)


def test_shape_valid_args():

  @j3d.assert_typing
  def fn(x: f32['b h'], y: f32['b w']) -> None:
    del x, y
    return

  fn(jnp.zeros((1, 3)), jnp.zeros((1, 2)))

  with pytest.raises(ValueError, match='Expected b=1.'):
    fn(jnp.zeros((1, 3)), jnp.zeros((2, 2)))  # Inconsistent batch size


def test_shape_valid_inner():

  @j3d.assert_typing
  def fn(x: f32['b l']) -> f32['l b']:
    x = einops.rearrange(x, 'b l -> l b')
    check(f32['l b'], x)

    with pytest.raises(ValueError, match='Expected b=1.'):
      check(f32['b l'], x)
    return x

  assert fn(jnp.zeros((1, 2))).shape == (2, 1)


def test_shape_valid_nested():

  @j3d.assert_typing
  def fn(x: f32['h w c'], nest: bool = True) -> i32['']:
    if nest:

      # Nested call should also trigger error
      with pytest.raises(ValueError, match='Rank should be the same.'):
        fn(jnp.zeros((5,)), nest=False)

      # Nested calls use another scope, so dimensions do not need to match
      return 1 + fn(einops.rearrange(x, 'h w c -> c h w'), nest=False)
    else:
      return 1

  assert fn(jnp.zeros((1, 2, 3))) == 2
  assert fn(jnp.zeros((1, 2, 3)), nest=False) == 1


def test_shape_valid_bad_return_type():

  @j3d.assert_typing
  def fn(x: f32['batch length'], rearrange: str) -> f32['length batch']:
    return einops.rearrange(x, rearrange)

  assert fn(jnp.zeros((1, 2)), 'b l -> l b').shape == (2, 1)

  with pytest.raises(ValueError, match='Rank should be the same.'):
    fn(jnp.zeros((1, 2)), 'b l -> (l b)')


def test_shape_valid_args_kwargs():

  @j3d.assert_typing
  def fn(*args: f32['b'], **kwargs: f32['']) -> int:
    return len(args) + len(kwargs)

  assert fn() == 0
  assert fn(jnp.zeros((1,)), jnp.zeros((1,)), a=4., b=jnp.array(3.)) == 4

  with pytest.raises(ValueError, match='Expected b=1.'):
    fn(jnp.zeros((1,)), jnp.zeros((2,)))

  with pytest.raises(ValueError, match='Rank should be the same.'):
    fn(a=1., b=jnp.zeros((2,)))
