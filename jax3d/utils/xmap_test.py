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

"""Tests for jax3d.utils.xmap."""

import inspect
import jax
import jax.numpy as jnp
import jax3d.public_api as j3d
from jax3d.utils import xmap as xmap_lib

import pytest


def test_xmap():

  @j3d.xmap(['b ...', '... b'], 'b ...')
  def add(x, y, scale=1):
    assert x.shape == (8,)  # x and y are vectorized
    assert y.shape == (8,)
    return (x + y) * scale

  # add(x, y)
  y = add(jnp.ones((16, 8)), jnp.ones((8, 16)))
  assert jnp.allclose(y, jnp.ones((16, 8)) * 2)

  # add(y=y, x=x)
  y = add(y=jnp.ones((8, 16)), x=jnp.ones((16, 8)))
  assert jnp.allclose(y, jnp.ones((16, 8)) * 2)

  # add(x, scale=-1, y=y)
  y = add(jnp.ones((16, 8)), scale=-1, y=jnp.ones((8, 16)))
  assert jnp.allclose(y, jnp.ones((16, 8)) * -2)


def test_xmap_real_examples():

  # Example 1
  @j3d.xmap(['b n k', 'k m'], 'b n m')
  def batch_matrix_single_matrix(x, y):
    return jnp.einsum('{n,b,k},{k,m}->{n,b,m}', x, y)

  x = jnp.ones((20, 5, 7))
  y = jnp.ones((7, 11))
  assert jnp.allclose(
      batch_matrix_single_matrix(x, y),
      jnp.einsum('bnk,km->bnm', x, y)
  )

  # Example 2
  grid_sample = j3d.xmap(
      jax.scipy.ndimage.map_coordinates,
      in_axes=['b ... c', 'b _ _'],
      out_axes='b _ c',
  )
  # (b h w c) == (2 2 2 1)
  imgs = jnp.array([
      [[1, 11], [2, 22]],  # img0
      [[3, 33], [4, 44]],  # img1
  ])[..., None]
  # b 2 n == (2 2 3)
  coordinates = jnp.array([
      [[0, 1, 0], [0, 0, 1]],  # coordinates in img0
      [[0, 1, 0], [0, 1, 1]],  # coordinates in img1
  ])
  # ['b h w c', 'b 2 n'] -> 'b n c'
  result = grid_sample(imgs, coordinates, order=0, mode='wrap')
  expected_result = jnp.array([
      [[1], [2], [11]],
      [[3], [44], [33]],
  ])
  assert  jnp.allclose(result, expected_result)


@pytest.mark.parametrize(
    ['spec_str', 'spec_dict'],
    [
        ('batch ...', {0: 'batch'}),
        ('b ...', {0: 'b'}),
        ('b c ...', {0: 'b', 1: 'c'}),
        ('b h w c', {0: 'b', 1: 'h', 2: 'w', 3: 'c'}),
        ('b _ _ c', {0: 'b', 3: 'c'}),
        ('... c', {-1: 'c'}),
        ('b ... c', {0: 'b', -1: 'c'}),
        ('... w c', {-2: 'w', -1: 'c'}),
        ('b ... w c', {0: 'b', -2: 'w', -1: 'c'}),
        ('_ b ... w _ c _', {1: 'b', -4: 'w', -2: 'c'}),
        ('...', {}),
    ]
)
def test_str_to_dict_shape_spec_valid(spec_str, spec_dict):
  assert xmap_lib._str_to_dict_shape_spec(spec_str) == spec_dict


@pytest.mark.parametrize(
    ['spec_str'],
    [
        ('b ... ...',),
    ]
)
def test_str_to_dict_shape_spec_invalid(spec_str):
  with pytest.raises(ValueError, match='Invalid format'):
    assert xmap_lib._str_to_dict_shape_spec(spec_str)


def test_normalize_dict_shape():
  assert xmap_lib._normalize_dict_shape(
      jnp.zeros((2, 2, 2)),
      {-3: 'a', 1: 'b', -1: 'c'},
  ) == {0: 'a', 1: 'b', 2: 'c'}


@pytest.mark.parametrize(
    ['num_arr_args', 'args', 'kwargs', 'expected_arr_args'],
    [
        (1, ('a', 'b'), {}, ('a',)),  # f(a, b)
        (2, ('a', 'b'), {}, ('a', 'b')),  # f(a, b)
        (1, ('a',), {'y': 'b'}, ('a',)),  # f(a, y=b)
        (2, ('a',), {'y': 'b'}, ('a', 'b')),  # f(a, y=b)
        (1, (), {'y': 'b', 'x': 'a'}, ('a',)),  # f(y=b, x=a)
        (2, (), {'y': 'b', 'x': 'a'}, ('a', 'b')),  # f(y=b, x=a)
    ]
)
def test_split_static_args(num_arr_args, args, kwargs, expected_arr_args):
  def fn(x, y=None, is_train=None):  # pylint: disable=unused-argument
    pass

  sig = inspect.signature(fn)

  arr_args, merge_fn = xmap_lib._split_static_args(
      fn, num_arr_args, args, kwargs
  )
  assert arr_args == expected_arr_args
  merged_args, merged_kwargs = merge_fn(arr_args)
  # TODO(jax3d): Should test that merged_args, merged_kwargs are actually
  # updated with arr_args by applying some transformation on arr_args
  assert sig.bind(*args, **kwargs) == sig.bind(*merged_args, **merged_kwargs)
