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

"""Tests for inspect_utils."""

from jax3d.visu3d.utils import inspect_utils
import pytest


class A:

  def fn(self, x, y, **kwargs):
    return {
        'self': self,
        'x': x,
        'y': y,
        'kwargs': kwargs,
    }


def fn(a, *, b):
  return {'a': a, 'b': b}


def test_signature_method():
  """Test signature utils."""

  sig = inspect_utils.Signature(A.fn)

  assert sig.has_var
  assert sig.fn_name == 'A.fn'

  a = A()
  bound_args = sig.bind(a, 1, y=2, kw0=3, kw1='a')
  assert bound_args.fn_name == 'A.fn'
  assert len(bound_args) == 4

  # Access through index
  assert bound_args[0].name == 'self'
  assert bound_args[1].name == 'x'
  assert bound_args[2].name == 'y'
  assert bound_args[3].name == 'kwargs'

  assert bound_args[0].value is a
  assert bound_args[1].value == 1
  assert bound_args[2].value == 2
  assert bound_args[3].value == dict(kw0=3, kw1='a')
  # Access through str
  assert bound_args['self'].value is a
  assert bound_args['x'].value == 1
  assert bound_args['y'].value == 2
  assert bound_args['kwargs'].value == dict(kw0=3, kw1='a')

  # Self is detected
  assert bound_args.has_self
  assert bound_args.self_value is a

  assert [arg.name for arg in bound_args] == ['self', 'x', 'y', 'kwargs']

  out = bound_args.call()
  assert out == {
      'self': a,
      'x': 1,
      'y': 2,
      'kwargs': dict(kw0=3, kw1='a'),
  }


def test_signature_function():
  """Test signature utils."""

  sig = inspect_utils.Signature(fn)

  assert not sig.has_var
  assert sig.fn_name == 'fn'

  bound_args = sig.bind(1, b=2)
  assert bound_args.fn_name == 'fn'
  assert len(bound_args) == 2

  # Access through index
  assert bound_args[0].value == 1
  assert bound_args[1].value == 2
  # Access through str
  assert bound_args['a'].value == 1
  assert bound_args['b'].value == 2

  # Self is detected
  assert not bound_args.has_self

  with pytest.raises(ValueError, match='does not have `self`'):
    _ = bound_args.self_value

  out = bound_args.call()
  assert out == {
      'a': 1,
      'b': 2,
  }

  def _other_fn(*args, **kwargs):
    y = fn(*args, **kwargs)
    return {f'_{k}': v for k, v in y.items()}

  out = bound_args.call(_other_fn)
  assert out == {
      '_a': 1,
      '_b': 2,
  }

  bound_args = bound_args.map(lambda x: x * 10)
  out = bound_args.call()
  assert out == {
      'a': 10,
      'b': 20,
  }
