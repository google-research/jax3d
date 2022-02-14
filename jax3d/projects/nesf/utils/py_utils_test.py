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

"""Tests for jax3d.utils.py_utils."""

import dataclasses
import functools

import jax3d.projects.nesf as jax3d
import pytest
import tensorflow as tf


def test_decorator_with_option():

  @jax3d.utils.decorator_with_option
  def forward_args(fn, x=1, y=10):
    return functools.partial(fn, x, y)

  # Decorator called with options
  @forward_args(2, y=20)
  def get_args(x, y):
    return (x, y)

  assert get_args() == (2, 20)  # pylint: disable=no-value-for-parameter

  # Decorator called without options
  @forward_args
  def get_args(x, y):  # pylint: disable=function-redefined
    return (x, y)

  assert get_args() == (1, 10)  # pylint: disable=no-value-for-parameter

  # Decorator called directly
  get_args = forward_args(lambda x, y: (x, y), 3, y=30)

  assert get_args() == (3, 30)  # pylint: disable=not-callable


def test_reraise():

  class CustomError(Exception):

    def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
      pass  # Do not call super() to ensure this would work with bad code.

  with pytest.raises(ValueError, match='Caught: '):
    with jax3d.utils.try_reraise('Caught: '):
      raise ValueError

  with pytest.raises(ValueError, match='Caught: With message'):
    with jax3d.utils.try_reraise('Caught: '):
      raise ValueError('With message')

  with pytest.raises(CustomError, match='123\nCaught!'):
    with jax3d.utils.try_reraise(suffix='Caught!'):
      raise CustomError(123)

  with pytest.raises(CustomError, match="('Caught: ', 123, {})"):
    with jax3d.utils.try_reraise(lambda: 'Caught: '):
      raise CustomError(123, {})

  with pytest.raises(Exception, match='Caught: '):
    with jax3d.utils.try_reraise('Caught: '):
      ex = CustomError(123, {})
      ex.args = 'Not a tuple'
      raise ex

  with pytest.raises(
      tf.errors.FailedPreconditionError, match='Caught: message'):
    with jax3d.utils.try_reraise('Caught: '):
      raise tf.errors.FailedPreconditionError(None, None, 'message')

  try:
    with jax3d.utils.try_reraise('Caught2: '):
      with jax3d.utils.try_reraise('Caught: '):
        e_origin = tf.errors.FailedPreconditionError(None, None, 'message')
        raise e_origin
  except tf.errors.FailedPreconditionError as e:
    assert isinstance(e, tf.errors.FailedPreconditionError)
    assert 'Caught2: Caught: message' in str(e)
    assert 'Caught2: Caught: message' in repr(e)
    # Attributes are correctly forwarded
    assert e.node_def is None
    # Only a single cause is set (so avoid nested context)
    assert e.__cause__ is None
    assert e.__context__ is e_origin
  else:
    raise ValueError('Exception not catched')

  with pytest.raises(ImportError, match='Caught: With message'):
    with jax3d.utils.try_reraise('Caught: '):
      raise ImportError('With message')

  with pytest.raises(FileNotFoundError, match='Caught: With message'):
    with jax3d.utils.try_reraise('Caught: '):
      raise FileNotFoundError('With message')


def test_cached_property():

  @dataclasses.dataclass
  class A:
    x: int
    counter: int = 0

    @jax3d.utils.cached_property
    def y(self):
      self.counter += 1
      return self.x * 10

  a = A(x=1)
  assert a.counter == 0
  assert a.y == 10  # pylint: disable=comparison-with-callable
  assert a.y == 10  # pylint: disable=comparison-with-callable
  a.x = 2  # Even after modifying x, y is still cached.
  assert a.y == 10  # pylint: disable=comparison-with-callable
  assert a.counter == 1
