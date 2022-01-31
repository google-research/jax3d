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

import jax3d.public_api as j3d


def test_decorator_with_option():

  @j3d.utils.decorator_with_option
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


def test_cached_property():

  @dataclasses.dataclass
  class A:
    x: int
    counter: int = 0

    @j3d.utils.cached_property
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
