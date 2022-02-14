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

"""Test utils."""

import functools
from typing import Any, Callable

import chex
from jax3d.projects.nesf.utils import np_utils
from jax3d.projects.nesf.utils.typing import Tree
import numpy as np


def assert_tree_all(
    fn: Callable[..., Any],
    *trees: Tree[Any],
) -> None:
  """`chex.assert_tree_all_equal_comparator` with automatic error message.

  ```
  jax3d.projects.nesf.testing.assert_tree_all(
  jnp.allclose, params0, params1)
  ```

  Args:
    fn: Comparator function
    *trees: Nested trees on which appy the function

  Returns:
    None
  """
  return chex.assert_tree_all_equal_comparator(
      fn,
      lambda x, y: f'Got: {fn}({x}, {y})',
      *trees,
      # By default chex raise error if Tree contain None. Unclear why.
      ignore_nones=True,
  )


def assert_tree_all_equal_spec(
    *trees: Tree[Any],
) -> None:
  """Check that arrays in the given trees have the same `dtype`/`shape`."""
  return chex.assert_tree_all_equal_comparator(
      lambda x, y: x.shape == y.shape and x.dtype == y.dtype,
      lambda x, y: f'{_repr_spec(x)} != {_repr_spec(y)}',
      *trees,
      # By default chex raise error if Tree contain None. Unclear why.
      ignore_nones=True,
  )


def _compare_array(x, y, *, return_err: bool, **kwargs):
  """Comparte 2 arrays."""
  err_msg = 'Error in value equality check: Values not approximately equal'

  try:
    if np_utils.is_array_str(x):
      # str arrays can't be compared with `assert_allclose`
      np.testing.assert_equal(x, y)
    else:
      np.testing.assert_allclose(x, y, err_msg=err_msg, **kwargs)
  except AssertionError as e:
    if return_err:
      return str(e)
    else:
      return False
  else:
    if return_err:
      return ''
    else:
      return True


def assert_trees_all_close(
    *trees: Tree[Any],
    **kwargs
) -> None:
  """Assert that 2 trees are close, but also works for `str` arrays."""
  chex.assert_tree_all_equal_comparator(
      functools.partial(_compare_array, return_err=False, **kwargs),
      functools.partial(_compare_array, return_err=True, **kwargs),
      *trees,
      # By default chex raise error if Tree contain None. Unclear why.
      ignore_nones=True,
  )


def _repr_spec(arr) -> str:
  """Returns the Spec repr string of the given tensor."""
  return f'{type(arr).__qualname__}(shape={arr.shape}, dtype={arr.dtype})'
