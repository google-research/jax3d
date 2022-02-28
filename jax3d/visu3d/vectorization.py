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

"""Vectorisation util."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from etils import enp
from etils import epy
from jax3d.visu3d import array_dataclass
from jax3d.visu3d import typing

_FnT = TypeVar('_FnT', bound=Callable)

# Any supported output (for now, only `Array` or `v3d.DataclassArray` supported
# but not `tuple`)
_OuputT = TypeVar('_OuputT')
_Ouput = Any


# TODO(epot): Is it possible to support `classmethod` too ? Auto-detecting
# batch shape might require assumptions, or additional argument to explitly
# set the expected shape. `@vectorize(inner_shape={'arg0': ()})`


# _FnT = Callable[..., _OuputT]
def vectorize_self(fn: _FnT) -> _FnT:
  """Vectorize a `v3d.DataclassArray` method.

  Allow to implement method in `v3d.DataclassArray` assuming `shape == ()`.

  This is similar to `jax.vmap` but:

  * Only work on `v3d.DataclassArray` methods
  * Only the `self` argument is vectorized. Other *args, **kwargs are
    passed as-is.
  * All the dataclass array batch shape are flattened.

  Example:

  ```
  @dataclasses.dataclass(frozen=True)
  class Point(v3d.DataclassArray):
    p: v3d.array_field(shape=(3,))

    @v3d.vectorize_self
    def first_value(self):
      return self.p[0]

  point = Point(p=[  # 4 points batched together
      [10, 11, 12],
      [20, 21, 22],
      [30, 31, 32],
      [40, 41, 42],
  ])
  point.first_value() == [10, 20, 30, 40]  # First value of each points
  ```

  Args:
    fn: DataclassArray method to decorate

  Returns:
    fn: Decorated function with vectorization applied to self.
  """

  @functools.wraps(fn)
  @epy.maybe_reraise(prefix=f'Error in {fn.__qualname__}: ')
  def decorated(
      self: array_dataclass.DataclassArray,
      *args: Any,
      **kwargs: Any,
  ) -> _Ouput:
    if not isinstance(self, array_dataclass.DataclassArray):
      raise TypeError(
          'v3d.vectorize_self should be applied on DataclassArray method. '
          f'Not: {type(self)}')

    if not self.shape:  # No batch shape, no-need to vectorize
      return fn(self, *args, **kwargs)

    original_shape = self.shape

    # 1. Flatten `self`
    self = self.flatten()

    # 2. Call the vectorized function
    vfn = _vmap_self(fn, xnp=self.xnp)
    out = vfn(self, *args, **kwargs)

    # 3. Unflatten the output
    out = _unflatten_tree(out, shape=original_shape)
    return out

  return decorated


@functools.lru_cache(maxsize=None)
def _vmap_self(fn: _FnT, *, xnp: enp.NpModule) -> _FnT:
  """Vectorize self using the `xnp` backend. Assume `self` was flatten."""
  if xnp is enp.lazy.np:
    return functools.partial(_vmap_self_np, fn)
  elif xnp is enp.lazy.jnp:
    return _vmap_self_jnp(fn)
  elif xnp is enp.lazy.tnp:
    return _vmap_self_tf(fn)
  raise TypeError(f'Invalid numpy module: {xnp}')


def _vmap_self_np(
    fn: _FnT,
    self: array_dataclass.DataclassArray,
    *args: Any,
    **kwargs: Any,
) -> _FnT:
  """vectorization using `np` backend."""
  # Numpy does not have vectorization, so unroll the loop
  outs = []
  for self_item in self:
    out = fn(self_item, *args, **kwargs)
    outs.append(out)

  # Stack output back together
  return _stack_tree(outs)


def _vmap_self_jnp(fn: _FnT) -> _FnT:
  """vectorization using `jax` backend."""

  # Because `jax.vmap` require static number of args to set `in_axes`,
  # we need to add indirection to collect args/kwargs.

  # Static args/kwargs
  def vfn(self, args, kwargs):
    return fn(self, *args, **kwargs)

  vfn = enp.lazy.jax.vmap(vfn, in_axes=(0, None, None))

  # Dynamic *args/**kwargs
  def apply_vfn(self, *args, **kwargs):
    return vfn(self, args, kwargs)

  return apply_vfn


def _vmap_self_tf(fn: _FnT) -> _FnT:
  """vectorization using `tf` backend."""
  # TODO(epot): Use `tf.vectorized_map()` once TF support custom nesting
  raise NotImplementedError(
      'vectorization not supported in TF yet due to lack of `tf.nest` '
      'support. Please upvote or comment b/152678472.')


def _stack_tree(vals: list[_OuputT]) -> _OuputT:
  """Stack the given tree."""
  assert vals
  val = vals[0]
  if isinstance(val, array_dataclass.DataclassArray):
    return array_dataclass.stack(vals, axis=0)
  elif enp.lazy.is_array(val):
    return enp.lazy.np.stack(vals, axis=0)
  else:
    raise NotImplementedError(f'Unsupported output type {type(val)}')


def _unflatten_tree(arrays: _OuputT, *, shape: typing.Shape) -> _OuputT:
  """Unflatten the given tree."""
  # TODO(epot): Also support non-array
  assert shape
  num_dims = enp.lazy.np.prod(shape)
  if (enp.lazy.is_array(arrays) or
      isinstance(arrays, array_dataclass.DataclassArray)):
    assert len(arrays.shape)  # `len`` because of b/198633198  # pylint: disable=g-explicit-length-test
    assert arrays.shape[0] == num_dims
    arrays = arrays.reshape(shape + arrays.shape[1:])
    return arrays
  else:
    raise NotImplementedError(f'Unsupported output type {type(arrays)}')
