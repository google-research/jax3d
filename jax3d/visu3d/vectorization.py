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

"""Vectorization util."""

from __future__ import annotations

import functools
# import typing
from typing import Any, Callable, Optional, Set, TypeVar

from etils import enp
from etils import epy
from jax3d.visu3d import array_dataclass
from jax3d.visu3d import np_utils
from jax3d.visu3d.typing import DcOrArray, Shape  # pylint: disable=g-multiple-import
from jax3d.visu3d.utils import inspect_utils

_FnT = TypeVar('_FnT', bound=Callable)

# Any supported output (for now, only `Array` or `v3d.DataclassArray` supported
# but not `tuple`)
_OutT = TypeVar('_OutT')
_Out = Any


# Metadata specifying which argument should be concidered static:
# @v3d.vectorize_method(static_args={'arg0'})
_StaticArgInfo = Optional[Set[str]]
# pyformat: disable
_MapNonStatic = Callable[
    [Callable, inspect_utils.BoundArgs],  # TODO(epot): Complete types
    inspect_utils.BoundArgs,
]
# pyformat: enable


# TODO(epot): Is it possible to support `classmethod` too ? Auto-detecting
# batch shape might require assumptions, or additional argument to explitly
# set the expected shape. `@vectorize(inner_shape={'arg0': ()})`


# TODO(pytype): Pytype crash when using `@overload` with `TypeVar`

# @typing.overload
# def vectorize_method(
#     fn: None = ...,
#     *,
#     static_args: _StaticArgInfo = ...,
# ) -> Callable[[_FnT], _FnT]:
#   ...


# # _FnT = Callable[..., _OutT]
# @typing.overload
# def vectorize_method(
#     fn: _FnT,
#     *,
#     static_args: _StaticArgInfo = ...,
# ) -> _FnT:
#   ...


def vectorize_method(
    fn=None,
    *,
    static_args=None,
):
  """Vectorize a `v3d.DataclassArray` method.

  Allow to implement method in `v3d.DataclassArray` assuming `shape == ()`.

  This is similar to `jax.vmap` but:

  * Only work on `v3d.DataclassArray` methods
  * Instead of vectorizing a single axis, `@v3d.vectorize_method` will vectorize
    over `*self.shape` (not just `self.shape[0]`). This is like if `vmap`
    was applied to `self.flatten()`
  * Axis with dimension `1` are brodcasted.

  For example, with `__matmul__(self, x: T) -> T`:

  ```python
  () @ (*x,) -> (*x,)
  (b,) @ (b, *x) -> (b, *x)
  (b,) @ (1, *x) -> (b, *x)
  (1,) @ (b, *x) -> (b, *x)
  (b, h, w) @ (b, h, w, *x) -> (b, h, w, *x)
  (1, h, w) @ (b, 1, 1, *x) -> (b, h, w, *x)
  ```

  Example:

  ```
  @dataclasses.dataclass(frozen=True)
  class Point(v3d.DataclassArray):
    p: v3d.array_field(shape=(3,))

    @v3d.vectorize_method
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
    static_args: If given, should be a set of the static argument names

  Returns:
    fn: Decorated function with vectorization applied to self.
  """
  # Called as decorator with options (`@v3d.vectorize_method(**options)`)
  if fn is None:
    return functools.partial(vectorize_method, static_args=static_args)

  # Signature util also make sure explicit error message are raised (e.g.
  # `Error in <fn> for arg <arg-name>` )
  sig = inspect_utils.Signature(fn)
  if sig.has_var:
    raise NotImplementedError(
        '`@v3d.vectorize_method` does not support function with variable args '
        f'(`*args` or `**kwargs`). For {sig.fn_name}. Please open an issue.')

  if static_args is not None:
    if not isinstance(static_args, set):
      raise TypeError(
          f'Unexpected `static_args={static_args!r}`. Expected `set`.')
  map_non_static = functools.partial(
      _map_non_static,
      static_args=static_args,
  )

  @functools.wraps(fn)
  @epy.maybe_reraise(prefix=lambda: f'Error in {fn.__qualname__}: ')
  def decorated(
      self: array_dataclass.DataclassArray,
      *args: Any,
      **kwargs: Any,
  ) -> _Out:
    if not isinstance(self, array_dataclass.DataclassArray):
      raise TypeError(
          'v3d.vectorize_method should be applied on DataclassArray method. '
          f'Not: {type(self)}')

    if not self.shape:  # No batch shape, no-need to vectorize
      return fn(self, *args, **kwargs)

    original_args = sig.bind(self, *args, **kwargs)

    # TODO(epot): Tree support (with masking inside args)

    # Validation
    # TODO(epot): Normalize `np`, `list` -> `xnp`
    assert_is_array = functools.partial(_assert_is_array, xnp=self.xnp)
    map_non_static(assert_is_array, original_args)

    # Broadcast and flatten args. Exemple:
    # Broadcast the batch shape when dim == 1:
    # (h, w), (h, w, c) -> (h, w), (h, w, c)
    # (h, w), (1, 1, c) -> (h, w), (h, w, c)
    # (1, 1), (h, w, c) -> (h, w), (h, w, c)
    # Flatten:
    # (h, w), (h, w, c) -> (b*h*w,), (b*h*w, c)
    flat_args, batch_shape = _broadcast_and_flatten_args(
        original_args,
        map_non_static=map_non_static,
    )

    # Call the vectorized function
    out = _vmap_method(
        flat_args,
        map_non_static=map_non_static,
        xnp=self.xnp,
    )

    # Unflatten the output
    out = _unflatten_tree(out, batch_shape=batch_shape)
    return out

  return decorated


def _broadcast_and_flatten_args(
    args: inspect_utils.BoundArgs[DcOrArray, DcOrArray],
    *,
    map_non_static: _MapNonStatic,
) -> tuple[inspect_utils.BoundArgs[DcOrArray, DcOrArray], Shape]:
  """Normalize the output to prepare for the vectorization."""
  assert args.has_self

  xnp = args.self_value.xnp
  batch_shape = args.self_value.shape
  assert batch_shape

  # 1. Compute the final batch shape
  def _collect_batch_shape(array: DcOrArray) -> None:
    # Validate and update the global broadcast shape
    # e.g.
    # _update_batch_shape((1, x1), (x0, 1, ...)) == (x0, x1)
    # _update_batch_shape((x0, x1), (x0, x1, ...)) == (x0, x1)
    # _update_batch_shape((1, 1), (x0, x1, ...)) == (x0, x1)
    nonlocal batch_shape
    batch_shape = _update_batch_shape(batch_shape, array.shape)

  map_non_static(_collect_batch_shape, args)

  # 2. Broadcast args
  broacast_array_fn = functools.partial(
      _broacast_and_flatten_to,
      batch_shape=batch_shape,
      xnp=xnp,
  )
  flat_args = map_non_static(broacast_array_fn, args)
  return flat_args, batch_shape


def _assert_is_array(array: DcOrArray, *, xnp: enp.NpModule) -> None:
  """Validate the value is an array."""
  if not np_utils.is_array(array):
    raise TypeError(f'Expected `v3d.DataclassArray` or `xnp.ndarray`. '
                    f'Got: {type(array)}')
  array_xnp = np_utils.get_xnp(array)
  if array_xnp is not xnp:
    raise ValueError(f'Expected {xnp.__name__}, got {array_xnp.__name__}')


def _update_batch_shape(batch_shape: Shape, shape: Shape) -> Shape:
  """Compute the new batch shape.

  ```
  _update_batch_shape(batch_shape=(x0, x1), shape=(1, 1, ...)) == (x0, x1)
  _update_batch_shape(batch_shape=(x0, x1), shape=(x0, 1, ...)) == (x0, x1)
  _update_batch_shape(batch_shape=(1, x1), shape=(x0, 1, ...)) == (x0, x1)
  _update_batch_shape(batch_shape=(x0, x1), shape=(x0, x1, ...)) == (x0, x1)
  ```

  Args:
    batch_shape: Current target shape
    shape: Other shape

  Returns:
    New target shape
  """
  if len(shape) < len(batch_shape):
    raise ValueError(
        f'Cannot vectorize shapes {shape} with {batch_shape}. '
        f'Shape should start with {batch_shape}, {(1,) * len(batch_shape)} '
        'or compatible shape.')
  new_batch_shape = []
  for arr_dim, target_dim in zip(shape, batch_shape):
    if arr_dim == target_dim:
      new_batch_shape.append(target_dim)
    elif arr_dim == 1 or target_dim == 1:
      new_batch_shape.append(arr_dim * target_dim)
    else:
      raise ValueError(f'Cannot vectorize shapes {shape} with {batch_shape}. '
                       f'Incompatible dim {arr_dim} != {target_dim}')

  # Update the batch shape
  return tuple(new_batch_shape)


def _broacast_and_flatten_to(
    array: DcOrArray,
    *,
    batch_shape: Shape,
    xnp: enp.NpModule,
) -> DcOrArray:
  """Apply broadcast and flatten op to the array/dataclass array."""
  inner_shape = array.shape[len(batch_shape):]
  final_shape = batch_shape + inner_shape
  if isinstance(array, array_dataclass.DataclassArray):
    array = array.broadcast_to(final_shape)
  elif isinstance(array, xnp.ndarray):
    array = xnp.broadcast_to(array, final_shape)
  else:
    raise TypeError(f'Unexpected array type: {type(array)}')
  return array.reshape((np_utils.size_of(batch_shape),) + inner_shape)


def _vmap_method(
    args: inspect_utils.BoundArgs,
    *,
    map_non_static: _MapNonStatic,
    xnp: enp.NpModule,
) -> _Out:
  """Vectorize self using the `xnp` backend. Assume `self` was flatten."""
  if xnp is enp.lazy.np:
    return _vmap_method_np(args, map_non_static=map_non_static)
  elif xnp is enp.lazy.jnp:
    return _vmap_method_jnp(args, map_non_static=map_non_static)
  elif xnp is enp.lazy.tnp:
    return _vmap_method_tf(args, map_non_static=map_non_static)
  raise TypeError(f'Invalid numpy module: {xnp}')


def _vmap_method_np(
    args: inspect_utils.BoundArgs[Any, _OutT],
    *,
    map_non_static: _MapNonStatic,
) -> _OutT:
  """vectorization using `np` backend."""
  # Numpy does not have vectorization, so unroll the loop
  outs = []
  for i in range(len(args.self_value)):  # Iterate over the first dimension
    args_slice = map_non_static(lambda x: x[i], args)  # pylint: disable=cell-var-from-loop
    out = args_slice.call()  # out = fn(self, *args, **kwargs)
    outs.append(out)

  # Stack output back together
  return _stack_tree(outs)


def _vmap_method_jnp(
    args: inspect_utils.BoundArgs[Any, _OutT],
    *,
    map_non_static: _MapNonStatic,
) -> _OutT:
  """vectorization using `jax` backend."""

  # Compute the signature static/in_axes
  # All axis are static...
  in_axes_args = args.map(lambda _: None)
  # ... except the non-static ones
  in_axes_args = map_non_static(lambda _: 0, in_axes_args)
  in_axes = tuple(arg.value for arg in in_axes_args)

  # Vectorize self and args
  vfn = _vmap_cached(args.fn, in_axes=in_axes)

  # Call `vfn(self, *args, **kwargs)`
  return args.call(vfn)


@functools.lru_cache(maxsize=None)
def _vmap_cached(fn: _FnT, *, in_axes) -> _FnT:
  """Like `jax.vmap` but cache the function."""
  return enp.lazy.jax.vmap(
      fn,
      in_axes=in_axes,
  )


def _vmap_method_tf(
    args: inspect_utils.BoundArgs[Any, _OutT],
    *,
    map_non_static: _MapNonStatic,
) -> _OutT:
  """vectorization using `tf` backend."""
  # TODO(epot): Use `tf.vectorized_map()` once TF support custom nesting
  raise NotImplementedError(
      'vectorization not supported in TF yet due to lack of `tf.nest` '
      'support. Please upvote or comment b/152678472.')


def _stack_tree(vals: list[_OutT]) -> _OutT:
  """Stack the given tree."""
  assert vals
  val = vals[0]
  if isinstance(val, array_dataclass.DataclassArray):
    return array_dataclass.stack(vals, axis=0)
  elif enp.lazy.is_array(val):
    return enp.lazy.np.stack(vals, axis=0)
  else:
    raise NotImplementedError(
        f'Unsupported output type {type(val)}. Only array or dataclass '
        'array supported for now.')


def _unflatten_tree(arrays: _OutT, *, batch_shape: Shape) -> _OutT:
  """Unflatten the given tree."""
  # TODO(epot): Also support non-array
  assert batch_shape
  batch_size = np_utils.size_of(batch_shape)
  if (enp.lazy.is_array(arrays) or
      isinstance(arrays, array_dataclass.DataclassArray)):
    assert len(arrays.shape)  # `len`` because of b/198633198  # pylint: disable=g-explicit-length-test
    assert arrays.shape[0] == batch_size
    arrays = arrays.reshape(batch_shape + arrays.shape[1:])
    return arrays
  else:
    raise NotImplementedError(
        f'Unsupported output type {type(arrays)}. Only array or dataclass '
        'array supported for now.')


def _map_non_static(
    fn: Callable[..., _Out],
    bound_args: inspect_utils.BoundArgs,
    *,
    static_args: _StaticArgInfo,
):
  """Call `bound_args.map` but without the static args."""

  def fn_without_static(arg: inspect_utils.BoundArg):
    # Argument is static. Forward as-is
    if static_args and arg.name in static_args:
      return arg.value
    else:
      return fn(arg.value)

  return bound_args.map_bound_arg(fn_without_static)
