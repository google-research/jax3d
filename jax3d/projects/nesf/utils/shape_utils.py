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

"""Shape conversion utils."""

import functools
from typing import Any, Callable, TypeVar, Union

import jax
import jax.numpy as jnp
from jax3d.projects.nesf import typing as jtyping
from jax3d.projects.nesf.utils import jax_utils
from jax3d.projects.nesf.utils import np_utils
from jax3d.projects.nesf.utils.typing import Array, Tensor, Tree  # pylint: disable=g-multiple-import
import numpy as np
import tensorflow as tf
from typing_extensions import Final  # pylint: disable=g-multiple-import  # pytype: disable=not-supported-yet


_ArrayInput = Union[
    Tensor,
    tf.TensorSpec,
    jax.ShapeDtypeStruct,
    Array,
]
_T1 = TypeVar(
    '_T1',
    tf.data.Dataset,
    Tensor,
    tf.TensorSpec,
    jax.ShapeDtypeStruct,
    Array,
)
_T2 = TypeVar('_T2')


class _UnknownType:
  pass


_UNKNOWN_TYPE: Final = _UnknownType


def _tree_map(
    fn: Callable[[_T1], _T2],
) -> Callable[[Tree[_T1]], Tree[_T2]]:  # pytype: disable=invalid-annotation
  """Decorator which wraps the function inside `jax.tree_map`.

  Additionally, it also recurses into `tf.data.Dataset` (into the inner
  `ds.element_spec`).

  Args:
    fn: Function with signature `(x) -> y`

  Returns:
    The returned function now accept/returns nested tree of input.
  """
  def _recurse_or_apply(array, **kwargs):
    """Apply `fn` or recurse into `tf.data.Dataset`."""
    if isinstance(array, tf.data.Dataset):
      return jax.tree_map(functools.partial(fn, **kwargs), array.element_spec)
    else:
      return fn(array, **kwargs)

  @functools.wraps(fn)
  def fn_with_tree_map(arrays, **kwargs):
    return jax.tree_map(functools.partial(_recurse_or_apply, **kwargs), arrays)

  return fn_with_tree_map


def _standardize_array(
    fn: Callable[[jax.ShapedArray], _T2],
) -> Callable[[_ArrayInput], _T2]:
  """Decorator which standardize input array.

  Args:
    fn: Function which only normalized array (with signature
      `fn(arr: jax.ShapedArray) -> Any`)

  Returns:
    fn: Function now accept any array-like (jnp, np, tf.Tensor,...).
      The function also has an additional `skip_non_arrays` kwarg. If `True`,
      the function forwards as-is non-array values. This allow to inspect
      tree which contain a mix of array and non-array values.
  """

  @functools.wraps(fn)
  def decorated(array, *, skip_non_arrays: bool = False):
    shaped_array = _maybe_standardize_array(array)
    if shaped_array is None:  # Array was NoneTensorSpec()
      return None
    elif shaped_array is _UNKNOWN_TYPE:  # Array was not recognised
      if skip_non_arrays:
        return array
      else:
        raise TypeError(f'Unknown array-like type: {array!r}')
    return fn(shaped_array)

  return decorated


@_tree_map
@_standardize_array
def shape_dtype_like(array: jax.ShapedArray) -> jax_utils.ShapeDtypeStruct:
  """Converts the nested tree input into ShapeDtype.

  This can be used to visualize a items of a jax tree in a compact way.

  ```
  model = ResNet()
  params = model.init(rng, ...)

  print(jax3d.projects.nesf.shape_dtype_like(params))
  ```

  Args:
    array: Nested tree of arrays,...

  Returns:
    Nested tree of `ShapeDtypeStruct` matching the input tree structure.

  """
  return jax_utils.ShapeDtypeStruct(shape=array.shape, dtype=array.dtype)


@_tree_map
@_standardize_array
def zeros_like(array: jax.ShapedArray) -> jnp.ndarray:
  """Converts the nested tree input to `jnp.zeros`.

  Can be used to initialize `jax`/`flax` models:

  ```
  ds = tfds.load('mnist', split='train')

  model = ResNet()
  model.init(rng, jax3d.utils.zeros_like(ds)['image'])
  ```

  Args:
    array: Nested tree of arrays,...

  Returns:
    Nested tree of `jnp.zeros` matching the input tree structure.
  """
  return jnp.zeros(shape=array.shape, dtype=array.dtype)


@functools.lru_cache()
def _get_none_spec() -> tf.TypeSpec:
  """Returns the tf.NoneTensorSpec()."""
  # We need this hack as NoneTensorSpec is not exposed in the public API.
  ds = tf.data.Dataset.range(0)
  ds = ds.map(lambda x: (x, None))
  return ds.element_spec[-1]


@_tree_map
@_standardize_array
def _tensor_spec_like(array: jax.ShapedArray) -> tf.TensorSpec:
  """Converts the nested tree input to `tf.TensorSpec`.

  This function does not convert None values to valid tensorspecs.

  Args:
    array: Nested tree of arrays,...

  Returns:
    Nested tree of `jnp.zeros` matching the input tree structure.
  """
  dtype = tf.string if array.dtype == np.dtype('O') else array.dtype
  return tf.TensorSpec(shape=array.shape, dtype=dtype)


def tensor_spec_like(array: jax.ShapedArray) -> tf.TensorSpec:
  """Converts the nested tree input to `tf.TensorSpec`.

  Can be used to initialize `tf.data.Dataset` generators:

  ```
  ds = tf.data.Dataset.from_generator(
      ex_generator_fn,
      output_signature=jax3d.projects.nesf.utils.tensor_spec_like(next(ex_generator_fn())),
  )
  ```

  Args:
    array: Nested tree of arrays,...

  Returns:
    Nested tree of `jnp.zeros` matching the input tree structure.
  """
  # Normalize `str` -> `tf.string`
  array = tf.nest.map_structure(_normalize_tensor_spec_str, array)
  array = _tensor_spec_like(array)
  # jax.tree_map skips None elements, so we need a separate tf.nest to replace
  # None values by NoneSpec().
  return tf.nest.map_structure(lambda x: _get_none_spec() if x is None else x,
                               array)


def _normalize_tensor_spec_str(x: Any) -> Any:
  """Normalize elements to tf.TensorSpec."""
  if isinstance(x, (str, bytes)):
    return tf.TensorSpec(shape=(), dtype=tf.string)
  else:
    return x


@_tree_map
@_standardize_array
def types_like(array: jax.ShapedArray) -> tf.TensorSpec:
  """Converts the nested tree input to `ArrayAlias` types.

  Can be used to check expected spec:

  ```
  assert j3d.types_like(out) == f32[1, 28, 28, 3]
  ```

  Args:
    array: Nested tree of arrays,...

  Returns:
    Nested tree of `jnp.zeros` matching the input tree structure.
  """
  return jtyping.ArrayAliasMeta(shape=array.shape, dtype=array.dtype)


def _maybe_standardize_array(
    array: _ArrayInput,
) -> Union[None, jax.ShapedArray, _UNKNOWN_TYPE]:
  """Normalize `tf.Tensor`, `jnp.ndarray`,... as `jax.ShapedArray`."""
  if isinstance(array, (jax.ShapeDtypeStruct, jnp.ndarray, np.ndarray,
                        np.generic)):
    shape = array.shape
    dtype = array.dtype
  elif isinstance(array, jtyping.ArrayAliasMeta):
    shape = (int(x) for x in array.shape.split())
    dtype = array.dtype
  elif isinstance(array, type(_get_none_spec())):
    return None
  elif isinstance(array, (tf.TensorSpec, tf.Tensor)):
    shape = array.shape
    dtype = array.dtype.as_numpy_dtype
  elif isinstance(array, bytes):
    shape = ()
    dtype = np.dtype('O')
  else:
    return _UNKNOWN_TYPE

  if np_utils.is_dtype_str(dtype):  # Normalize `str` dtype
    dtype = np.dtype('O')
  return jax.ShapedArray(shape=shape, dtype=dtype)
