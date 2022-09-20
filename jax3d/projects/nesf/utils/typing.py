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

"""Typing utils."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from etils import epath
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


# *********** Common typing ***********

_T = TypeVar('_T')

# Recursive type for jax.tree
Tree = Union[_T, Any]

# Could replace by `typing.Protocol`
Dataclass = Any

# *********** File-related typing ***********

# Accept both `str` and `pathlib.Path`-like
PathLike = epath.PathLike  # Union[os.PathLike, str]
PathLikeCls = epath.PathLikeCls  # Used in isintance(p, PathLikeCls)

# *********** Tensor-related typing ***********

Tensor = Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]
# Match both `np.dtype('int32')` and np.int32
DType = Union[np.dtype, Type[np.generic]]

# Shape definition spec (e.g. `h w c`, `batch ...`)
ShapeSpec = str

DTYPE_NP_TO_COMPACT_STR: Dict[Optional[np.dtype], str] = {
    np.dtype('uint8'): 'ui8',
    np.dtype('uint32'): 'ui32',
    np.dtype('int32'): 'i32',
    np.dtype('float32'): 'f32',
    np.dtype('float64'): 'f64',
    np.dtype('bool_'): 'bool_',
    np.dtype('O'): 'O',
    None: 'Array',
}


_EllipsisType = type(Ellipsis)  # TODO(py3.10): Use types.EllipsisType
_ShapeItem = Union[ShapeSpec, int, _EllipsisType]
_ShapeSpecInput = Union[_ShapeItem, Tuple[_ShapeItem, ...]]


class ArrayAliasMeta(type):
  """Metaclass to create array aliases.

  This allow to annotate the array shape/dtype with named axis.
  The dtype is defined by the class name (`f32` for `float32`, `ui8` for
  `uint8`, `Array` for any type).
  The shape is defined either as tuple of `str`, `int` or `...`. (e.g
  `f32['b h w c']`, `f32[32, 256, 256, 3]`, `f32[..., 'h w', 3]`).

  All tuple values are concatenated, so `f32[..., 'h', 'w', 'c']` is the
  same as `f32['... h w c']`.

  """
  shape: ShapeSpec
  dtype: DType

  def __new__(
      cls,
      shape: Optional[_ShapeSpecInput],
      dtype: Optional[Type[DType]],
  ):
    dtype = np.dtype(dtype) if dtype else None
    # Normalize to str
    if shape is None:
      shape = '...'
    elif isinstance(shape, tuple):
      shape = ' '.join(_normalize_shape_item(x) for x in shape)
    else:
      shape = _normalize_shape_item(shape)
    return super().__new__(cls, DTYPE_NP_TO_COMPACT_STR[dtype], (cls,), {
        'shape': shape,
        'dtype': dtype,
    })

  def __init__(cls, shape: Optional[ShapeSpec], dtype: Optional[Type[DType]]):
    del shape, dtype
    super().__init__(cls, cls.__name__, (cls,), {})  # pytype: disable=wrong-arg-count

  def __getitem__(cls, shape: _ShapeSpecInput) -> 'ArrayAliasMeta':
    return ArrayAliasMeta(shape=shape, dtype=cls.dtype)

  def __eq__(cls, other: 'ArrayAliasMeta') -> bool:
    return (
        isinstance(other, ArrayAliasMeta)
        and cls.shape == other.shape
        and cls.dtype == other.dtype
    )

  def __hash__(cls) -> int:
    return hash((cls.shape, cls.dtype))

  def __repr__(cls) -> str:
    return f'{cls.__name__}[{cls.shape}]'

  def __instancecheck__(cls, instance: jnp.array) -> bool:
    """`isinstance(array, f32['h w c'])`."""
    from jax3d.projects.nesf.utils import shape_validation  # pylint: disable=g-import-not-at-top
    try:
      shape_validation.assert_match_array_alias(instance, cls)
    except (TypeError, ValueError):
      return False
    else:
      return True

  @classmethod
  def check(cls, array: jnp.array) -> None:
    """Check that the given array match the specs."""
    from jax3d.projects.nesf.utils import shape_validation  # pylint: disable=g-import-not-at-top
    shape_validation.assert_match_array_alias(array, cls)


def _normalize_shape_item(item: _ShapeItem) -> ShapeSpec:
  if isinstance(item, str):
    return item
  elif isinstance(item, int):
    return str(item)
  elif isinstance(item, _EllipsisType):
    return '...'
  else:
    raise TypeError(f'Invalid shape type {type(item)} of: {item}')


Array = ArrayAliasMeta(shape=None, dtype=None)
f32 = ArrayAliasMeta(shape=None, dtype=jnp.float32)
ui8 = ArrayAliasMeta(shape=None, dtype=jnp.uint8)
ui32 = ArrayAliasMeta(shape=None, dtype=jnp.uint32)
i32 = ArrayAliasMeta(shape=None, dtype=jnp.int32)
bool_ = ArrayAliasMeta(shape=None, dtype=jnp.bool_)
StrArray = ArrayAliasMeta(shape=None, dtype=np.dtype('O'))  # pytype: disable=wrong-arg-types  # typed-numpy

# Random number generator jax key
PRNGKey = ui32[2]

# Any activation function for f32.
ActivationFn = Callable[[f32['...']], f32['...']]

_ArrayT = TypeVar('_ArrayT', bound=ArrayAliasMeta)

# ArrayLike indicates that any `jnp.array` input is also supported.
# For example: `ArrayLike[i32[2]]` accept `(28, 28)`, `[x, y]`, `np.ones((2,))`
ArrayLike = Union[_ArrayT, Tuple[Any, ...], List[Any]]
