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

"""Dataclass array."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Callable, Iterable, Iterator, Tuple, TypeVar, Union

from etils import edc
from etils import enp
from etils.array_types import Array
from jax3d.visu3d import np_utils
from jax3d.visu3d import py_utils
from jax3d.visu3d.typing import DType, Shape  # pylint: disable=g-multiple-import
import numpy as np

if typing.TYPE_CHECKING:
  from jax3d.visu3d import transformation

lazy = enp.lazy

# Any valid numpy indices slice ([x], [x:y], [:,...], ...)
_IndiceItem = Union[type(Ellipsis), None, int, slice, Any]
_Indices = Tuple[_IndiceItem]  # Normalized slicing
_IndicesArg = Union[_IndiceItem, _Indices]

_Dc = TypeVar('_Dc')

_METADATA_KEY = 'v3d_field'


class DataclassArray:
  """Dataclass which behaves like an array.

  Usage:

  ```python
  @dataclasses.dataclass
  class Square(DataclassArray):
    pos: Array['*shape 2'] = array_field(shape=(2,))
    scale: Array['*shape'] = array_field(shape=())

  # Create 2 square batched
  p = Square(pos=[[x0, y0], [x1, y1], [x2, y2]], scale=[scale0, scale1, scale2])
  p.shape == (3,)
  p.pos.shape == (3, 2)
  p[0] == Square(pos=[x0, y0], scale=scale0)

  p = p.reshape((3, 1))  # Reshape the inner-shape
  p.shape == (3, 1)
  p.pos.shape == (3, 1, 2)
  ```

  """
  _shape: Shape
  _xnp: enp.NpModule
  _array_fields: list[_ArrayField]

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    # TODO(epot): Could have smart __repr__ which display types if array have
    # too many values.
    edc.dataclass_utils.add_repr(cls)
    cls._v3d_tree_map_registered = False

  def __post_init__(self) -> None:
    """Validate and normalize inputs."""
    # Register the tree_map here instead of `__init_subclass__` as `jax` may
    # not have been registered yet during import
    cls = type(self)
    if enp.lazy.has_jax and not cls._v3d_tree_map_registered:  # pylint: disable=protected-access
      enp.lazy.jax.tree_util.register_pytree_node_class(cls)
      cls._v3d_tree_map_registered = True  # pylint: disable=protected-access

    # Validate and normalize array fields (e.g. list -> np.array,...)
    array_fields = [
        _ArrayField(  # pylint: disable=g-complex-comprehension
            name=f.name,
            host=self,
            **f.metadata[_METADATA_KEY].to_dict(),
        ) for f in dataclasses.fields(self) if _METADATA_KEY in f.metadata
    ]
    if not array_fields:
      raise ValueError(
          f'{self.__class__.__qualname__} should have at least one '
          '`v3d.array_field`')

    # Filter `None` values
    array_fields = [f for f in array_fields if f.value is not None]

    # Validate the array type is consistent (all np or all jnp but not both)
    xnps = py_utils.groupby(
        array_fields,
        key=lambda f: f.xnp,
        value=lambda f: f.name,
    )
    if len(xnps) > 1:
      xnps = {k.__name__: v for k, v in xnps.items()}
      raise ValueError(f'Conflicting numpy types: {xnps}')

    # Validate the batch shape is consistent
    shapes = py_utils.groupby(
        array_fields,
        key=lambda f: f.host_shape,
        value=lambda f: f.name,
    )
    if len(shapes) > 1:
      raise ValueError(f'Conflicting batch shapes: {shapes}')

    if not xnps:  # No values
      # Inside `jax.tree_utils`, tree-def can be created with `None` values.
      assert not shapes
      xnps = (np,)
      shapes = (None,)

    # TODO(epot): Support broadcasting

    # Cache results
    (xnp,) = xnps
    (shape,) = shapes
    # Should the state be stored in a separate object to avoid collisions ?
    self._setattr('_shape', shape)
    self._setattr('_xnp', xnp)
    self._setattr('_array_fields', array_fields)

  # ====== Array functions ======

  @property
  def shape(self) -> Shape:
    """Returns the batch shape common to all fields."""
    return self._shape

  def reshape(self: _Dc, shape: Union[tuple[int, ...], str]) -> _Dc:
    """Reshape the batch shape according to the pattern."""
    if isinstance(shape, str):
      # TODO(epot): Have an einops.rearange version which only look at the
      # first `self.shape` dims.
      # einops.rearrange(x,)
      raise NotImplementedError

    def _reshape(f):
      return f.value.reshape((*shape, *f.inner_shape))

    return self._map_field(_reshape)

  def flatten(self: _Dc) -> _Dc:
    """Flatten the batch shape."""
    return self.reshape((-1,))

  def broadcast_to(self: _Dc, shape: Shape) -> _Dc:
    """Broadcast the batch shape."""
    return self._map_field(
        lambda f: self.xnp.broadcast_to(f.value, shape + f.inner_shape))

  def __getitem__(self: _Dc, indices: _IndicesArg) -> _Dc:
    """Slice indexing."""
    indices = np.index_exp[indices]  # Normalize indices
    # Replace `...` by explicit shape
    indices = _to_absolute_indices(indices, shape=self.shape)
    return self._map_field(lambda f: f.value[indices])

  # _Dc[n *d] -> Iterator[_Dc[*d]]
  def __iter__(self: _Dc) -> Iterator[_Dc]:
    """Iterate over the outermost dimension."""
    if not self.shape:
      raise ValueError(f'Cannot iterate on {self!r}: No batch shape.')

    # Similar to `etree.unzip(self)` (but work with any backend)
    field_names = [f.name for f in self._array_fields]
    field_values = [f.value for f in self._array_fields]
    for vals in zip(*field_values):
      yield self.replace(**dict(zip(field_names, vals)))

  def map_field(
      self: _Dc,
      fn: Callable[[Array['*din']], Array['*dout']],
  ) -> _Dc:
    """Apply a transformation on all arrays from the fields."""
    return self._map_field(lambda f: fn(f.value))

  # ====== Dataclass/Conversion utils ======

  replace = edc.dataclass_utils.replace

  def as_np(self: _Dc) -> _Dc:
    """Returns the instance as containing `np.ndarray`."""
    return self.as_xnp(enp.lazy.np)

  def as_jax(self: _Dc) -> _Dc:
    """Returns the instance as containing `jnp.ndarray`."""
    return self.as_xnp(enp.lazy.jnp)

  def as_tf(self: _Dc) -> _Dc:
    """Returns the instance as containing `tf.Tensor`."""
    return self.as_xnp(enp.lazy.tnp)

  def as_xnp(self: _Dc, xnp: enp.NpModule) -> _Dc:
    """Returns the instance as containing `xnp.ndarray`."""
    return self.map_field(xnp.asarray)

  # ====== Internal ======

  @property
  def xnp(self) -> enp.NpModule:
    """Returns the numpy module of the class (np, jnp, tnp)."""
    return self._xnp

  def apply_transform(
      self: _Dc,
      tr: transformation.Transform,
  ) -> _Dc:
    """Transform protocol.

    Applied the transformation on it-self. Called during:

    ```python
    my_obj = tr @ my_obj  # Call `my_obj.apply_transform(tr)`
    ```

    Args:
      tr: Transformation to apply
    """
    raise NotImplementedError(
        f'{self.__class__.__qualname__} does not support `v3d.Transform`.')

  def _map_field(
      self: _Dc,
      fn: Callable[[_ArrayField], Array['*dout']],
  ) -> _Dc:
    """Apply a transformation on all array fields structure."""
    # TODO(epot): Should we have a non-batched version where the transformation
    # is applied on each leaf (with some vectorization) ?
    # Like: .map_leaf(Callable[[_Dc], _Dc])
    # Would be trickier to support np/TF.
    new_values = {f.name: fn(f) for f in self._array_fields}
    return self.replace(**new_values)

  def tree_flatten(self):
    """`jax.tree_utils` support."""
    children_values = [f.value for f in self._array_fields]
    children_names = [f.name for f in self._array_fields]
    return (children_values, children_names)

  @classmethod
  def tree_unflatten(cls, children_names, children_values):
    """`jax.tree_utils` support."""
    children_kwargs = dict(zip(children_names, children_values))
    return cls(**children_kwargs)

  def _setattr(self, name: str, value: Any) -> None:
    """Like setattr, but support `frozen` dataclasses."""
    object.__setattr__(self, name, value)

  def assert_same_xnp(self, x: Union[Array[...], DataclassArray]) -> None:
    """Assert the given array is of the same type as the current object."""
    xnp = np_utils.get_xnp(x)
    if xnp is not self.xnp:
      raise ValueError(
          f'{self.__class__.__name__} is {self.xnp.__name__} but got input '
          f'{xnp.__name__}. Please cast input first.')


def stack(
    arrays: Iterable[_Dc],  # list[_Dc['*shape']]
    *,
    axis: int = 0,
) -> _Dc:  # _Dc['len(arrays) *shape']:
  """Stack dataclasses together."""
  arrays = list(arrays)
  first_arr = arrays[0]
  cls = type(first_arr)

  # This might have some edge cases if user try to stack subclasses
  types = py_utils.groupby(
      arrays,
      key=type,
      value=lambda x: type(x).__name__,
  )
  if False in types:
    raise TypeError(
        f'v3.stack got conflicting types as input: {list(types.values())}')

  xnp = first_arr.xnp
  if axis != 0:
    # If axis < 0, we should normalize the axis such as the last axis is
    # before the inner shape
    # axis = self._to_absolute_axis(axis)
    raise NotImplementedError('Please open an issue.')
  new_vals = {  # pylint: disable=g-complex-comprehension
      f.name: xnp.stack(
          [getattr(arr, f.name) for arr in arrays],
          axis=axis,
      ) for f in first_arr._array_fields  # pylint: disable=protected-access
  }
  return cls(**new_vals)


def _count_not_none(indices: _Indices) -> int:
  """Count the number of non-None and non-ellipsis elements."""
  return len([k for k in indices if k is not np.newaxis and k is not Ellipsis])


def _to_absolute_indices(indices: _Indices, *, shape: Shape) -> _Indices:
  """Normalize the indices to replace `...`, by `:, :, :`."""
  assert isinstance(indices, tuple)
  ellipsis_count = indices.count(Ellipsis)
  if ellipsis_count > 1:
    raise IndexError("an index can only have a single ellipsis ('...')")
  valid_count = _count_not_none(indices)
  if valid_count > len(shape):
    raise IndexError(f'too many indices for array. Batch shape is {shape}, but '
                     f'rank-{valid_count} was provided.')
  if not ellipsis_count:
    return indices
  ellipsis_index = indices.index(Ellipsis)
  start_elems = indices[:ellipsis_index]
  end_elems = indices[ellipsis_index + 1:]
  ellipsis_replacement = [slice(None)] * (len(shape) - valid_count)
  return (*start_elems, *ellipsis_replacement, *end_elems)


def array_field(
    shape: Shape,
    dtype: DType = float,
    **field_kwargs,
) -> dataclasses.Field:
  """Dataclass array field.

  See `v3d.DataclassArray` for example.

  Args:
    shape: Inner shape of the field
    dtype: Type of the field
    **field_kwargs: Args forwarded to `dataclasses.field`

  Returns:
    The dataclass field.
  """
  # TODO(epot): Validate shape, dtype
  v3d_field = _ArrayFieldMetadata(
      inner_shape=shape,
      dtype=dtype,
  )
  return dataclasses.field(**field_kwargs, metadata={_METADATA_KEY: v3d_field})


@edc.dataclass
@dataclasses.dataclass
class _ArrayFieldMetadata:
  """Metadata of the array field (shared across all instances).

  Attributes:
    inner_shape: Inner shape
    dtype: Type of the array
  """
  inner_shape: Shape
  dtype: DType

  def __post_init__(self):
    """Normalizing/validating the shape/dtype."""
    self.inner_shape = tuple(self.inner_shape)
    if None in self.inner_shape:
      raise ValueError(f'Shape should be defined. Got: {self.inner_shape}')
    if self.dtype is int:
      self.dtype = np.int32
    if self.dtype is float:
      self.dtype = np.float32

  def to_dict(self) -> dict[str, Any]:
    """Returns the dict[field_name, field_value]."""
    return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@edc.dataclass
@dataclasses.dataclass
class _ArrayField(_ArrayFieldMetadata):
  """Array field of a specific dataclass instance.

  Attributes:
    name: Instance of the attribute
    host: Dataclass instance who this field is attached too
    xnp: Numpy module
  """
  name: str
  host: DataclassArray
  xnp: enp.NpModule = dataclasses.field(init=False)

  def __post_init__(self):
    if self.value is None:  # No validation when there is no value
      return
    # Convert and normalize the array
    self.xnp = lazy.get_xnp(self.value, strict=False)
    value = self.xnp.asarray(self.value, dtype=self.dtype)
    self.host._setattr(self.name, value)  # pylint: disable=protected-access
    if self.host_shape + self.inner_shape != value.shape:
      raise ValueError(
          f'Expected {value.shape} last dimensions to be {self.inner_shape} '
          f'for {self.name!r}')

  @property
  def value(self) -> Array['...']:
    """Access the `host.<field-name>`."""
    return getattr(self.host, self.name)

  @property
  def host_shape(self) -> Shape:
    """Host shape (batch shape shared by all fields)."""
    if not self.inner_shape:
      return self.value.shape
    else:
      return self.value.shape[:-len(self.inner_shape)]
