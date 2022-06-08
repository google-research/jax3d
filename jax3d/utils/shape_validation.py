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

"""Utils to validate the shape."""

import contextlib
import dataclasses
import functools
import inspect
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Tuple, TypeVar

from etils import array_types
import jax.numpy as jnp
from jax3d.utils import jax_utils
from jax3d.utils import py_utils

_AnyFn = Callable[..., Any]
_Fn = TypeVar('_Fn', bound=_AnyFn)
_TypingAnnotation = Any


def assert_typing(fn: _Fn) -> _Fn:
  """Validate the array shape/dtype at runtime.

  Usage:

  ```
  @jax3d.assert_typing
  def reduce(x: f32['h w c']) -> f['h w']:
    return x.mean(-1)

  reduce(jnp.ones((1, 2)))  # Error: Expected rank 3
  ```

  Args:
    fn: Function to check types.

  Returns:
    decorated function
  """

  @functools.wraps(fn)
  def decorated(*args, **kwargs):
    with _ShapeTracker.track():
      try:
        sig = inspect.signature(fn)
        _assert_type_signature(sig, args, kwargs)
        y = fn(*args, **kwargs)
        _assert_type(y, sig.return_annotation, name='returned value')
        return y
      except Exception as e:  # pylint: disable=broad-except
        fn_name = getattr(fn, '__qualname__', fn)
        py_utils.reraise(e, f'Error in {fn_name}:\n')

  return decorated


@dataclasses.dataclass(frozen=True, eq=False)
class _ShapeTracker:
  """Scope which track the shape.

  Attributes:
    _shapes: Mapping from shape to value
  """
  _shapes: Dict[str, int] = dataclasses.field(default_factory=dict)

  # Shape tracker contains the stack of `@assert_typing` scopes.
  # Each `@assert_typing` enter a new scope so function typing do not interfer
  # with each others even during nested calls.
  _shape_tracker_stack: ClassVar[List['_ShapeTracker']] = []

  @classmethod
  @contextlib.contextmanager
  def track(cls) -> Iterator[None]:
    new_tracker_scope = cls()
    try:
      cls._shape_tracker_stack.append(new_tracker_scope)
      yield
    finally:
      cls._shape_tracker_stack.pop()

  @classmethod
  def current(cls) -> '_ShapeTracker':
    """Returns the current `_ShapeTracker`."""
    try:
      curr_tracker = cls._shape_tracker_stack[-1]
    except IndexError:
      raise AssertionError(
          'Calling `check` from outside a variable tracking '
          'scope. Make sure to decorate your function with `@assert_typing`'
      )
    return curr_tracker

  def track_and_validate_shape(
      self,
      shape: Tuple[int, ...],
      expected_shape: array_types.typing.ShapeSpec,
  ) -> None:
    """Register new named axis and validate the shape.

    Args:
      shape: Array shape example: `(28, 28, 3)`
      expected_shape: Expected shape spec annotation (e.g. 'h w c')
    """
    shape_names = expected_shape.split()
    if len(shape_names) != len(shape):
      raise ValueError('Rank should be the same.')
    for name, value in zip(shape_names, shape):
      expected_value = self._shapes.setdefault(name, value)
      if expected_value != value:
        raise ValueError(f'Expected {name}={expected_value}, got {value}')

  def resolve_spec(self, shape_spec: array_types.typing.ShapeSpec) -> str:
    """Returns the shape_spec with each named axis replaced by value."""
    shape_names = shape_spec.split()
    shape_values = [str(self._shapes.get(name, name)) for name in shape_names]
    return ' '.join(shape_values)


def _assert_type_signature(
    sig: inspect.Signature,
    args: Any,
    kwargs: Any,
) -> None:
  """Check the signature type of the function."""
  bargs = sig.bind(*args, **kwargs)
  bargs.apply_defaults()

  for param, value in zip(sig.parameters.values(), bargs.arguments.values()):
    if param.kind == inspect.Parameter.VAR_POSITIONAL:
      for v in value:
        _assert_type(v, param.annotation, name=param.name)
    elif param.kind == inspect.Parameter.VAR_KEYWORD:
      for k, v in value.items():
        _assert_type(v, param.annotation, name=k)
    else:
      _assert_type(value, param.annotation, name=param.name)


def _assert_type(
    value: Any,
    annotation: Optional[_TypingAnnotation],
    name: str,
) -> None:
  """Check that a specific value match its annotated type."""
  # TODO(jax3d): Should recurse into `Dict`, `List`, `dataclass`,...
  if not isinstance(annotation, array_types.ArrayAliasMeta):
    return
  try:
    assert_match_array_alias(value, annotation)
  except Exception as e:  # pylint: disable=broad-except
    py_utils.reraise(e, prefix=f'Bad argument {name}: ')


def assert_match_array_alias(
    array: jnp.ndarray,
    expected_spec: array_types.ArrayAliasMeta,
) -> None:
  """Check that a specific value match its annotated type."""
  if isinstance(array, (int, float)):
    array = jnp.array(array)
  if not isinstance(array, jnp.ndarray):
    raise TypeError(f'Expected {expected_spec} array. Got {type(array)}')

  try:
    # Should have a way of checking a type is a subset of another:
    # if dtype1 < dtype2:
    if (not isinstance(expected_spec.dtype, array_types.dtypes.AnyDType) and
        expected_spec.dtype.np_dtype != array.dtype):
      raise ValueError('Dtype do not match')
    elif expected_spec.shape != '...':
      _ShapeTracker.current().track_and_validate_shape(
          shape=array.shape,
          expected_shape=expected_spec.shape,
      )
  except Exception as e:  # pylint: disable=broad-except
    curr_spec = jax_utils.ShapeDtypeStruct(array.shape, array.dtype)
    expected_repr = _ShapeTracker.current().resolve_spec(expected_spec.shape)
    py_utils.reraise(e, f'{curr_spec} != {expected_spec} ({expected_repr}): ')
