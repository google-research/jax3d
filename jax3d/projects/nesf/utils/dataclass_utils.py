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

"""Dataclass utils."""

import dataclasses
import enum
from typing import Any, Generic, Optional, Type, TypeVar, Union

from jax3d.projects.nesf.utils.typing import Dataclass

_InT = TypeVar('_InT')
_OutT = TypeVar('_OutT')
_EnumT = TypeVar('_EnumT', bound=enum.Enum)


class DataclassField(Generic[_InT, _OutT]):
  """Abstract descriptor which perform data validation/conversion.

  This is somewhat similar to `traitlets`, but directly compatible with
  dataclasses.

  Example:

  ```
  class PositiveInteger(DataclassField[int, int]):

    def _default(self):
      return 123

    def _validate(self, value: int) -> int:
      if i < 0:
        raise ValueError('i should be positive')
      return i


  @dataclasses.dataclass
  class A:
    x: int = PositiveInteger()

  A(x=1)  # works
  A(x=-1)  # ValueError: i should be positive
  assert A().x == 123  # Default factory
  ```

  """

  def __init__(self, default: _InT = dataclasses.MISSING) -> None:
    """Constructor.

    Args:
      default: Default field value. If not passed, the value will be required.
    """
    # Attribute name and objtype refer to the object in which the descriptor
    # is applied. E.g. in the docstring example:
    # * _attribute_name = 'x'
    # * _objtype = A
    self._attribute_name: Optional[str] = None
    self._objtype: Optional[Type[Dataclass]] = None

    # Whether the descriptor is a required or not. If set to True,
    # calling `A()` will raise "Missing required kwargs 'x'" unless
    # `_default()` is overwritten.
    self._is_missing = default is dataclasses.MISSING

    # Default value (when `_is_missing is False`)
    # Should validate really called at import time ?
    self._default_value = None if self._is_missing else self._validate(default)

    # Whether `__get__` was called once. See `__get__` for details.
    self._first_getattr_call: bool = True

  def __set_name__(self, objtype: Type[Dataclass], name: str) -> None:
    self._objtype = objtype
    self._attribute_name = name

  def __get__(
      self,
      obj: Optional[Dataclass],
      objtype: Optional[Type[Dataclass]] = None,
  ) -> _OutT:
    # Called as `MyDataclass.my_path`
    if obj is None:
      if not self._is_missing:
        return self._default_value
      # If `_default` is overwritten, we need to send `default_factory`
      # to dataclass.
      if self.is_default_overwritten() and self._first_getattr_call:
        # Count the number of times `dataclasses.dataclass(cls)` calls
        # `getattr(cls, f.name)`.
        # The first time, we return a `dataclasses.Field` to let dataclass
        # do the magic.
        # The second time, `dataclasses.dataclass` delete the descriptor if
        # `isinstance(getattr(cls, f.name, None), Field)`. So it is very
        # important to return anything except a `dataclasses.Field`.
        # This rely on implementation detail, but seems to hold for python
        # 3.6-3.10.
        self._first_getattr_call = False
        return dataclasses.field(default_factory=self._default)
      else:
        # If no default value is provided, `MyDataclass.my_path` should raise
        # AttributeError.
        raise AttributeError(
            f"type object '{self._objtype.__qualname__}' has no attribute "
            f"'{self._attribute_name}'"
        )
    # Called as `my_dataclass.my_path`
    return _getattr(
        obj,
        self._attribute_name,
        dataclasses.MISSING if self._is_missing else self._default_value
    )

  def __set__(self, obj: Dataclass, value: _InT) -> None:
    _setattr(obj, self._attribute_name, self._validate(value))

  def is_default_overwritten(self) -> bool:
    return type(self)._default is not DataclassField._default

  def _default(self) -> _OutT:
    """Abstract factory which returns the default value if none is provided.

    Returns:
      value: The default value.
    """
    raise NotImplementedError('Abstract method.')

  def _validate(self, value: _InT) -> _OutT:
    """Abstract method which validate or convert attribute value.

    Note: Calling `_validate` twice on a value should be a no-op (so
    `_validate(_validate(x)) == _validate(x)`)

    Args:
      value: Input value to validate/convert, as passed in `__init__` or `=`

    Returns:
      value: The value, eventually converted/updated.
    """
    return value


def _getattr(
    obj: Dataclass,
    attribute_name: str,
    default: Union[_OutT, type(dataclasses.MISSING)],
) -> _OutT:
  """Returns the `obj.attribute_name`."""
  _init_dataclass_state(obj)
  # Accessing the attribute before it was set (e.g. before super().__init__)
  if (
      attribute_name not in obj._dataclass_field_values  # pylint: disable=protected-access
      and default is dataclasses.MISSING
  ):
    raise AttributeError(
        f"type object '{type(obj).__qualname__}' has no attribute "
        f"'{attribute_name}'"
    )
  else:
    return obj._dataclass_field_values.get(attribute_name, default)  # pylint: disable=protected-access


def _setattr(
    obj: Dataclass,
    attribute_name: str,
    value: Any,
) -> None:
  """Set the `obj.attribute_name = value`."""
  # Note: In `dataclasses.dataclass(frozen=True)`, obj.__setattr__ will
  # correctly raise a `FrozenInstanceError` before `DataclassField.__set__` is
  # called.
  _init_dataclass_state(obj)
  obj._dataclass_field_values[attribute_name] = value  # pylint: disable=protected-access


def _init_dataclass_state(obj: Dataclass) -> None:
  """Initialize the object state containing all DataclassField values."""
  if not hasattr(obj, '_dataclass_field_values'):
    # Use object.__setattr__ for frozen dataclasses
    object.__setattr__(obj, '_dataclass_field_values', {})


class EnumField(DataclassField[Union[str, _EnumT], _EnumT]):
  """Enum field which auto-convert `str` in value.

  Example:

  ```python
  @dataclasses.dataclass
  class A:
    my_enum: j3d.utils.EnumField(MyEnum.DEFAULT)
    required_enum = j3d.utils.EnumField(enum_cls=MyEnum)
    optional_enum = j3d.utils.EnumField(None, enum_cls=MyEnum)

  a = A(
      required_enum='some_value',
  )
  assert a.required_enum is MyEnum.SOME_VALUE
  ```

  """

  def __init__(
      self,
      default: Union[
          str, None, _EnumT, type(dataclasses.MISSING)
      ] = dataclasses.MISSING,  # pylint: disable=bad-whitespace
      *,
      enum_cls: Optional[Type[_EnumT]] = None,
      **kwargs: Any,
  ):
    """Constructor.

    Args:
      default: Default enum value.
      enum_cls: Enum class. Only required if `default` is None or missing.
      **kwargs: Forwarded to `DataclassField`
    """
    # Try to auto-infer enum type from the param.
    if isinstance(default, enum.Enum):
      if enum_cls is None:
        enum_cls = type(default)
      elif not isinstance(default, enum_cls):
        raise ValueError(f'Conflicting enum types: {default} is not {enum_cls}')
    self._enum_cls: Type[_EnumT] = enum_cls  # pytype: disable=annotation-type-mismatch
    self._str2enum = {x.name.lower(): x for x in self._enum_cls}
    super().__init__(default, **kwargs)

  def _validate(self, value: Union[str, None, _EnumT]) -> Optional[_EnumT]:  # pytype: disable=signature-mismatch
    """Validate the value."""
    if isinstance(value, str):
      value = value.lower()  # pytype: disable=attribute-error
      if value not in self._str2enum:
        raise ValueError(
            f'Enum should be one of {list(self._str2enum.keys())}. '
            f'Not {value!r}.'
        )
      return self._str2enum[value]
    elif isinstance(value, self._enum_cls):
      return value  # pytype: disable=bad-return-type
    elif value is None:
      return None
    else:
      raise TypeError(f'Invalid input {value}')
