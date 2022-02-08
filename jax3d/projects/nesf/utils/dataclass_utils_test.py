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

"""Tests for jax3d.projects.nesf.utils.dataclass."""

import dataclasses
import enum
from typing import Any, Optional

import jax3d.projects.nesf as j3d
import pytest


class MyEnum(enum.Enum):
  DEFAULT = enum.auto()
  SOME_VALUE = enum.auto()


class MyOtherEnum(enum.Enum):
  DEFAULT = enum.auto()
  SOME_VALUE = enum.auto()


def test_enum_field():

  @dataclasses.dataclass
  class A:
    # pytype: disable=annotation-type-mismatch
    enum_required: MyEnum = j3d.utils.EnumField(enum_cls=MyEnum)
    enum_optional: Optional[MyEnum] = j3d.utils.EnumField(None, enum_cls=MyEnum)
    enum_optional2: MyEnum = j3d.utils.EnumField('default', enum_cls=MyEnum)
    enum_infered: MyEnum = j3d.utils.EnumField(MyEnum.DEFAULT)
    # pytype: enable=annotation-type-mismatch

  a = A(enum_required='some_value')  # pytype: disable=wrong-arg-types
  assert a.enum_required is MyEnum.SOME_VALUE
  assert a.enum_optional is None
  assert a.enum_optional2 is MyEnum.DEFAULT
  assert a.enum_infered is MyEnum.DEFAULT

  # Works on lowercase, uppercase str, enum.Enum
  a.enum_optional = 'some_value'  # pytype: disable=annotation-type-mismatch
  assert a.enum_optional is MyEnum.SOME_VALUE

  a.enum_optional = 'DEFAULT'  # pytype: disable=annotation-type-mismatch
  assert a.enum_optional is MyEnum.DEFAULT

  a.enum_optional = MyEnum.SOME_VALUE
  assert a.enum_optional is MyEnum.SOME_VALUE

  with pytest.raises(ValueError, match='Enum should be one of'):
    a.enum_optional = 'invalid_value'  # pytype: disable=annotation-type-mismatch

  with pytest.raises(TypeError, match='Invalid input'):
    a.enum_optional = MyOtherEnum.DEFAULT  # pytype: disable=annotation-type-mismatch


class MyField(j3d.utils.DataclassField):
  """Field with default value."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.default_was_called = False

  def _default(self):
    self.default_was_called = True
    return 123

  def _validate(self, value):
    return value * 10


class MyFieldNoDefault(j3d.utils.DataclassField):
  """Field with no default factory."""

  def _validate(self, value):
    return value * 10


@pytest.mark.parametrize('frozen', [True, False])
def test_default_factory_field_no_default(frozen: bool):

  @dataclasses.dataclass(frozen=frozen)
  class A:
    x: Any = MyField()

  my_field = A.__dict__['x']

  # Default factory is call at instantiation, not class creation time
  assert not my_field.default_was_called
  a = A()
  assert my_field.default_was_called
  assert a.x == 1230  # Validate is called on the default value

  # The function is called each time a new instance is created with default
  # params
  my_field.default_was_called = False  # Reset
  a = A()
  assert my_field.default_was_called
  assert a.x == 1230

  my_field.default_was_called = False  # Reset
  a = A(x=456)
  assert a.x == 4560

  # Updating the value should only work for non-frozen instances.
  if frozen:
    with pytest.raises(dataclasses.FrozenInstanceError):
      a.x = 678
  else:
    a.x = 678
    assert a.x == 6780


@pytest.mark.parametrize('frozen', [True, False])
@pytest.mark.parametrize('field_cls', [MyField, MyFieldNoDefault])
def test_default_field_with_default(frozen: bool, field_cls):

  @dataclasses.dataclass(frozen=frozen)
  class A:
    x: Any = field_cls(789)

  a = A()
  # TODO(epot): Fix bug
  # Currently, `_validate` is called twice on the default value:
  # 1) In `MyField.__init__` call
  # 2) In `A.__init__` during `self.x = x`
  # Somehow this is consistent with how traitlets behave so it shouldn't be
  # a major issue in practice.
  assert a.x == 78900

  a = A(x=456)
  assert a.x == 4560

  if isinstance(field_cls, MyField):
    # _default() should never have been called (as explicit default value
    # provided)
    my_field = A.__dict__['x']
    assert not my_field.default_was_called

  # Updating the value should only work for non-frozen instances.
  if frozen:
    with pytest.raises(dataclasses.FrozenInstanceError):
      a.x = 678
  else:
    a.x = 678
    assert a.x == 6780


@pytest.mark.parametrize('frozen', [True, False])
def test_field_no_default(frozen: bool):

  @dataclasses.dataclass(frozen=frozen)
  class A:
    x: Any = MyFieldNoDefault()

  # If no default factory, an error should be raised.
  with pytest.raises(TypeError, match=r'__init__\(\) missing 1 required'):
    A()

  class ChildA(A):

    def __init__(self):
      # No default yet (init not called), so raise attribute error
      with pytest.raises(AttributeError, match=" has no attribute 'x'"):
        _ = self.x
      super().__init__(123)

  @dataclasses.dataclass(frozen=frozen)
  class ADefault:
    x: Any = MyFieldNoDefault(789)

  a0 = A(456)
  a1 = ChildA()
  a2 = ADefault()
  assert a0.x == 4560
  assert a1.x == 1230
  assert a2.x == 78900

  with pytest.raises(AttributeError, match=" has no attribute 'x'"):
    _ = A.x


@pytest.mark.parametrize('frozen', [True, False])
def test_multi_instances_default(frozen: bool):
  """Test with multiple instances of A."""

  @dataclasses.dataclass(frozen=frozen)
  class A:
    x: Any = MyField(default=123)

  a0 = A()
  a1 = A(x=456)
  a2 = A()
  a3 = A(x=789)
  assert a0.x == 12300
  assert a1.x == 4560
  assert a2.x == 12300
  assert a3.x == 7890
  assert A.x == 1230

  # Updating the value should only work for non-frozen instances.
  if frozen:
    with pytest.raises(dataclasses.FrozenInstanceError):
      a0.x = 678
  else:
    a0.x = 678
    a3.x = 2
    assert a0.x == 6780
    assert a3.x == 20


@pytest.mark.parametrize('frozen', [True, False])
def test_default_field_bad_access(frozen: bool):

  class A:
    x: Any = MyField()

  # A.x is accessed before `dataclasses.dataclass` was applied
  _ = A.x

  A = dataclasses.dataclass(frozen=frozen)(A)  # pylint: disable=invalid-name

  # Default factory not called, so explicit error raised
  with pytest.raises(TypeError, match=r'__init__\(\) missing 1 required'):
    A()


@pytest.mark.parametrize('repr_', [True, False])
@pytest.mark.parametrize('eq', [True, False])
@pytest.mark.parametrize('order', [True, False])
@pytest.mark.parametrize('unsafe_hash', [True, False])
@pytest.mark.parametrize('frozen', [True, False])
def test_default_factory(
    repr_: bool,
    eq: bool,
    order: bool,
    unsafe_hash: bool,
    frozen: bool,
):
  """Make sure that default factory works for all combinations of dataclass."""
  if order and not eq:  # eq must be true if order is true
    return

  make_dataclass = dataclasses.dataclass(
      # default_factory doesn't make sense if init = False
      init=True,
      repr=repr_,
      eq=eq,
      order=order,
      unsafe_hash=unsafe_hash,
      frozen=frozen,  # pytype: disable=not-supported-yet
  )

  @make_dataclass
  class A:
    x: Any = MyField()

  my_field = A.__dict__['x']
  assert not my_field.default_was_called
  assert A().x == 1230
  assert my_field.default_was_called

  # Updating the value should only work for non-frozen instances.
  a = A()
  if frozen:
    with pytest.raises(dataclasses.FrozenInstanceError):
      a.x = 678
  else:
    a.x = 678
    assert a.x == 6780
