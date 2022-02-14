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

"""Tests for jax3d.projects.nesf.nerfstatic.utils.gin_utils."""

import dataclasses
import pathlib
import textwrap
from typing import Any

import gin
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
import pytest


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class A:
  a: int


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class B(A):
  b: int


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class C(B):
  c: int
  c2: int = 2


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class Params:
  model: Any = gin_utils.ConfigField(A, required=True)


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class ChildParams(Params):
  x: Any = gin_utils.ConfigField(A)
  y: Any = gin_utils.ConfigField(A)
  z: int = 1


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class NestedParamsOptionArgs:
  x: Any = 1


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class NestedParamsOption:
  args0: Any = gin_utils.ConfigField(NestedParamsOptionArgs)
  args1: Any = gin_utils.ConfigField(NestedParamsOptionArgs)


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class NestedParams:
  option0: Any = gin_utils.ConfigField(NestedParamsOption)
  option1: Any = gin_utils.ConfigField(NestedParamsOption)


@pytest.fixture(scope='function', autouse=True)
def _reset_gin():
  gin_utils._VALID_SCOPES.update([
      'root',
      'child',
      'parent',
  ])
  gin.clear_config()
  yield
  # Ensure scopes set inside the test were valid
  gin_utils.validate_scope_names()


def test_log_params_to_disk(tmp_path: pathlib.Path):
  gin.parse_config("""
  Params.model = 'B'
  A.a = 1
  B.b = 'hello'
  """)
  params = Params()  # Ensure C.a is referenced.   # pytype: disable=missing-parameter
  path = tmp_path / 'args.params.py'
  gin_utils.log_params_to_disk(path, params)
  assert path.read_text() == textwrap.dedent(
      """\
      Params(
          model=B(
              a=1,  # Required
              b='hello',  # Required
          ),  # Default factory overwritten
      )
      """
  )


def test_config_field():

  # Create expected default values before the gin parsing
  default_a = A(a=1)
  default_b = B(a=1, b=2)
  default_c = C(a=1, b=2, c=3)

  gin.parse_config("""
  A.a = 1
  B.b = 2
  C.c = 3
  """)

  # By default, ConfigField is required
  with pytest.raises(TypeError, match='Missing 1 required argument'):
    _ = Params()

  params = Params(model='A')
  assert type(params.model) is A  # pylint: disable=unidiomatic-typecheck
  assert params.model == default_a
  assert repr(params) == str(params)
  assert repr(params) == textwrap.dedent(
      """\
      Params(
          model=A(
              a=1,
          ),
      )"""
  )
  assert params.to_gin_operative_repr() == textwrap.dedent(  # pytype: disable=attribute-error
      """\
      Params(
          model=A(
              a=1,  # Required
          ),
      )"""
  )

  params = Params(model='B')
  assert type(params.model) is B  # pylint: disable=unidiomatic-typecheck
  assert params.model == default_b

  params = Params(model='C')
  assert type(params.model) is C  # pylint: disable=unidiomatic-typecheck
  assert params.model == default_c
  assert repr(params) == textwrap.dedent(
      """\
      Params(
          model=C(
              a=1,
              b=2,
              c=3,
              c2=2,
          ),
      )"""
  )

  # Values can be passed directly
  params = Params(model=C(b=234))   # pytype: disable=missing-parameter
  assert params.model == C(a=1, b=234, c=3)

  # Invalid value
  with pytest.raises(ValueError, match='Invalid param str'):
    Params(model='D')


def test_config_field_gin_error():

  gin.parse_config("""
  A.a = 1
  B.b = 2
  C.b = 3  # b is defined in C
  C.c = 3
  """)

  with pytest.raises(ValueError, match='Duplicated gin bindings'):
    Params(model='C')


def test_gin_config_field_wrong_input():

  # gin.configurable instead of gin_utils
  @gin.configurable
  @dataclasses.dataclass
  class NotAConfigurable:
    x: int

  with pytest.raises(
      TypeError, match='should be a `gin_utils.dataclass_configurable`'):
    gin_utils.ConfigField(NotAConfigurable)

  with pytest.raises(TypeError, match='is not a dataclass'):

    @gin_utils.dataclass_configurable  # pylint: disable=unused-variable
    class NotADataclass:
      a: int

  @gin.configurable  # pylint: disable=unused-variable
  @dataclasses.dataclass
  class NotAConfigurableChild(A):
    x: int

  with pytest.raises(
      TypeError, match='should be a `gin_utils.dataclass_configurable`'):
    Params(model='NotAConfigurableChild')


def test_to_gin_operative_repr():

  gin.parse_config("""
  B.b = 2
  C.c = 3
  C.c2 = 4
  """)

  assert C(a=1).to_gin_operative_repr() == textwrap.dedent(  # pytype: disable=missing-parameter,attribute-error
      """\
      C(
          a=1,
          b=2,  # Required
          c=3,  # Required
          c2=4,  # Default: 2
      )"""
  )


def test_scope():
  gin.parse_config("""
  A.a = 1
  B.b = 2
  C.c = 3

  Params.model = 'C'
  Params_model/A.a = 10
  Params_model/B.b = 20

  ChildParams.x = 'C'

  ChildParams.y = 'B'
  ChildParams_y/A.a = 100
  ChildParams_y/B.b = 200
  """)
  params = Params()
  assert params == Params(model=C(a=10, b=20, c=3))
  assert params.model.a == 10
  assert params.model.b == 20
  assert params.model.c == 3
  assert params.to_gin_operative_repr() == textwrap.dedent(  # pytype: disable=attribute-error
      """\
      Params(
          model=C(
              a=10,  # Required
              b=20,  # Required
              c=3,  # Required
              c2=2,
          ),  # Default factory overwritten
      )"""
  )

  params = ChildParams()
  assert params == ChildParams(
      model=C(a=10, b=20, c=3),
      x=C(a=1, b=2, c=3),
      y=B(a=100, b=200),
  )
  assert params.model.a == 10
  assert params.model.b == 20
  assert params.model.c == 3
  assert params.model.c2 == 2
  assert params.x.a == 1
  assert params.x.b == 2
  assert params.x.c == 3
  assert params.x.c2 == 2
  assert params.y.a == 100
  assert params.y.b == 200
  assert params.to_gin_operative_repr() == textwrap.dedent(  # pytype: disable=attribute-error
      """\
      ChildParams(
          model=C(
              a=10,  # Required
              b=20,  # Required
              c=3,  # Required
              c2=2,
          ),  # Default factory overwritten
          x=C(
              a=1,  # Required
              b=2,  # Required
              c=3,  # Required
              c2=2,
          ),  # Default factory overwritten
          y=B(
              a=100,  # Required
              b=200,  # Required
          ),  # Default factory overwritten
          z=1,
      )"""
  )


def test_scope_nested():

  gin.parse_config("""
  NestedParams.option0 = 'NestedParamsOption'
  NestedParamsOption.args1 = 'NestedParamsOptionArgs'

  NestedParams_option0/NestedParamsOption_args0/NestedParamsOptionArgs.x = '00'
  NestedParams_option1/NestedParamsOptionArgs.x = '1'
  NestedParams_option1/NestedParamsOption_args1/NestedParamsOptionArgs.x = '11'
  """)
  params = NestedParams()
  params_explicit = NestedParams(
      option0=NestedParamsOption(
          args0=NestedParamsOptionArgs(x='00'),
          args1=NestedParamsOptionArgs(x=1),
      ),
      option1=NestedParamsOption(
          args0=NestedParamsOptionArgs(x='1'),
          args1=NestedParamsOptionArgs(x='11'),
      ),
  )
  assert params == params_explicit

  assert params.option0.args0.x == '00'
  assert params.option0.args1.x == 1
  assert params.option1.args0.x == '1'
  assert params.option1.args1.x == '11'
  assert params_explicit.option0.args0.x == '00'
  assert params_explicit.option0.args1.x == 1
  assert params_explicit.option1.args0.x == '1'
  assert params_explicit.option1.args1.x == '11'

  assert repr(params) == textwrap.dedent(
      """\
      NestedParams(
          option0=NestedParamsOption(
              args0=NestedParamsOptionArgs(
                  x='00',
              ),
              args1=NestedParamsOptionArgs(
                  x=1,
              ),
          ),
          option1=NestedParamsOption(
              args0=NestedParamsOptionArgs(
                  x='1',
              ),
              args1=NestedParamsOptionArgs(
                  x='11',
              ),
          ),
      )"""
  )
  assert params.to_gin_operative_repr() == textwrap.dedent(  # pytype: disable=attribute-error
      """\
      NestedParams(
          option0=NestedParamsOption(
              args0=NestedParamsOptionArgs(
                  x='00',  # Default: 1
              ),
              args1=NestedParamsOptionArgs(
                  x=1,
              ),  # Default factory overwritten
          ),  # Default factory overwritten
          option1=NestedParamsOption(
              args0=NestedParamsOptionArgs(
                  x='1',  # Default: 1
              ),
              args1=NestedParamsOptionArgs(
                  x='11',  # Default: 1
              ),  # Default factory overwritten
          ),
      )"""
  )


def test_scope_nested_inner_only():
  # Only set the default inner values

  gin.parse_config("""
  NestedParamsOption_args0/NestedParamsOptionArgs.x = 'b_default'
  NestedParams_option0/NestedParamsOption_args0/NestedParamsOptionArgs.x = 'a'
  NestedParamsOption_args1/NestedParamsOptionArgs.x = 'b'
  """)
  params = NestedParams()
  params_explicit = NestedParams(
      option0=NestedParamsOption(
          args0=NestedParamsOptionArgs(x='a'),
          args1=NestedParamsOptionArgs(x='b'),
      ),
      option1=NestedParamsOption(
          args0=NestedParamsOptionArgs(x='b_default'),
          args1=NestedParamsOptionArgs(x='b'),
      ),
  )
  assert repr(params) == repr(params_explicit)
  assert params == params_explicit


def test_valid_scope_names():
  """Check that an error is raised if a wrong scope is used."""
  gin_utils.validate_scope_names()

  gin.clear_config()
  # typo: use `Params_model` -> `ChildParams_model`
  gin.parse_config("""
  ChildParams_model/C.c = 3
  """)
  with pytest.raises(ValueError, match='Invalid scope'):
    gin_utils.validate_scope_names()

  gin.clear_config()
  gin.parse_config("""
  Params_model/C.c = 3
  """)
  gin_utils.validate_scope_names()


def test_gin_macro_invalid():
  gin.parse_config("""
  Params.model = @C()
  """)
  with pytest.raises(ValueError, match='Gin macro forbidden'):
    Params()

  gin.clear_config()
  gin.parse_config("""
  Params.model = @C
  """)
  with pytest.raises(ValueError, match='Gin macro forbidden'):
    Params()

  # Also check with inheritance
  gin.clear_config()
  gin.parse_config("""
  Params.model = @C
  ChildParams.x = 'A'
  ChildParams.y = 'A'
  """)
  with pytest.raises(ValueError, match='Gin macro forbidden'):
    ChildParams()


def test_default_factory_create_cls():

  @gin_utils.dataclass_configurable
  @dataclasses.dataclass
  class FactoryParamsCreateCls:
    params: Any = gin_utils.ConfigField(A)

  gin.parse_config("""
  A.a = 123
  """)
  assert FactoryParamsCreateCls().params == A(a=123)


def test_default_factory_required():

  @gin_utils.dataclass_configurable
  @dataclasses.dataclass
  class FactoryParamsRequired:
    params: Any = gin_utils.ConfigField(A, required=True)

  gin.parse_config("""
  A.a = 123
  """)

  with pytest.raises(TypeError, match='Missing 1 required argument'):
    _ = FactoryParamsRequired()

  assert FactoryParamsRequired(params='A').params == A(a=123)

  gin.parse_config("""
  FactoryParamsRequired.params = 'A'
  """)

  assert FactoryParamsRequired().params == A(a=123)


def test_default_factory_explicit():

  @gin_utils.dataclass_configurable
  @dataclasses.dataclass
  class FactoryParamsExplicit:
    params: Any = gin_utils.ConfigField(B)

  gin.parse_config("""
  A.a = 123
  B.b = 234
  """)

  assert FactoryParamsExplicit().params == B(a=123, b=234)

  # Default factory is compatible with scope
  gin.parse_config("""
  FactoryParamsExplicit_params/A.a = 10
  FactoryParamsExplicit_params/B.b = 100
  """)
  assert FactoryParamsExplicit().params == B(a=10, b=100)

  # Default factory can be overwritten
  gin.parse_config("""
  FactoryParamsExplicit.params = 'A'
  """)
  assert FactoryParamsExplicit().params == A(a=10)


def test_get_topmost_parent_class():
  assert gin_utils._get_topmost_parent_class(A) is A
  assert gin_utils._get_topmost_parent_class(B) is A
  assert gin_utils._get_topmost_parent_class(C) is A
  assert gin_utils._get_topmost_parent_class(Params) is Params
  assert gin_utils._get_topmost_parent_class(ChildParams) is Params


@gin.configurable
def fn_in_scope(x=1):
  return x


@pytest.mark.parametrize(
    'config', [
        # Nested scopes are correctly applied
        """
        fn_in_scope.x = 'expected'
        """,
        """
        child/fn_in_scope.x = 'expected'
        """,
        """
        parent/child/fn_in_scope.x = 'expected'
        """,
        """
        root/parent/child/fn_in_scope.x = 'expected'
        """,
        """
        root/fn_in_scope.x = 'expected'
        """,
        """
        root/parent/fn_in_scope.x = 'expected'
        """,
        # Missing inner scope should still work
        """
        root/child/fn_in_scope.x = 'expected'
        """,
        # More specific scope take precedence over more generic ones
        """
        fn_in_scope.x = 'bad'
        child/fn_in_scope.x = 'bad'
        parent/child/fn_in_scope.x = 'bad'
        root/parent/child/fn_in_scope.x = 'expected'
        """,
        """
        parent/child/fn_in_scope.x = 'expected'
        child/fn_in_scope.x = 'bad'
        fn_in_scope.x = 'bad'
        """,
        """
        fn_in_scope.x = 'bad'
        child/fn_in_scope.x = 'expected'
        """,
        """
        parent/fn_in_scope.x = 'bad'
        child/fn_in_scope.x = 'expected'
        """,
        """
        root/fn_in_scope.x = 'bad'
        child/fn_in_scope.x = 'expected'
        """,
        """
        parent/fn_in_scope.x = 'bad'
        root/child/fn_in_scope.x = 'expected'
        """,
        """
        root/child/fn_in_scope.x = 'bad'
        parent/child/fn_in_scope.x = 'expected'
        """,
    ],
)
def test_gin_scope_order(config):
  # Test that the scope order does not change

  gin.clear_config()
  gin.parse_config(config)

  with gin.config_scope('root'):
    with gin.config_scope('parent'):
      with gin.config_scope('child'):
        assert gin_utils._get_bindings('fn_in_scope') == {'x': 'expected'}  # pytype: disable=wrong-arg-types
