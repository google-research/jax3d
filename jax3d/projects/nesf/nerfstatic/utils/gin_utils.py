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

"""Utilities for interacting with Gin."""

import collections
import dataclasses
import difflib
import functools
import itertools
import textwrap
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple, Type, TypeVar, Union

import gin

import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.utils.typing import Dataclass, PathLike  # pylint: disable=g-multiple-import


# ClsT/Cls is the dataclass on which `@gin_utils.dataclass_configurable` is
# applied
_ClsT = TypeVar('_ClsT')
_Instance = Dataclass
_Cls = Type[_Instance]

ConfigurableDataclass = _Instance

# Set of valid scope names. All other scope names are rejected.
_VALID_SCOPES: Set[str] = set()


def log_params_to_disk(path: PathLike, params: _Instance) -> None:
  """Write on disk the full nested config values (inculding default values).

  Args:
    path: filepath to write params to.
    params: params to log
  """
  p = j3d.Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(params.to_gin_operative_repr() + '\n')


def dataclass_configurable(cls: _ClsT) -> _ClsT:
  r"""Wrapper around `gin.configurable`.

  This is a drop-in replacement of `gin.configurable` applied for dataclasses.

  Dataclass decorated through this supports inheritance. So gin will
  correctly inject bindings from the parent classes.

  ```python
  @gin_utils.dataclass_configurable
  @dataclasses.dataclass
  class DatasetParams:
    # Params common to all classes
    batch_size: int

  @gin_utils.dataclass_configurable
  @dataclasses.dataclass
  class KlevrParams(DatasetParams):
    # Params specific to klevr dataset
    num_scenes: int

  gin.parse_config(\"""
  DatasetParams.batch_size = 64
  KlevrParams.num_scenes = 3
  \""")

  assert KlevrParams() == KlevrParams(batch_size=64, num_scenes=3)
  ```

  Note: Both parent and child class should be decorated through this. In the
  gin config, each field can only be set once `Param.field = value`.

  Args:
    cls: The dataclass to register with gin

  Returns:
    cls: The registered class
  """
  # TODO(epot): Could automatically apply `ConfigField` descriptors when
  # we detect cls.__annotation__ is a dataclass_configurable.
  # * Make sure type annotations are still correctly forwarded (as
  #   `@dataclass_configurable` would have to internally apply `@dataclass`).
  # * Make sure it works when type annotations is `Optional[ConfigParams]` (or
  #   a forward-ref `str`)
  # * As classes are immutable, we could also simply only inject the magic
  #   through __init__ instead of using descriptor (but we should be careful
  #   about default factory)

  if not dataclasses.is_dataclass(cls):
    raise TypeError(f'{cls} is not a dataclass')

  # Inspect the class to register the valid scopes
  _register_valid_scopes(cls)

  # Track child subclass for inheritance support
  # Called upon definition of a child class. See PEP 487.
  def __init_subclass__(sub_cls, **kwargs):  # pylint: disable=invalid-name
    # Call `parent(cls).__init_subclass__(sub_cls, **kwargs)` to register
    # `sub_cls` in the parents.
    super(cls, sub_cls).__init_subclass__(**kwargs)
    cls._CONFIG_SUBCLASS[sub_cls.__name__] = sub_cls  # pylint: disable=protected-access

  # Keep track of all childs of `cls` (including itself) by mapping
  # child_cls.__name__ -> child_cls.
  # This allow `ConfigField(DatasetParams)` to initialize any child (e.g
  # `KlevrParams`)
  # Params(dataset='KlevrParams')  # params.dataset is a KlevrParams() instance
  cls._CONFIG_SUBCLASS = {cls.__name__: cls}  # pylint: disable=protected-access
  cls.__init_subclass__ = classmethod(__init_subclass__)

  # Apply gin.configurable first before wrapping the __init__ (as gin also
  # wrap the `__init__`)
  cls = gin.configurable(cls)

  # Wrap the `__init__` to inject gin bindings from the parents.
  # Note: Explicitly given kwargs always takes precedence
  cls.__init__ = _add_parent_kwargs(cls.__init__)

  cls.__repr__ = dataclass_repr

  cls.to_gin_operative_repr = to_gin_operative_repr
  return cls


def _iter_parent_gin_bindings(
    instance: _Instance,
) -> Iterator[Tuple[_Cls, Dict[str, Any]]]:
  """Yields the cls, gin_bindings kwargs of all parent classes."""
  for cls in type(instance).mro():
    if cls is object:
      continue
    # Sanity check to make sure dataclasses are properly decorated
    # with gin_utils, and not gin.
    _assert_gin_dataclass_configurable(cls)
    yield cls, _get_bindings(cls)


def _get_bindings(cls: _Cls) -> Dict[str, Any]:
  """Returns the gin arguments injected to `cls.__init__`."""
  # The default nested gin scopes resolution order is confusing:
  # `Child_x/Param.x = 1` is not ignored if the current scope is
  # `Parent_x/Child_x/`, but applied if scope is `Child_x/Parent_x/`.
  # Instead, we overwrite the gin resolution order to apply scopes
  # from less to most specific.
  # Example: For scope `scope0/scope1/`, we apply in order of preference:
  # * scope0/scope1/MyParam.x = 1
  # * scope1/MyParam.x = 1
  # * scope0/MyParam.x = 1
  # * MyParam.x = 1

  bindings = {}
  all_scopes = gin.current_scope()
  with gin.config_scope(None):  # Clear all scopes
    for i in range(len(all_scopes) + 1):  # From less to more specific
      for curr_scopes in itertools.combinations(all_scopes, i):
        with gin.config_scope('/'.join(curr_scopes)):
          new_kwargs = gin.get_bindings(
              cls, resolve_references=False, inherit_scopes=False)
          bindings.update(new_kwargs)

  return bindings


def dataclass_repr(self: _Instance) -> str:
  """Dataclass.__repr__ method (pretty print one field-per lines)."""
  lines = [
      f'{field.name}={getattr(self, field.name)!r},\n'
      for field in dataclasses.fields(self)
      if field.repr
  ]
  if lines:
    fields_str = '\n' + textwrap.indent(''.join(lines), '    ')
  else:
    fields_str = ''
  return f'{type(self).__qualname__}({fields_str})'


def _field_repr_with_default(
    instance: _Instance,
    field: dataclasses.Field,
    is_gin_binding: bool = False,
) -> str:
  """Representation of a single dataclass field (e.g. `x=1,  # Default: 3`)."""
  value = getattr(instance, field.name)
  if _is_gin_dataclass_configurable(value):
    # Add the gin scope here (for the inner `_iter_parent_gin_bindings`
    # so the default values are correctly highlighted). For example, if
    # param.mlp_rgb.num_layers uses default but not param.mlp_sigma.num_layers
    with gin.config_scope(_scope_name(type(instance), field.name)):
      value_str = value.to_gin_operative_repr()
  else:
    value_str = repr(value)

  if is_gin_binding:  # Value was overwritten by gin
    if field.default is not dataclasses.MISSING:
      default_str = f'Default: {field.default!r}'
    elif field.default_factory is not dataclasses.MISSING:
      default_str = 'Default factory overwritten'
    else:
      default_str = 'Required'
    default_str = f'  # {default_str}'
  else:
    default_str = ''

  return f'{field.name}={value_str},{default_str}\n'


def to_gin_operative_repr(self) -> str:
  """Similar to `__repr__` but hightlight the field updated by gin."""
  # Merge all overwritten kwargs together
  all_binding_kwargs = {}
  for _, binding_kwargs in _iter_parent_gin_bindings(self):
    all_binding_kwargs.update(binding_kwargs)

  lines = [
      _field_repr_with_default(
          self, field, is_gin_binding=field.name in all_binding_kwargs
      ) for field in dataclasses.fields(self)
  ]

  if lines:
    fields_str = '\n' + textwrap.indent(''.join(lines), '    ')
  else:
    fields_str = ''
  return f'{type(self).__qualname__}({fields_str})'


class ConfigField(j3d.utils.DataclassField[_Cls, _Cls]):
  r"""Field representing a sub-config object.

  Example:

  ```python
  @gin_utils.dataclass_configurable
  @dataclasses.dataclass
  class Model:
    num_layers: int = 3

  @gin_utils.dataclass_configurable
  @dataclasses.dataclass
  class NerfModel(Model):
    output_dim: int = 10

  @gin_utils.dataclass_configurable
  @dataclasses.dataclass
  class PSFModel(Model):
    activation: str = 'relu'


  @gin_utils.dataclass_configurable
  @dataclasses.dataclass
  class Params:
    model: Model = ConfigField(NerfModel)

  gin.parse_config(\"""
    Params.model = 'PSFModel'
    Model.num_layers = 3
    PSFModel.activation = 'relu'
  \""")

  params = Params()
  params.model == PSFModel(num_layers=3, activation='relu')
  ```

  """

  def __init__(
      self,
      cls: _Cls,
      *,
      required: bool = False,
  ):
    """Constructor.

    Args:
      cls: The `@gin_utils.dataclass_configurable` default class to construct
        In gin, this field can be overwritten by any related class (cls sharing
        the same parent). In the above example, `NerfModel` is the default but
        `PSFModel` and `Model` are also accepted.
      required: If `True`, the field (`Params.model` in the above example) is
        required.
    """
    _assert_gin_dataclass_configurable(cls)
    self._default_cls = cls
    self._cls = _get_topmost_parent_class(cls)
    self._required = required
    super().__init__()

  def _default(self) -> str:  # pytype: disable=signature-mismatch
    # Returns as string so the construction happens in `_validate` (with
    # correct scope and checks).
    if self._required:
      raise TypeError(f'Missing 1 required argument: {self._attribute_name}')
    else:
      return self._default_cls.__name__

  def _validate(self, value: _Cls) -> _Cls:
    # Validate that the class on which the descriptor is attached has been
    # decorated with `@gin_utils.dataclass_configurable`.
    # We cannot check in `__set_name__` as the decorator isn't yet applied
    # when the class is created.
    _assert_gin_dataclass_configurable(self._objtype)

    if isinstance(value, (type(None), self._cls)):
      return value
    elif isinstance(value, str):
      if value.startswith('@'):
        err_msg = '\nGin macro should not be wrapped in quotes.'
      else:
        err_msg = ''
      if value not in self._cls._CONFIG_SUBCLASS:  # pylint: disable=protected-access
        raise ValueError(
            f'Invalid param str {value!r}. '  # pylint: disable=protected-access
            f'Should be one of {self._cls._CONFIG_SUBCLASS.keys()}'
            + err_msg
        )
      cls = self._cls._CONFIG_SUBCLASS[value]  # pylint: disable=protected-access
      _assert_gin_dataclass_configurable(cls)

      # Construct the field. Similar to:
      #
      # with gin.config_scope('ConfigParams_model'):
      #   config_params.model = cls()
      #
      # Note: When inheritance the scope name has to be the dataclass on which
      # the field is defined `ParentCls_field` (an error is raised if trying
      # to use `ChildCls_field` as scope).
      scope_name = _scope_name(self._objtype, self._attribute_name)
      with gin.config_scope(scope_name):
        with j3d.utils.try_reraise(f'{scope_name}/ '):
          return cls()
    else:
      raise TypeError(f'Invalid config type: {value}')


def _add_parent_kwargs(init_fn: Callable[..., None]) -> Callable[..., None]:
  """Wraps `__init__` of a dataclass such that parent gin bindings are injected.

  Args:
    init_fn: Current __init__ function.

  Returns:
    init_fn: New __init__ function
  """

  @functools.wraps(init_fn)
  def __init__(self, **given_kwargs):  # pylint: disable=invalid-name
    """New init_fn which inject the gin bindings from all parent classes."""

    # kwargs forwarded to __init__ (merging of all `gin.get_bindings(cls)`)
    init_kwargs: Dict[str, Any] = {}
    # Mapping `cls: gin.get_bindings(cls)` for better debug messages
    cls_to_kwargs: Dict[str, Dict[str, Any]] = {}
    # Mapping kwarg_name to the list of class using this kwarg
    # Check that there is no overlapp between the kwargs to avoid conflicts
    # (e.g. if both `NerfParam.num_layers` and `Model.num_layers` are defined)
    common_kwargs: Dict[str, List[_Cls]] = collections.defaultdict(list)
    for cls, gin_kwargs in _iter_parent_gin_bindings(self):

      # Extract all bindings from the current class (e.g. All `NerfModel.xyz`)
      for k, v in gin_kwargs.items():
        # We cannot use gin macro as this would introduce all types of subtle
        # bugs about scope not being applied properly.
        if isinstance(v, gin.config.ConfigurableReference):
          raise ValueError(
              f'Gin macro forbidden: {k}={v}. Please use '
              '`gin_utils.ConfigField` with str instead ('
              "@MlpParams() -> 'MlpParams').")
        common_kwargs[k].append(cls)

      cls_to_kwargs[cls] = gin_kwargs
      init_kwargs.update(gin_kwargs)

    # Make sure there is no `ParentCls.x`, `ChildCls.x` conflicts
    _assert_bindings_defined_once(self, common_kwargs)

    # Merge gin with explicitly given kwargs
    init_kwargs.update(given_kwargs)

    try:
      return init_fn(self, **init_kwargs)
    except TypeError as e:  # Bad arguments
      lines = [
          f'    {cls.__qualname__}: {list(kwargs)}'  # pytype: disable=attribute-error  # py39-upgrade
          for cls, kwargs in cls_to_kwargs.items()
      ]
      lines = '\n'.join(lines)
      suffix = (
          f'\n  Gin had values bound for:\n{lines}\n'
      )
      j3d.utils.reraise(e, suffix=suffix)

  return __init__


def _assert_bindings_defined_once(
    instance: _Instance,
    common_kwargs: Dict[str, List[_Cls]],
) -> None:
  """Validate that bindings are only extracted once."""
  common_kwargs = {
      k: common_classes for k, common_classes in common_kwargs.items()  # pytype: disable=annotation-type-mismatch
      if len(common_classes) != 1
  }
  if not common_kwargs:
    return

  all_fields = []
  for field_name, all_cls in common_kwargs.items():
    all_fields.extend(f'{cls.__name__}.{field_name}' for cls in all_cls)
  all_fields = ', '.join(all_fields)
  # TODO(epot): Could check `__annotations__` and `__dataclass_fields__` to
  # make fields are not refering to parent class
  raise ValueError(
      f'Duplicated gin bindings for {type(instance).__name__}. Bindings should '
      f'be defined in a single place. Got: {all_fields}'
  )


def _get_topmost_parent_class(cls: Type[Any]) -> Type[Any]:
  """Returns the top-most parent class (excluding object).

  Args:
    cls: The cls from which extract the parent.

  Returns:
    The top-most parent class
  """
  *all_parent_cls, object_cls = cls.mro()
  assert object_cls is object
  for parent_cls in all_parent_cls:
    _assert_gin_dataclass_configurable(parent_cls)
  return all_parent_cls[-1]


def _is_gin_dataclass_configurable(
    instance_or_cls: Union[_Instance, _Cls]) -> bool:
  """Asserts the dataclass was decorated with `gin_utils`."""
  return hasattr(instance_or_cls, '_CONFIG_SUBCLASS')


def _assert_gin_dataclass_configurable(cls: _Cls) -> None:
  """Asserts the dataclass was decorated with `gin_utils`."""
  # Use __dict__ instead of hasattr to ignore parents.
  if cls.__dict__.get('_CONFIG_SUBCLASS', None) is None:
    raise TypeError(
        f'{cls} should be a `gin_utils.dataclass_configurable` dataclass.'
    )


# Could add a `gin.assert_scopes_in(scopes)` directly in gin, but not clear
# how to validate relative order of scopes.
def validate_scope_names() -> None:
  """Asserts all scope names are valid."""
  for (scope, selector), bindings_kwargs in gin.config._CONFIG.items():  # pylint: disable=protected-access
    if not scope:
      continue
    for scope_part in scope.split('/'):
      # Could have a smarter scope validation which also check
      # whether the relative order of scope parts is valid.
      if scope_part not in _VALID_SCOPES:
        error_string = f"Invalid scope {scope} ('{scope_part}' not recognised)."
        close_matches = difflib.get_close_matches(scope_part, _VALID_SCOPES)
        if close_matches:
          error_string += f'\nDid you meant: {scope_part} -> {close_matches}\n'
        raise ValueError(error_string)


def _register_valid_scopes(cls: _Cls) -> None:
  """Register scopes which can be."""
  for field in dataclasses.fields(cls):
    # Use `__dict__` to support inheritance (field only registered once so
    # `ParentCls_field` is valid scope, but not `ChildCls_field` )
    if isinstance(cls.__dict__.get(field.name), ConfigField):
      _VALID_SCOPES.add(_scope_name(cls, field.name))


def _scope_name(cls: _Cls, field_name: str) -> str:
  return f'{cls.__name__}_{field_name}'
