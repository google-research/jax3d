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

"""Signature parsing utils."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, Generic, Iterator, Mapping, Optional, TypeVar, Union

from etils import edc
from etils import epy

_ArgT = TypeVar('_ArgT')
_NewArgT = TypeVar('_NewArgT')
_OutT = TypeVar('_OutT')
_Fn = Callable[..., _OutT]


@edc.dataclass
@dataclasses.dataclass
class Signature(Generic[_OutT]):
  """Wrapper around `inspect.Signature` with additional features.

  It allow to validate/transform function args before calling the function.

  Usage:

  ```python
  sig = Signature(fn)

  bound_args = sig.bind(*args, **kwargs)

  for arg in bound_args:  # Iterate over args/kwargs
    print(f'{arg.fn_name} called with {arg.name}={arg.value!r}')

  bound_args = bound_args.map(_validate_or_transform_args)

  y = bound_args.call()  # Call the function
  ```

  """
  fn: _Fn[_OutT]
  signature: inspect.Signature = dataclasses.field(init=False)

  def __post_init__(self):
    self.signature = inspect.Signature.from_callable(self.fn)

  @property
  def parameters(self) -> Mapping[str, inspect.Parameter]:
    return self.signature.parameters

  @property
  def has_var(self) -> bool:
    """Returns `True` if the signature has `*args` or `**kwargs`."""
    var_kind = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return any(p.kind in var_kind for p in self.parameters.values())

  @property
  def fn_name(self) -> str:
    """Returns function qualname."""
    return self.fn.__qualname__

  # TODO(pytype): pytype wrongly infer `_ArgT`
  # def bind(self, *args: _ArgT, **kwargs: _ArgT) -> BoundArgs[_ArgT, _OutT]:
  def bind(self, *args: Any, **kwargs: Any) -> BoundArgs[Any, _OutT]:
    bound_args = self.signature.bind(*args, **kwargs)
    return BoundArgs(bound_args=bound_args, signature=self)


@edc.dataclass
@dataclasses.dataclass
class BoundArgs(Generic[_ArgT, _OutT]):
  """Bounded arguments (wrapper around `inspect.BoundArguments`)."""
  bound_args: inspect.BoundArguments
  signature: Signature

  @property
  def has_self(self) -> bool:
    """Return `True` if the argument has `self`."""
    # Ideally, this should be moved in `Signature`
    return bool(len(self) and self[0].is_self)

  @property
  def self_bound_arg(self) -> BoundArg[_ArgT]:
    """Return the `self` bounded arg of the bound call."""
    if not self.has_self:
      raise ValueError(
          f'{self.fn_name} does not have `self` arg: '
          f'{self.signature.signature}')
    return self[0]

  @property
  def self_value(self) -> _ArgT:
    """Return the `self` arg of the bound call."""
    return self.self_bound_arg.value

  @property
  def fn(self) -> _Fn[_OutT]:
    return self.signature.fn

  @property
  def fn_name(self) -> str:
    return self.signature.fn_name

  def call(self, fn: Optional[_Fn[_OutT]] = None) -> _OutT:
    """Call the function.

    Args:
      fn: Function to call. If not given, the original `fn` (bound to the
        signature) is called.

    Returns:
      Output of `fn(*args, **kwargs)`
    """
    if fn is None:
      fn = self.fn
    return fn(*self.bound_args.args, **self.bound_args.kwargs)

  @epy.cached_property  # pytype: disable=invalid-annotation
  def _arguments_list(self) -> list[BoundArg[_ArgT]]:
    arg_items = self.bound_args.arguments.items()
    return [
        BoundArg(  # pylint: disable=g-complex-comprehension
            name=name,
            value=value,
            pos=i,
            bound_args=self,
        ) for i, (name, value) in enumerate(arg_items)
    ]

  @epy.cached_property  # pytype: disable=invalid-annotation
  def _arguments_dict(self) -> dict[str, BoundArg[_ArgT]]:
    return {arg.name: arg for arg in self._arguments_list}  # pylint: disable=not-an-iterable

  def __getitem__(self, key: Union[int, str]) -> BoundArg[_ArgT]:
    """Can access a key through name or position."""
    if isinstance(key, (int, slice)):
      return self._arguments_list[key]
    elif isinstance(key, str):
      return self._arguments_dict[key]
    else:
      raise TypeError(
          f'{self.__class__.__qualname__} indices should be str, int or '
          f'slice. Not {type(key)} ({key!r})')

  def __len__(self) -> int:
    return len(self._arguments_list)

  def __iter__(self) -> Iterator[BoundArg[_ArgT]]:
    return iter(self._arguments_list)

  def map(
      self,
      fn: Callable[[_ArgT], _NewArgT],
  ) -> BoundArgs[_NewArgT, _OutT]:
    """Apply validation/modification to the arguments value.

    Example:

    ```python
    def fn(x, y):
      return x + y

    sig = Signature(fn)
    bound_args = sig.bind(1, y=10)  # bound_args(x=1, y=10)

    bound_args = bound_args.map(lambda x: x * 2)  # bound_args(x=2, y=20)
    assert bound_args.call() == 22  # fn(2, 20) == 22
    ```

    Args:
      fn: Function applied on each args/kwargs values. The returned value will
        be the new arg.

    Returns:
      The new `BoundArgs` with updated args.
    """

    def _fn(arg: BoundArg[_ArgT]) -> _NewArgT:
      return fn(arg.value)

    return self.map_bound_arg(_fn)

  def map_bound_arg(
      self,
      fn: Callable[[BoundArg[_ArgT]], _NewArgT],
  ) -> BoundArgs[_NewArgT, _OutT]:
    """Apply validation/modification to the arguments value."""

    def _fn(arg: BoundArg[_ArgT]) -> _NewArgT:  # pytype: disable=invalid-annotation
      try:
        return fn(arg)
      except Exception as e:  # pylint: disable=broad-except
        epy.reraise(
            e,
            prefix=
            f'Error in {self.fn_name} for arg {arg.pos} ({arg.name!r}): ',
        )

    bound_args = inspect.BoundArguments(
        signature=self.bound_args.signature,
        arguments={arg.name: _fn(arg) for arg in self},  # pytype: disable=wrong-arg-types
    )
    return self.replace(bound_args=bound_args)  # pytype: disable=attribute-error


@edc.dataclass
@dataclasses.dataclass
class BoundArg(Generic[_ArgT]):
  """Bounded argument."""
  name: str
  value: _ArgT
  pos: int
  bound_args: BoundArgs

  @property
  def signature(self) -> Signature:
    return self.bound_args.signature

  @property
  def parameter(self) -> inspect.Parameter:
    return self.signature.parameters[self.name]

  @property
  def fn_name(self) -> str:
    return self.signature.fn_name

  @property
  def is_self(self) -> bool:
    """Return `True` if the argument is `self`."""
    # pyformat: disable
    return (
        self.pos == 0
        and self.name == 'self'
        and self.parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
    # pyformat: enable
