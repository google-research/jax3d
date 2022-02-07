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

"""Python utils."""

import collections.abc
import contextlib
import functools
import typing
from typing import Any, Callable, NoReturn, Optional, TypeVar, Union

__all__ = [
    'cached_property',
    'decorator_with_option',
    'reraise',
    'try_reraise',
]


_T = TypeVar('_T')


# TODO(jax3d): Typing annotations (protocol with @typing.overload on __call__)
def decorator_with_option(
    decorator_fn,
):
  """Wraps a decorator to correctly forward decorator options.

  `decorator_with_option` is applied on decorators. Usage:

  ```
  @jax3d.utils.decorator_with_option
  def my_decorator(fn, x=None, y=None):
    ...
  ```

  The decorated decorator can then be used with or without options, or
  called directly.

  ```
  @my_decorator(x, y=y)
  def fn():
    ...

  @my_decorator
  def fn():
    ...

  fn = my_decorator(fn, x, y=y)
  ```

  Args:
    decorator_fn: The decorator with signature `(fn, *option, **option_kwargs)`

  Returns:
    The `decorator_fn` which now can be used as decorator with option.
  """

  @functools.wraps(decorator_fn)
  def decorated(*args: Any, **kwargs: Any) -> Any:
    fn = args[0] if args else None
    if not isinstance(fn, collections.abc.Callable):

      def decorated_with_options(fn):
        return decorator_fn(fn, *args, **kwargs)
      return decorated_with_options

    return decorator_fn(fn, *args[1:], **kwargs)

  return decorated


def reraise(
    e: Exception,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> NoReturn:
  """Reraise an exception with an additional message."""
  prefix = prefix or ''
  suffix = '\n' + suffix if suffix else ''

  # If unsure about modifying the function inplace, create a new exception
  # and stack it in the chain.
  if (
      # Exceptions with custom error message
      type(e).__str__ is not BaseException.__str__
      # This should never happens unless the user plays with Exception
      # internals
      or not hasattr(e, 'args')
      or not isinstance(e.args, tuple)
  ):
    msg = f'{prefix}{e}{suffix}'
    # Could try to dynamically create a
    # `type(type(e).__name__, (ReraisedError, type(e)), {})`, but should be
    # careful when nesting `reraise` as well as compatibility with external
    # code.
    # Some base exception class (ImportError, OSError) and subclasses (
    # ModuleNotFoundError, FileNotFoundError) have custom `__str__` error
    # message. We re-raise those with same type to allow except in caller code.

    class WrappedException(type(e)):
      """Exception proxy with additional message."""

      def __init__(self, msg):
        # We explicitly bypass super() as the `type(e).__init__` constructor
        # might have special kwargs
        Exception.__init__(self, msg)  # pylint: disable=non-parent-init-called

      def __getattr__(self, name: str):
        # Capture `e` through closure. We do not pass e through __init__
        # to bypass `Exception.__new__` magic which add `__str__` artifacts.
        return getattr(e, name)

      __repr__ = BaseException.__repr__
      __str__ = BaseException.__str__

    WrappedException.__name__ = type(e).__name__
    WrappedException.__qualname__ = type(e).__qualname__
    WrappedException.__module__ = type(e).__module__
    new_exception = WrappedException(msg)
    # Propagate the eventual context
    cause = e.__cause__ or e.__context__
    raise new_exception.with_traceback(e.__traceback__) from cause
  # Otherwise, modify the exception in-place
  elif len(e.args) <= 1:
    exception_msg = e.args[0] if e.args else ''
    e.args = (f'{prefix}{exception_msg}{suffix}',)
    raise  # pylint: disable=misplaced-bare-raise
  # If there is more than 1 args, concatenate the message with other args
  # For example: raise ValueError(123, my_object)
  # will raise:
  # ValueError: ('prefix', 123, <object X at 0x7f907006ce70>)
  else:
    # Pass all arguments but filter empty strings (to filter empty
    # prefix/suffix)
    e.args = tuple(
        p for p in (prefix, *e.args, suffix)  if not isinstance(p, str) or p
    )
    raise  # pylint: disable=misplaced-bare-raise


@contextlib.contextmanager
def try_reraise(
    prefix: Union[None, str, Callable[[], str]] = None,
    suffix: Union[None, str, Callable[[], str]] = None,
):
  """Context manager which reraise exceptions with an additional message.

  Contrary to `raise ... from ...` and `raise Exception().with_traceback(tb)`,
  this function tries to modify the original exception, to avoid nested
  `During handling of the above exception, another exception occurred:`
  stacktraces. This result in cleaner more compact error messages.

  Args:
    prefix: Prefix to add to the exception message. Can be a function for
      lazy-evaluation.
    suffix: Suffix to add to the exception message. Can be a function for
      lazy-evaluation.

  Yields:
    None
  """
  try:
    yield
  except Exception as e:  # pylint: disable=broad-except
    # Lazy-evaluate function
    prefix = prefix() if callable(prefix) else prefix
    suffix = suffix() if callable(suffix) else suffix
    reraise(e, prefix=prefix, suffix=suffix)


class cached_property(property):  # pylint: disable=invalid-name
  """Backport of `functools.cached_property`.

  Warning: This should only be used in non-mutable objects.

  """

  def __get__(self, obj, objtype=None):
    # See https://docs.python.org/3/howto/descriptor.html#properties
    if obj is None:
      return self
    if self.fget is None:  # pytype: disable=attribute-error
      raise AttributeError('Unreadable attribute.')
    attr = '__cached_' + self.fget.__name__  # pytype: disable=attribute-error
    cached = getattr(obj, attr, None)
    if cached is None:
      cached = self.fget(obj)  # pytype: disable=attribute-error
      # Use `object.__setattr__` for compatibility with frozen dataclasses
      object.__setattr__(obj, attr, cached)
    return cached


if typing.TYPE_CHECKING:
  # TODO(b/171883689): There is likelly better way to annotate descriptors

  def cached_property(fn: Callable[[Any], _T]) -> _T:  # pylint: disable=function-redefined
    return fn(None)
