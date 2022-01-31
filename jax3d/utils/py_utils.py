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
import functools
from typing import Any

from etils import epy

__all__ = [
    'cached_property',
    'decorator_with_option',
    'reraise',
    'try_reraise',
]


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


reraise = epy.reraise
try_reraise = epy.maybe_reraise
cached_property = epy.cached_property
