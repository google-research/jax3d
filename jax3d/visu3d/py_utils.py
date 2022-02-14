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

from __future__ import annotations

import collections
from typing import Callable, Iterable, TypeVar

# If these are reused, we could move them to `epy`.

_K = TypeVar('_K')
_Tin = TypeVar('_Tin')
_Tout = TypeVar('_Tout')


def identity(x: _Tin) -> _Tin:
  """Pass through function."""
  return x


def groupby(
    iterable: Iterable[_Tin],
    *,
    key: Callable[[_Tin], _K],
    value: Callable[[_Tin], _Tout] = identity,
) -> dict[_K, list[_Tout]]:
  """Similar to `itertools.groupby` but return result as a `dict()`.

  Example:

  ```python
  out = py_utils.groupby(
      ['555', '4', '11', '11', '333'],
      key=len,
      value=int,
  )
  # Order is consistent with above
  assert out == {
      3: [555, 333],
      1: [4],
      2: [11, 11],
  }
  ```

  Other difference with `itertools.groupby`:

   * Iterable do not need to be sorted. Order of the original iterator is
     preserved in the group.
   * Transformation can be applied to the value too

  Args:
    iterable: The iterable to group
    key: Mapping applied to group the values (should return a hashable)
    value: Mapping applied to the values

  Returns:
    The dict
  """
  groups = collections.defaultdict(list)
  for v in iterable:
    groups[key(v)].append(value(v))
  return dict(groups)
