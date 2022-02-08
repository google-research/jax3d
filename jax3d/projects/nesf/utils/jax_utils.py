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

"""Jax utils."""

import jax

from jax3d.projects.nesf import typing as jtyping


class ShapeDtypeStruct(jax.ShapeDtypeStruct):
  """Typing annotation containing the shape."""

  def __repr__(self) -> str:
    shape_str = ' '.join([str(s) for s in self.shape])
    dtype_str = jtyping.DTYPE_NP_TO_COMPACT_STR.get(self.dtype, self.dtype.name)
    return f'{dtype_str}[{shape_str}]'

  __str__ = __repr__
