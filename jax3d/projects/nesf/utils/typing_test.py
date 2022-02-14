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

"""Tests for jax3d.projects.nesf.utils.typing_examples."""

import jax.numpy as jnp
from jax3d.projects.nesf.utils.typing import Array, f32, ui8  # pylint: disable=g-multiple-import
import pytest


@pytest.mark.parametrize(
    'alias, repr_, shape, dtype',
    [
        (Array, 'Array[...]', '...', None),  # No dtype/shape defined
        (Array[''], 'Array[]', '', None),
        (Array['h w c'], 'Array[h w c]', 'h w c', None),
        (f32, 'f32[...]', '...', jnp.float32),  # float32
        (f32['x'], 'f32[x]', 'x', jnp.float32),
        (ui8['h w'], 'ui8[h w]', 'h w', jnp.uint8),
        (f32[1], 'f32[1]', '1', jnp.float32),  # int
        (f32[()], 'f32[]', '', jnp.float32),  # tuple
        (f32[1, 3], 'f32[1 3]', '1 3', jnp.float32),  # tuple[int]
        (f32[4, 'h', 'w'], 'f32[4 h w]', '4 h w', jnp.float32),  # tuple[str]
        (f32[...], 'f32[...]', '...', jnp.float32),  # With elipsis
        (f32[..., 3], 'f32[... 3]', '... 3', jnp.float32),
    ],
)
def test_array_alias(alias, repr_, shape, dtype):
  assert repr(alias) == repr_
  assert str(alias) == repr_
  assert alias.shape == shape
  assert alias.dtype == dtype


def test_array_eq():
  assert f32['h w'] == f32['h', 'w']
  assert f32['1 2 3'] == f32[1, 2, 3]
  assert f32[...] == f32['...']

  assert f32['h w'] != f32['h c']
  assert f32['h w'] != Array['h w']
  assert f32['h w'] != ui8['h w']

  assert {f32['h w'], f32['h w'], f32['h', 'w']} == {f32['h w']}
