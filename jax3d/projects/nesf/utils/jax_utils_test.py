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

"""Tests for jax3d.projects.nesf.utils.jax_utils."""

import jax.numpy as jnp
from jax3d.projects.nesf.utils import jax_utils


def test_shape_dtype_repr():
  assert repr(jax_utils.ShapeDtypeStruct((), jnp.float32)) == 'f32[]'
  assert repr(jax_utils.ShapeDtypeStruct((1, 3), jnp.uint8)) == 'ui8[1 3]'
  assert repr(jax_utils.ShapeDtypeStruct((), jnp.complex64)) == 'complex64[]'
  # `str()` works too:
  assert str(jax_utils.ShapeDtypeStruct((1,), jnp.int32)) == 'i32[1]'
