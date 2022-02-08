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

"""Numpy utils."""

import jax.numpy as jnp
import numpy as np


def is_dtype_str(dtype) -> bool:
  """Returns True if the dtype is `str`."""
  if type(dtype) is object:  # tf.string.as_numpy_dtype is object  # pylint: disable=unidiomatic-typecheck
    return True
  return np.dtype(dtype).kind in {'O', 'S', 'U'}


def is_array_str(x) -> bool:
  """Returns True if the given array is a `str` array."""
  # `Tensor(shape=(), dtype=tf.string).numpy()` returns `bytes`.
  if isinstance(x, (bytes, str)):
    return True
  elif not isinstance(x, (np.ndarray, jnp.ndarray)):
    raise TypeError(f'Cannot check `str` on non-array {type(x)}: {x!r}')
  return is_dtype_str(x.dtype)
