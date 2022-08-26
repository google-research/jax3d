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

"""JAX implementations common metrics in NeRF code."""

import jax.numpy as jnp


def psnr(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Compute the Peak Signal-to-Noise Ratio between two values."""
  return -10.0 * jnp.log10(jnp.mean((y - x)**2))
