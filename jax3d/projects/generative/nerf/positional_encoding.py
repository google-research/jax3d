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

"""Implementation of sinusoidal positional encoding."""

from typing import Callable, Optional

from etils.array_types import FloatArray
import jax.numpy as jnp


def sinusoidal(
    position: FloatArray,
    minimum_frequency_power: int,
    maximum_frequency_power: int,
    include_identity: bool = False,
    filter_fn: Optional[Callable[[FloatArray],
                                 FloatArray]] = None) -> FloatArray:
  """Computes the psotional encoding value from sample positions.

  Arguments:
    position: [..., D] sample positions.
    minimum_frequency_power: Minimum value of p in frequency = 2^p.
    maximum_frequency_power: Maximum value of p in frequency = 2^p.
    include_identity: Whether to include identity mapping in the encoding.
    filter_fn: A mapping from frequency to weight for filtering.

  Returns:
    The per-sample encoding vectors.
  """
  # Compute the sinusoidal encoding components
  frequency = 2.0**jnp.arange(minimum_frequency_power, maximum_frequency_power)
  angle = position[..., None, :] * frequency[:, None]
  encoding = jnp.sin(jnp.stack([angle, angle + 0.5 * jnp.pi], axis=-2))

  # Filter components according to frequency
  if filter_fn is not None:
    response = filter_fn(frequency)
    encoding *= jnp.concatenate([response, response], axis=0)[:, None]

  # Flatten encoding dimensions
  encoding = encoding.reshape(*position.shape[:-1], -1)

  # Add identity component
  if include_identity:
    encoding = jnp.concatenate([position, encoding], axis=-1)

  return encoding
