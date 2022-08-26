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

"""Common modules and utilities for transformer attention mechanisms."""

import einops
import flax.linen as nn
import gin
import jax.numpy as jnp


class SingleHeadAttention(nn.Module):
  """Layer for computing single-head attention."""

  @nn.compact
  def __call__(self, keys, values, queries, return_attention=False):
    """Compute the result of attention given queries, keys and values.

    Args:
      keys: Per-token key vectors.
      values: Per-token value vectors.
      queries: Query vectors.
      return_attention: If true, also return attention matrices.

    Returns:
      values_out: Vectors resulting from applying attention to values.
      attention: Optional attention matrix values.
    """
    attention = jnp.einsum("B ... Q Z, B T Z -> B ... Q T", queries, keys)
    attention = nn.softmax(attention / jnp.sqrt(keys.shape[-1]))

    values_out = jnp.einsum("B ... Q T, B T Z -> B ... Q Z", attention, values)

    if return_attention:
      return values_out, attention
    else:
      return values_out


@gin.configurable
class MultiHeadAttention(nn.Module):
  """Layer for computing multi-head attention."""
  n_heads: int = 16

  @nn.compact
  def __call__(self, keys, values, queries, return_attention=False):
    """Compute the result of attention given queries, keys and values.

    Args:
      keys: Per-token key vectors.
      values: Per-token value vectors.
      queries: Query vectors.
      return_attention: If true, also return attention matrices.

    Returns:
      values_out: Vectors resulting from applying attention to values.
      attention: Optional attention matrix values.
    """
    head_key_dim = keys.shape[-1] // self.n_heads
    keys = einops.rearrange(
        keys, "... T (H Z) -> ... H T Z", H=self.n_heads, Z=head_key_dim)
    queries = einops.rearrange(
        queries, "... Q (H Z) -> ... H Q Z", H=self.n_heads, Z=head_key_dim)

    head_value_dim = values.shape[-1] // self.n_heads
    values = einops.rearrange(
        values, "... T (H Z) -> ... H T Z", H=self.n_heads, Z=head_value_dim)

    attention = jnp.einsum("B ... H Q Z, B H T Z -> B ... H Q T", queries, keys)
    attention = nn.softmax(attention / jnp.sqrt(keys.shape[-1]))

    values_out = jnp.einsum("B ... H Q T, B H T Z -> B ... H Q Z", attention,
                            values)
    values_out = einops.rearrange(values_out, "... H Q Z -> ... Q (H Z)")

    if return_attention:
      return values_out, attention
    else:
      return values_out
