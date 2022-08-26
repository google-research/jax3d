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

"""Model definitions for 2D autoencoder."""

from typing import Any, Dict, Iterable, Sequence, Tuple, Union

import einops
from etils.array_types import PRNGKey
import flax
import flax.linen as nn
import gin
import jax.numpy as jnp
from jax3d.projects.generative.nerf import attention
from jax3d.projects.generative.nerf import nerf


_EPS = 1e-6


def _iterable_prod(items: Iterable[int]) -> int:
  product = int(1)
  for item in items:
    product *= item
  return product


@gin.configurable
class TransformerDecoder(nn.Module):
  """Transformer Neural field decoder module."""
  decoder_stages: int = 5
  mlp_width: int = 256
  mlp_depth: int = 3
  key_width: int = 64
  value_width: int = 256
  head_width: int = 256
  head_depth: int = 2
  use_single_head: bool = False

  @nn.compact
  def __call__(self, positions, latent_tokens):
    with gin.config_scope("attention"):
      posenc = nerf.PositionalEncoding()(positions)

    net = nn.Dense(self.value_width)(posenc)

    for _ in range(self.decoder_stages):
      skip = net

      keys = nn.Dense(self.key_width)(latent_tokens)
      values = nn.Dense(self.value_width)(latent_tokens)
      queries = nn.Dense(keys.shape[-1])(net)

      if self.use_single_head:
        attention_layer = attention.SingleHeadAttention()
      else:
        attention_layer = attention.MultiHeadAttention()

      net = attention_layer(keys, values, queries)
      net = net + skip
      net = nn.LayerNorm()(net)
      skip = net

      for _ in range(self.mlp_depth):
        net = nn.Dense(self.mlp_width)(net)
        net = nn.relu(net)
      net = nn.Dense(self.value_width)(net)

      net = net + skip
      net = nn.LayerNorm()(net)

    for _ in range(self.head_depth):
      net = nn.Dense(self.head_width)(net)
      net = nn.relu(net)

    rgb = nn.Dense(3)(net)
    rgb = nn.sigmoid(rgb)

    return rgb


@gin.configurable
class HyperNetworkDecoder(nn.Module):
  """HyperNetwork Neural field decoder module."""
  mlp_width: int = 64
  mlp_depth: int = 8
  hyper_width: int = 64
  hyper_depth: int = 1
  split_latent_to_skips: bool = True

  @nn.compact
  def __call__(self, positions, latent_tokens):
    with gin.config_scope("attention"):
      posenc = nerf.PositionalEncoding()(positions)

    latent = jnp.tile(latent_tokens[:, None],
                      (1, _iterable_prod(positions.shape[1:-1]), 1, 1))
    latent = latent.reshape(positions.shape[0], *positions.shape[1:-1], -1)

    if self.split_latent_to_skips:
      latent_skips = jnp.split(latent, self.mlp_depth, axis=-1)
    else:
      latent_skips = [latent] * self.mlp_depth

    net = posenc
    for i in range(self.mlp_depth):
      hyper_net = latent_skips[i]
      for _ in range(self.hyper_depth):
        hyper_net = nn.Dense(self.hyper_width)(hyper_net)
        hyper_net = nn.LayerNorm()(hyper_net)
        hyper_net = nn.relu(hyper_net)

      in_units = net.shape[-1]
      out_units = self.mlp_width

      w = nn.Dense(in_units * out_units)(hyper_net)
      w = einops.rearrange(w, "... (N M) -> ... N M", N=out_units, M=in_units)
      b = nn.Dense(out_units)(hyper_net)

      net = jnp.einsum("... M, ... N M -> ... N", net, w) + b
      net = nn.LayerNorm()(net)
      net = nn.relu(net)

    rgb = nn.Dense(3)(net)
    rgb = nn.sigmoid(rgb)

    return rgb


@gin.configurable
class MLPDecoder(nn.Module):
  """MLP Neural field decoder module."""
  mlp_width: int = 256
  mlp_depth: int = 8
  skips: Sequence[int] = (0, 2, 4, 6)
  split_latent_to_skips: bool = True

  @nn.compact
  def __call__(self, positions, latent_tokens):
    with gin.config_scope("attention"):
      posenc = nerf.PositionalEncoding()(positions)

    latent = jnp.tile(latent_tokens[:, None],
                      (1, _iterable_prod(positions.shape[1:-1]), 1, 1))
    latent = latent.reshape(positions.shape[0], *positions.shape[1:-1], -1)

    if self.split_latent_to_skips:
      latent_skips = jnp.split(latent, len(self.skips), axis=-1)
    else:
      latent_skips = [latent] * len(self.skips)

    net = jnp.zeros_like(latent[..., :0])

    j = 0
    for i in range(self.mlp_depth):
      if i in self.skips:
        net = jnp.concatenate([net, latent_skips[j], posenc], axis=-1)
        j += 1

      net = nn.Dense(self.mlp_width)(net)
      net = nn.relu(net)

    rgb = nn.Dense(3)(net)
    rgb = nn.sigmoid(rgb)

    return rgb


@gin.configurable
class CNNEncoder(nn.Module):
  """Vision transformer-style image encoder."""
  depth: int = 4
  filter_size: Tuple[int, int] = (3, 3)
  num_tokens: int = 64
  token_dim: int = 128

  @nn.compact
  def __call__(self, image):
    net = 2 * (image - 0.5)

    for i in range(self.depth):
      net = nn.Conv(16 * 2**i, self.filter_size, padding="SAME")(net)
      net = nn.GroupNorm(num_groups=8)(net)
      net = nn.leaky_relu(net)
      net = nn.max_pool(net, window_shape=(2, 2), strides=(2, 2))

    # Extract patches as tokens
    sqrt_t = int(self.num_tokens**(0.5))
    tokens = einops.rearrange(
        net, "N (ht A) (wt B) C -> N (ht wt) (A B C)", ht=sqrt_t, wt=sqrt_t)

    keys = nn.Dense(64)(tokens)
    values = nn.Dense(self.token_dim)(tokens)
    queries = nn.Dense(64)(tokens)

    tokens_out = attention.MultiHeadAttention()(keys, values, queries)
    tokens_out = nn.LayerNorm()(tokens_out)

    return tokens_out


ModelParameters = Union[Dict[str, Any], flax.core.FrozenDict[str, Any]]


@gin.configurable()
class Model(nn.Module):
  """2D image autoencoder model."""
  model_type: str = "mlp"

  def setup(self):
    self.encoder = CNNEncoder()
    if self.model_type == "transformer":
      self.decoder = TransformerDecoder()
    elif self.model_type == "mlp":
      self.decoder = MLPDecoder()
    elif self.model_type == "hypernetwork":
      self.decoder = HyperNetworkDecoder()

  def __call__(self, image, pixels=None, step=None, is_training=False):
    """Forward pass for the model."""
    latent_tokens = self.encoder(image)

    if pixels is None:
      x = jnp.linspace(0.0, 1.0, image.shape[2])
      y = jnp.linspace(0.0, 1.0, image.shape[1])
      pixels = jnp.stack(jnp.meshgrid(x, y, indexing="xy"), axis=-1)
      pixels = jnp.tile(pixels[None], (image.shape[0], 1, 1, 1))
      pixels = einops.rearrange(pixels, "B H W uv -> B (H W) uv")
      out_shape = image.shape
    else:
      out_shape = tuple(pixels.shape[:-1]) + (3,)

    rgb = self.decoder(pixels, latent_tokens)
    rgb = rgb.reshape(*out_shape)

    return rgb

  def initialize_parameters(self, rng_key: PRNGKey,
                            image_size) -> ModelParameters:
    batch_size = 7
    pixel_batch_size = 11
    return self.init(
        rng_key,
        jnp.zeros((batch_size, *image_size, 3)),
        jnp.zeros((batch_size, pixel_batch_size, 2)))
