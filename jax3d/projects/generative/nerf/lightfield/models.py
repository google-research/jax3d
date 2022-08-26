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

"""Transformer model definitions for Face NeRF project."""

import collections
import math
from typing import Any, Dict, Iterable, Sequence, Union

import einops
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
from etils.array_types import PRNGKey
import flax
from flax import jax_utils
from flax import struct
import flax.linen as nn
import gin
import jax
import jax.numpy as jnp
from jax3d.projects.generative.nerf import attention
from jax3d.projects.generative.nerf import camera as jax_camera
from jax3d.projects.generative.nerf import nerf


_EPS = 1e-6
# Rays are origin + direction. Therefore 3 + 3 = 6 DOFs.
RAY_DIMS = 6


def _schedule_progress(schedule_start, schedule_end, step=None):
  if step is not None:
    duration = 1 + schedule_end - schedule_start
    schedule_progress = (step - schedule_start) / duration
    schedule_progress = jnp.clip(schedule_progress, 0.0, 1.0)
  else:
    schedule_progress = 1.0

  return schedule_progress


def _iterable_prod(items: Iterable[int]) -> int:
  product = int(1)
  for item in items:
    product *= item
  return product


@gin.configurable(denylist=["rays"])
def ray_mapping(rays, use_plucker=True):
  """Ray-level positional encoding."""
  if not use_plucker:
    origins = rays[..., 0:3]
    directions = rays[..., 3:6]

    with gin.config_scope("origins"):
      origin_posenc = nerf.PositionalEncoding()(origins)

    with gin.config_scope("directions"):
      direction_posenc = nerf.PositionalEncoding()(directions)

    return jnp.concatenate([origin_posenc, direction_posenc], axis=-1)

  else:
    origins = rays[..., 0:3]
    directions = rays[..., 3:6]

    m = jnp.cross(origins, directions, axis=-1)

    plucker = jnp.concatenate([directions, m], axis=-1)

    with gin.config_scope("plucker"):
      posenc = nerf.PositionalEncoding()(plucker)

    return posenc


@gin.configurable
class TransformerDecoder(nn.Module):
  """Transformer Neural field decoder module."""
  decoder_stages: int = 5
  token_embedding_dim: int = 64
  trunk_width: int = 256
  trunk_depth: int = 3
  mlp_width: int = 256
  mlp_depth: int = 3
  key_width: int = 256
  head_width: int = 256
  head_depth: int = 2
  use_single_head: bool = False

  @nn.compact
  def __call__(self, rays, latent_tokens):
    num_tokens = latent_tokens.shape[-2]
    token_embeddings = self.param("token_embeddings", nn.initializers.normal(),
                                  (num_tokens, self.token_embedding_dim),
                                  jnp.float32)
    token_embeddings = einops.repeat(
        token_embeddings, "T Z -> K T Z", K=latent_tokens.shape[0])

    net = jnp.concatenate([latent_tokens, token_embeddings], axis=-1)
    for _ in range(self.trunk_depth):
      net = nn.Dense(self.trunk_width)(net)
      net = nn.LayerNorm()(net)
      net = nn.relu(net)

    mapped_tokens = net

    posenc = ray_mapping(rays)

    net = nn.Dense(self.mlp_width)(posenc)

    for _ in range(self.decoder_stages):
      skip = net

      keys = nn.Dense(self.key_width)(mapped_tokens)
      values = nn.Dense(self.mlp_width)(mapped_tokens)
      queries = nn.Dense(self.key_width)(net)

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

      net = nn.Dense(self.mlp_width)(net)

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
  token_embedding_dim: int = 64
  split_latent_to_skips: bool = True

  @nn.compact
  def __call__(self, rays, latent_tokens):
    posenc = ray_mapping(rays)

    num_tokens = latent_tokens.shape[-2]
    token_embeddings = self.param("token_embeddings", nn.initializers.normal(),
                                  (num_tokens, self.token_embedding_dim),
                                  jnp.float32)
    token_embeddings = einops.repeat(
        token_embeddings, "T Z -> K T Z", K=latent_tokens.shape[0])

    latent_tokens = jnp.concatenate([latent_tokens, token_embeddings], axis=-1)

    latent = jnp.tile(latent_tokens[:, None],
                      (1, _iterable_prod(rays.shape[1:-1]), 1, 1))
    latent = latent.reshape(rays.shape[0], *rays.shape[1:-1], -1)

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
  skips: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7)
  split_latent_to_skips: bool = True

  @nn.compact
  def __call__(self, rays, latent_tokens):
    posenc = ray_mapping(rays)

    latent = jnp.tile(latent_tokens[:, None],
                      (1, _iterable_prod(rays.shape[1:-1]), 1, 1))
    latent = latent.reshape(rays.shape[0], *rays.shape[1:-1], -1)

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


ModelParameters = Union[Dict[str, Any], flax.core.FrozenDict[str, Any]]


@struct.dataclass
class ModelInputs:
  latent_tokens: jnp.ndarray


@gin.configurable()
class Model(nn.Module):
  """GLO light field model."""
  model_type: str = "mlp"

  def setup(self):
    if self.model_type == "transformer":
      self.decoder = TransformerDecoder()
    elif self.model_type == "mlp":
      self.decoder = MLPDecoder()
    elif self.model_type == "hypernetwork":
      self.decoder = HyperNetworkDecoder()

  def __call__(self,
               inputs: ModelInputs,
               rays: FloatArray["K R", 6],
               rng=None,
               step=None,
               is_training=False):
    """Light field forward pass."""
    rgb = self.decoder(rays, inputs.latent_tokens)
    return {
        "gamma_rgb": rgb
    }

  def create_image_renderer(self):
    """Generate a compiled function to render images of arbitrary resolution.

    Returns:
      A render function with the following signature:

      Args:
        camera: A camera object defining the view to render from.
        latents: A ConditionVariables object with all network conditioning.
        samples_per_device_per_batch: Number of samples to compute on each
          device for each batch. Same as the number of samples passed to each
          model function invocation.
        initial_samples_per_ray: Optional override of NeRF initial sample count.
        importance_samples_per_ray: Optional override of NeRF importance sample
          count.
        near: Optional override of the near plane distance for rendering.
        far: Optional override of the far plane distance for rendering.
        rng: Optional rng key for random sampling and regularization.
        step: The current training iteration for the purposes of scheduling.

      Returns:
        Dictionary containing:
          gamma_rgb: Rendered color image (gamma encoded).
          linear_rgb: Rendered color image (in linear RGB).
          fg_gamma_rgb: Rendered foreground color image (gamma encoded).
          bg_gamma_rgb: Rendered background color image (gamma encoded).
          depth: Rendered depth image.
          alpha: Rendered alpha map.
          analytic_normal: Rendered analytic normal map.
    """

    # Only parallelize over pixels and model parameters, other arguments are the
    # same for all batches. Static and dynamic arguments are passed separately:
    # dynamic arguments consist of values which are the same while rendering an
    # entire frame, but can change between frames without recompiling, whereas a
    # change to a static argument requires a recompile.
    def parallel_function(rays, model_parameters, dynamic_kwargs,
                          static_kwargs):
      return self.apply(
          model_parameters, rays=rays, **dynamic_kwargs, **static_kwargs)

    compiled_parallel_function = jax.pmap(
        parallel_function,
        in_axes=(0, 0, None, None),
        static_broadcasted_argnums=3)

    def render_image(camera,
                     inputs,
                     model_parameters: ModelParameters,
                     rng=None,
                     step=None):
      del rng
      replicated_model_parameters = jax_utils.replicate(model_parameters)

      # Prepend a singular dimension to our latent code. We only have one
      # code for our single image (not batches of latent codes during training).
      latent_tokens = inputs.latent_tokens
      latent_tokens = einops.rearrange(latent_tokens, "... -> 1 ...")
      inputs = inputs.replace(latent_tokens=latent_tokens)

      rays_per_device_per_batch = 2**16
      rays_per_batch = rays_per_device_per_batch * jax.local_device_count()

      width, height = camera["image_size"]
      total_rays = width * height
      num_batches = int(math.ceil(total_rays / rays_per_batch))

      # Offset coordinates by 0.5 to center sample points within each pixel.
      pixel_x_coordinates = jnp.arange(width) + 0.5
      pixel_y_coordinates = jnp.arange(height) + 0.5
      pixel_coordinates = jnp.stack(
          jnp.meshgrid(pixel_x_coordinates, pixel_y_coordinates, indexing="xy"),
          axis=-1)
      pixel_coordinates = pixel_coordinates.reshape(-1, 2)

      origins, directions = jax_camera.pixels_to_rays(camera, pixel_coordinates)
      rays = jnp.concatenate([origins, directions], axis=-1)

      results = collections.defaultdict(list)
      for i in range(num_batches):
        batch_rays = rays[i * rays_per_batch:(i + 1) * rays_per_batch]
        unpadded_batch_size = batch_rays.shape[0]
        padding_amount = rays_per_batch - unpadded_batch_size
        rays_padded = jnp.pad(batch_rays, ((0, padding_amount), (0, 0)))
        ray_batches = rays_padded.reshape(jax.local_device_count(), 1,
                                          rays_per_device_per_batch, 6)
        batch_result = compiled_parallel_function(
            ray_batches, replicated_model_parameters, {
                "inputs": inputs,
                "step": step
            },
            flax.core.FrozenDict({}))

        # pylint: disable=cell-var-from-loop
        def unpad(tensor):
          flattened = tensor.reshape(-1, tensor.shape[-1])
          return flattened[:unpadded_batch_size]

        for key in batch_result:
          results[key].append(unpad(batch_result[key]))

      for key in results:
        result_i = jnp.concatenate(results[key])
        results[key] = result_i.reshape(height, width, *result_i.shape[1:])

      return results

    return render_image

  def initialize_parameters(self, rng_key: PRNGKey, num_tokens,
                            token_dim) -> ModelParameters:
    batch_size = 7
    num_rays = 13  # Prime numbers to help catch shape errors.
    return self.init(
        rng_key,
        ModelInputs(
            latent_tokens=jnp.zeros((batch_size, num_tokens, token_dim))),
        jnp.zeros((batch_size, num_rays, RAY_DIMS)))
