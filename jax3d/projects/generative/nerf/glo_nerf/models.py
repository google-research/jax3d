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

"""Model definitions for GLO NeRF."""

import collections
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Union

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
from jax3d.projects.generative.common import image_utility
from jax3d.projects.generative.nerf import attention
from jax3d.projects.generative.nerf import camera as jax_camera
from jax3d.projects.generative.nerf import nerf
from jax3d.public_api import volume_rendering
import numpy as np


_EPS = 1e-6
# Rays are origin + direction. Therefore 3 + 3 = 6 DOFs.
RAY_DIMS = 6
# Number of samples we can comfortably evaluate. This was originally determined
# on V100s.  More testing should be done to find an optimal value for TPUs.
SAMPLES_PER_DEVICE = 2**13


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


@gin.configurable
class BackgroundModel(nn.Module):
  """Simple MLP model for predicting background color from ray direction.

  Attributes:
    depth: Number of layers in the MLP.
    width: Number of neurons in each MLP layer.
  """
  depth: int = 5
  width: int = 256

  @nn.compact
  def __call__(self, latent_codes, ray_directions):
    """Compute the background colors for a set of rays.

    Args:
      latent_codes: [K, Z] per-image latent codes.
      ray_directions: [K, R, 3] per-ray unit direction vectors.

    Returns:
      [K, R, 3] per-ray RGB values.
    """
    with gin.config_scope("background_encoding"):
      direction_encoding = nerf.PositionalEncoding()(ray_directions)

    ray_count = ray_directions.shape[1]
    net = jnp.concatenate([
        direction_encoding,
        jnp.tile(latent_codes[:, None], (1, ray_count, 1))
    ],
                          axis=-1)

    for _ in range(self.depth):
      net = nn.Dense(self.width)(net)
      net = nn.relu(net)

    rgb = nn.Dense(3)(net)
    return nn.sigmoid(rgb)


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
  def __call__(self, positions, latent_tokens):
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

    with gin.config_scope("attention"):
      posenc = nerf.PositionalEncoding()(positions)

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

    density = nn.Dense(1)(net)
    density = nn.softplus(density)[..., 0]

    rgb = nn.Dense(3)(net)
    rgb = nn.sigmoid(rgb)

    return density, rgb


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
  def __call__(self, positions, latent_tokens):
    with gin.config_scope("attention"):
      posenc = nerf.PositionalEncoding()(positions)

    num_tokens = latent_tokens.shape[-2]
    token_embeddings = self.param("token_embeddings", nn.initializers.normal(),
                                  (num_tokens, self.token_embedding_dim),
                                  jnp.float32)
    token_embeddings = einops.repeat(
        token_embeddings, "T Z -> K T Z", K=latent_tokens.shape[0])

    latent_tokens = jnp.concatenate([latent_tokens, token_embeddings], axis=-1)

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

      w = nn.Dense(in_units * out_units)(hyper_net / 10.0)
      w = einops.rearrange(w, "... (N M) -> ... N M", N=out_units, M=in_units)
      b = nn.Dense(out_units)(hyper_net / 10.0)

      net = jnp.einsum("... M, ... N M -> ... N", net, w) + b
      net = nn.LayerNorm()(net)
      net = nn.relu(net)

    density = nn.Dense(1)(net)
    density = nn.softplus(density - 5.0)[..., 0]

    rgb = nn.Dense(3)(net)
    rgb = nn.sigmoid(rgb)

    return density, rgb


@gin.configurable
class MLPDecoder(nn.Module):
  """MLP Neural field decoder module."""
  mlp_width: int = 256
  mlp_depth: int = 8
  skips: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7)
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

    density = nn.Dense(1)(net)
    density = nn.softplus(density)[..., 0]

    rgb = nn.Dense(3)(net)
    rgb = nn.sigmoid(rgb)

    return density, rgb


ModelParameters = Union[Dict[str, Any], flax.core.FrozenDict[str, Any]]


@struct.dataclass
class ModelInputs:
  latent_tokens: jnp.ndarray
  transformer_cache: Optional[Dict[str, jnp.ndarray]] = None


@gin.configurable()
class Model(nn.Module):
  """GLO NeRF Model."""
  initial_samples_per_ray: int = 64
  importance_samples_per_ray: int = 64
  use_background_model: bool = True
  model_type: str = "mlp"

  predict_gamma_encoded_rgb: bool = True
  interval_length: float = 4.0

  def setup(self):
    if self.model_type == "transformer":
      self.decoder = TransformerDecoder()
    elif self.model_type == "mlp":
      self.decoder = MLPDecoder()
    elif self.model_type == "hypernetwork":
      self.decoder = HyperNetworkDecoder()

    if self.use_background_model:
      self.background_model = BackgroundModel()

  def evaluate_nerf(
      self,
      points: FloatArray["N", ..., 3],
      directions: FloatArray["N", ..., 3],
      inputs: ModelInputs,
      step: int,
      noise_rng: Optional[PRNGKey] = None) -> Dict[str, FloatArray]:
    """NeRF decoder forward pass."""
    density, rgb = self.decoder(points, inputs.latent_tokens)

    results = {"sample_density": density, "sample_values": {"rgb": rgb,}}
    return results

  def surface_normal_from_density(self, points: FloatArray["N", ...,
                                                           3], inputs: ...,
                                  step: int) -> FloatArray["N", ..., 3]:
    """Compute the normals of the density field at the given points."""

    # Input points are of the form [ N, ..., 3]. Compute the product of the
    # inner dimensions ("..."), and flatten.
    inner_dims_product = _iterable_prod(points.shape[1:-1])
    reshaped_points = points.reshape(points.shape[0], inner_dims_product, 3)

    # Function to compute the scalar density
    def density_fn(fn_points):
      density, _ = self.decoder(fn_points, inputs.latent_tokens)
      return jnp.sum(density)

    # Compute the gradient via jax.grad()
    density_gradient = jax.grad(density_fn)(reshaped_points)
    normals = -density_gradient / (
        jnp.sqrt(jnp.sum(density_gradient**2, axis=-1, keepdims=True) + _EPS))

    # Reshape points back to the original input shape
    return normals.reshape(points.shape)

  def __call__(self,
               inputs: ModelInputs,
               rays: FloatArray["K R", 6],
               rng=None,
               near=None,
               far=None,
               initial_samples_per_ray=None,
               importance_samples_per_ray=None,
               step=None,
               return_additional_sample_data=False,
               is_training=False):
    """GLO NeRF volume rendering forward pass."""
    return_values = {}

    origins = rays[..., 0:3]
    directions = rays[..., 3:6]

    num_rays = origins.shape[1]

    # Set near and far based on the bounding sphere of the [-1, 1] cube.
    origin_distances = jnp.linalg.norm(origins, axis=-1)
    if near is None:
      near = jnp.maximum(origin_distances - self.interval_length / 2, 0.0)
    else:
      near = jnp.tile(near.reshape(-1, 1), (1, num_rays))
    if far is None:
      far = origin_distances + self.interval_length / 2
    else:
      far = jnp.tile(far.reshape(-1, 1), (1, num_rays))

    if initial_samples_per_ray is None:
      initial_samples_per_ray = self.initial_samples_per_ray
    if importance_samples_per_ray is None:
      importance_samples_per_ray = self.importance_samples_per_ray

    if rng is not None:
      rngs = jax.random.split(rng, 4)
    else:
      rngs = tuple(None for _ in range(4))
    (initial_sampling_rng, importance_sampling_rng, initial_noise_rng,
     importance_noise_rng) = rngs

    (initial_sample_depths,
     initial_sample_coordinates) = volume_rendering.sample_along_rays(
         ray_origins=origins,
         ray_directions=directions,
         near=near,
         far=far,
         sample_count=initial_samples_per_ray,
         deterministic=initial_sampling_rng is None,
         rng=initial_sampling_rng)

    initial_sample_directions = einops.repeat(
        directions, "K R xyz -> K R S xyz", S=initial_samples_per_ray, xyz=3)

    initial_nerf_result = self.evaluate_nerf(initial_sample_coordinates,
                                             initial_sample_directions, inputs,
                                             step, initial_noise_rng)

    initial_render_results = volume_rendering.volume_rendering(
        sample_values=(),  # We ignore the accumulated RGB from initial samples
        sample_density=initial_nerf_result["sample_density"],
        depths=initial_sample_depths,
        opaque_final_sample=not self.use_background_model)

    # Compute interval centers to use as bin edges for hierarchical sampling
    bin_edges = (initial_sample_depths[..., 1:] +
                 initial_sample_depths[..., :-1]) / 2
    pdf_values = initial_render_results.sample_weights[..., 1:-1]

    # Draw importance samples from empirical PDF of initial sample weights
    importance_sample_depths = volume_rendering.sample_piecewise_constant_pdf(
        bin_edges=bin_edges,
        weights=pdf_values,
        sample_count=importance_samples_per_ray,
        deterministic=importance_sampling_rng is None,
        rng=importance_sampling_rng)

    # Prevent importance sample positions from affecting density gradients
    importance_sample_depths = jax.lax.stop_gradient(importance_sample_depths)

    if jax.default_backend() == "tpu" and is_training:
      # This branch handles the case of training on TPU, where backpropagating
      # through a sort operation incurs unacceptable slowdown due to scatter
      # operations. In this case it is more efficient to just re-query the NeRF
      # at the initial sample points.
      combined_sample_depths = jnp.concatenate(
          [initial_sample_depths, importance_sample_depths], axis=2)
      combined_sample_depths = jnp.sort(combined_sample_depths, axis=2)

      combined_sample_coordinates = (
          origins[..., None, :] +
          combined_sample_depths[..., None] * directions[..., None, :])

      total_samples_per_ray = (
          initial_samples_per_ray + importance_samples_per_ray)
      combined_sample_directions = einops.repeat(
          directions, "K R xyz -> K R S xyz", S=total_samples_per_ray, xyz=3)

      combined_nerf_result = self.evaluate_nerf(combined_sample_coordinates,
                                                combined_sample_directions,
                                                inputs, step,
                                                importance_noise_rng)

    else:
      # In this branch, we evaluate the NeRF separately at the initial and
      # importance sample locations, and we then need to merge them together in
      # depth-sorted order before the final volume rendering integration.
      # Due to the extreme inefficiency of gather and scatter operations on TPU,
      # we need to fit all the depth and sample values into a single sort call,
      # which is fast for eval on TPU using the efficient XLA sort operation.
      # This requires some tensor shape gymnastics to fit everything into two
      # key and value tensors with the same shape.

      importance_sample_coordinates = (
          origins[..., None, :] +
          importance_sample_depths[..., None] * directions[..., None, :])

      importance_sample_directions = einops.repeat(
          directions,
          "K R xyz -> K R S xyz",
          S=importance_samples_per_ray,
          xyz=3)

      importance_nerf_result = self.evaluate_nerf(importance_sample_coordinates,
                                                  importance_sample_directions,
                                                  inputs, step,
                                                  importance_noise_rng)

      # We first append importance sample depths and values along the sample
      # axis so they can be combined together in order by a sort operation.
      concatenated_sample_depths = jnp.concatenate(
          [initial_sample_depths, importance_sample_depths], axis=2)

      # We need to reduce the result tree structure to a list of leaves to
      # operate on the leaves as a sequence.
      initial_nerf_result_list, result_treedef = jax.tree_util.tree_flatten(
          initial_nerf_result)
      importance_nerf_result_list, _ = jax.tree_util.tree_flatten(
          importance_nerf_result)

      # Once we have the leaves as a list, we can append them as well.
      concatenated_nerf_result_list = list(
          jnp.concatenate([c, f], axis=2)
          for (c,
               f) in zip(initial_nerf_result_list, importance_nerf_result_list))

      # We then flatten all the per-sample values into a single dimension,
      # saving the original shapes so they can be restored.
      result_shapes = []
      results_flattened = []
      for result in concatenated_nerf_result_list:
        result_shapes.append(result.shape)
        results_flattened.append(
            einops.rearrange(result, "K R S ... -> K R S (...)"))

      # Now that they are broadcast-compatible, we can pack all sample values
      # into a single tensor.
      results_packed = jnp.concatenate(results_flattened, axis=3)

      # XLA's sort operation wants keys and values to have exactly the same
      # shape, so we broadcast the depths to match the packed sample values.
      depths_packed = einops.repeat(
          concatenated_sample_depths,
          "K R S -> K R S P",
          P=results_packed.shape[-1])

      combined_depths_packed, combined_results_packed = jax.lax.sort(
          [depths_packed, results_packed], dimension=2)

      # We can now ignore the tiling on the sorted depth values.
      combined_sample_depths = combined_depths_packed[..., 0]

      # To get back the result tree structure, we start by unpacking the sorted,
      # flattened values into individual tensors.
      elements_per_result = list(np.prod(s[3:]) for s in result_shapes)
      split_indices = np.cumsum(elements_per_result)[:-1]
      results_unpacked = jnp.split(
          combined_results_packed, split_indices, axis=3)

      # After unpacking, we can restore their original unflattened shapes.
      combined_nerf_result_list = []
      for result, shape in zip(results_unpacked, result_shapes):
        combined_nerf_result_list.append(result.reshape(*shape))

      # Finally, we can restore the original result tree structure.
      combined_nerf_result = jax.tree_unflatten(result_treedef,
                                                combined_nerf_result_list)

    if return_additional_sample_data and "jacobian" in combined_nerf_result:
      return_values["jacobian"] = combined_nerf_result["jacobian"]

    # Note than any operation taking a weighted sum of gamma encoded RGB values
    # gives an incorrect result, including the volume rendering and the exposure
    # model below. It can often be overlooked however as the error is lessened
    # when one weight dominates, as it does for solid surfaces. Plus, much
    # released NeRF code ignores this, so is required for comparison.
    render_results = volume_rendering.volume_rendering(
        sample_values=combined_nerf_result["sample_values"],
        sample_density=combined_nerf_result["sample_density"],
        depths=combined_sample_depths,
        opaque_final_sample=not self.use_background_model)

    # We approximate the surface location as the weighted average of sample
    # positions along the ray.
    expected_depth = render_results.ray_depth[..., None]
    surface_points = origins + directions * expected_depth
    return_values["depth"] = expected_depth

    normals = self.surface_normal_from_density(surface_points, inputs, step)
    return_values["analytic_normal"] = normals

    if return_additional_sample_data:
      return_values["sample_weights"] = render_results.sample_weights

    alpha = render_results.ray_alpha[..., None]
    return_values["alpha"] = alpha

    foreground_rgb = render_results.ray_values["rgb"]

    if self.use_background_model:
      background_latent = inputs.latent_tokens[..., 0, :]
      background_rgb = self.background_model(background_latent, directions)
      # foreground_rgb has alpha premultiplied by volume rendering, so we only
      # need to add it to the attenuated background value.
      pixel_rgb = foreground_rgb + (1.0 - alpha) * background_rgb
    else:
      background_rgb = jnp.zeros_like(foreground_rgb)
      pixel_rgb = foreground_rgb

    # Up until this point RGB values could be interpreted a gamma encoded or
    # linear. We now pick an interpretation and optionally apply exposure.
    if self.predict_gamma_encoded_rgb:
      return_values["fg_gamma_rgb"] = foreground_rgb
      return_values["bg_gamma_rgb"] = background_rgb
      return_values["gamma_rgb"] = pixel_rgb
      return_values["linear_rgb"] = image_utility.srgb_gamma_to_linear(
          pixel_rgb)

    else:
      fg_linear_rgb = foreground_rgb
      bg_linear_rgb = background_rgb
      pixel_linear_rgb = pixel_rgb

      return_values["fg_gamma_rgb"] = image_utility.linear_to_srgb_gamma(
          fg_linear_rgb)
      return_values["bg_gamma_rgb"] = image_utility.linear_to_srgb_gamma(
          bg_linear_rgb)
      return_values["linear_rgb"] = pixel_linear_rgb
      return_values["gamma_rgb"] = image_utility.linear_to_srgb_gamma(
          pixel_linear_rgb)

    return return_values

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
                     samples_per_device_per_batch=SAMPLES_PER_DEVICE,
                     initial_samples_per_ray=None,
                     importance_samples_per_ray=None,
                     near=None,
                     far=None,
                     rng=None,
                     step=None):
      replicated_model_parameters = jax_utils.replicate(model_parameters)

      # Prepend a singular dimension to our latent code. We only have one
      # code for our single image (not batches of latent codes during training).
      latent_tokens = inputs.latent_tokens
      latent_tokens = einops.rearrange(latent_tokens, "... -> 1 ...")
      inputs = inputs.replace(latent_tokens=latent_tokens)

      if initial_samples_per_ray is None:
        initial_samples_per_ray = self.initial_samples_per_ray
      if importance_samples_per_ray is None:
        importance_samples_per_ray = self.importance_samples_per_ray

      total_samples_per_ray = (
          initial_samples_per_ray + importance_samples_per_ray)
      rays_per_device_per_batch = (
          samples_per_device_per_batch // total_samples_per_ray)
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

      if rng is not None:
        rngs = jax.random.split(rng, num_batches)
      else:
        rngs = tuple(None for _ in range(num_batches))
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
                "near": near,
                "far": far,
                "rng": rngs[i],
                "step": step
            },
            flax.core.FrozenDict({
                "initial_samples_per_ray": initial_samples_per_ray,
                "importance_samples_per_ray": importance_samples_per_ray,
            }))

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
