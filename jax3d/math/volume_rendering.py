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

"""Common math functions for volumetric rendering applications."""

import enum
from typing import Any, Optional, Sequence, Tuple, Union

import chex
from etils.array_types import FloatArray
from etils.epy import StrEnum
from etils.etree import Tree
import jax
import jax.numpy as jnp


class SamplingStrategy(StrEnum):
  """Strategies for random sampling in a 1D space like a ray or empirical PDF.

  Possible values:
    STRATIFIED: Distribute samples evenly into regularly spaced bins.
    UNIFORM: Draw samples independently from a uniform distribution.
  """
  STRATIFIED = enum.auto()
  UNIFORM = enum.auto()


def sample_1d(
    *,
    sample_count: int,
    batch_shape: Sequence[int],
    strategy: SamplingStrategy,
    rng: jax.random.KeyArray,
    dtype: jnp.dtype = jnp.float32,
) -> FloatArray["*batch_shape S"]:
  """Samples points from the 1D domain [0, 1) using the specified strategy.

  Note that 'SamplingStrategy.STRATIFIED' will yield samples sorted in ascending
  order, while 'SamplingStrategy.UNIFORM' will not yield samples in any
  particular order.

  Args:
    sample_count: How many samples to draw for each batch element.
    batch_shape: The shape of the leading dimensions of the sample array. Each
      index into these dimensions will correspond to a set of 'sample_count'
      independent samples.
    strategy: How to draw the samples. 'SamplingStrategy.STRATIFIED' will divide
      the [0, 1) range into 'sample_count' bins and draw one sample from a each
      according to a uniform distribution. 'SamplingStrategy.UNIFORM' will draw
      'sample_count' independent samples from the entire domain.
    rng: RNG for the random sampling.
    dtype: The data type for the returned array of samples.

  Returns:
    A batched array containing the samples.
  """
  strategy = SamplingStrategy(strategy)

  # Initialize t with leading bin edges
  t = jnp.linspace(0.0, 1.0, sample_count, endpoint=False, dtype=dtype)
  t = t.reshape(*([1] * len(batch_shape)), sample_count)
  t = jnp.tile(t, tuple(batch_shape) + (1,))

  if strategy == SamplingStrategy.STRATIFIED:
    # Randomly perturb points within depth bins
    perturbation = jax.random.uniform(rng, t.shape, dtype=dtype)
    perturbation /= sample_count
  elif strategy == SamplingStrategy.UNIFORM:
    # Replace t with random samples from [0, 1)
    t = jax.random.uniform(rng, t.shape, dtype=dtype)
    perturbation = 0.0

  t += perturbation
  return jnp.clip(t, 0.0, 1.0 - jnp.finfo(dtype).eps)


def sample_1d_grid(
    *,
    sample_count: int,
    batch_shape: Sequence[int],
    dtype: jnp.dtype = jnp.float32,
) -> FloatArray["*batch_shape S"]:
  """Samples points on a regular grid from the 1D domain [0, 1).

  Args:
    sample_count: How many samples to draw for each batch element.
    batch_shape: The shape of the leading dimensions of the sample array. Each
      index into these dimensions will correspond to a set of 'sample_count'
      independent samples.
    dtype: The data type for the returned array of samples.

  Returns:
    A batched array containing the samples.
  """
  t = jnp.linspace(0.0, 1.0, sample_count, endpoint=False, dtype=dtype)
  t += 0.5 / sample_count
  t = t.reshape(*([1] * len(batch_shape)), sample_count)
  t = jnp.tile(t, tuple(batch_shape) + (1,))
  return t


def sample_along_rays(
    *,
    ray_origins: FloatArray[..., 3],
    ray_directions: FloatArray[..., 3],
    near: Union[float, FloatArray[...]],
    far: Union[float, FloatArray[...]],
    sample_count: int,
    deterministic: bool,
    rng: Optional[jax.random.KeyArray] = None,
    use_linear_disparity: bool = False,
) -> Tuple[FloatArray[..., "S"], FloatArray[..., "S", 3]]:
  """Randomly or uniformly samples positions along rays for volume rendering.

  Args:
    ray_origins: Origin point for each ray in world space.
    ray_directions: Direction vectors for each ray. If not normalized, the ray
      sample depths will be scaled by the magnitude of this vector.
    near: Near cutoff values. Sampling will begin at this distance from the ray
      origin. Can either be a scalar or a shape broadcast-compatible with the
      ray batch dimensions.
    far: Far cutoff values. Sampling will end at this distance from the ray
      origin. Can either be a scalar or a shape broadcast-compatible with the
      ray batch dimensions.
    sample_count: The number of samples to draw for each ray.
    deterministic: Whether to use a grid instead of random stratified sampling.
    rng: RNG for the random sampling. Must be set when deterministic=False.
    use_linear_disparity: If true, sample linearly in disparity (inverse depth)
      which will space points more densely towards the near cutoff. Warning: if
      this flag is set, 'near' MUST be > zero, and should ideally be as close as
      possible to the true minimum depth value for the scene.

  Returns:
    sample_depths: The distance from the corresponding ray origin to each sample
      point.
    sample_positions: The world space coordinates of each sample point.
  """
  dtype = ray_origins.dtype
  batch_shape = ray_origins.shape[:-1]

  # This check can only be done for python scalars. When near is an array,
  # violating this condition will produce NaNs.
  is_py_scalar = isinstance(near, (int, float))
  if use_linear_disparity and is_py_scalar and near <= 0:
    raise ValueError("When use_linear_disparity=True, near MUST be > 0.")

  # Generate uniformly sampled points between zero and one
  if deterministic:
    t = sample_1d_grid(
        sample_count=sample_count, batch_shape=batch_shape, dtype=dtype)
  else:
    t = sample_1d(
        sample_count=sample_count,
        batch_shape=batch_shape,
        strategy=SamplingStrategy.STRATIFIED,
        dtype=dtype,
        rng=rng)

  # Convert to sample depth values along the rays
  near = jnp.broadcast_to(near, batch_shape)[..., None]
  far = jnp.broadcast_to(far, batch_shape)[..., None]
  if use_linear_disparity:
    sample_depths = 1.0 / (1.0 / near * (1.0 - t) + 1.0 / far * t)
  else:
    sample_depths = near + (far - near) * t

  # Compute world-space sample positions
  world_positions = (
      ray_origins[..., None, :] +
      sample_depths[..., None] * ray_directions[..., None, :])

  # In the case of non-normalized ray directions, we need to scale the depths
  # to match the sample positions for correct computation of intervals.
  ray_scales = jnp.linalg.norm(ray_directions, axis=-1)
  world_depths = sample_depths * ray_scales[..., None]

  return world_depths, world_positions


@chex.dataclass(frozen=True)
class VolumeRenderingResult:
  """A struct containing return values from a call to volume_rendering."""
  ray_values: Tree[FloatArray[..., "N"]]
  ray_alpha: FloatArray[...]
  ray_depth: FloatArray[...]
  sample_weights: FloatArray[..., "S"]
  sample_intervals: FloatArray[..., "S"]


def volume_rendering(
    *,
    sample_values: Tree[FloatArray[..., "S N"]],
    sample_density: FloatArray[..., "S"],
    depths: FloatArray[..., "S"],
    background_values: Optional[Any] = None,
    opaque_final_sample: bool = False,
) -> VolumeRenderingResult:
  """Accumulates values sampled along rays from a volumetric field.

  Given some per-sample quantities and densities, this function evaluates the
  classical volume rendering equation to produce per-ray rendered values.

  Args:
    sample_values: A pytree of per-sample quantities to accumulate. The shape of
      each leaf value should be [..., S, N], where N is the dimensionality of
      the quantity, e.g. [..., S, 3] for RGB radiance.
    sample_density: The density of the volumetric representation at each sample
      point. Must be strictly non-negative.
    depths: The distance of each sample from the origin of the ray. This value
      should be in world-space distance coordinates for correct calculation of
      sample opacity.
    background_values: Optional pytree with the structure as 'sample_values' to
      provide a background for transparent pixels. Each leaf value should be
      broadcastable to [..., N].
    opaque_final_sample: If true, force the opacity of the last sample along
      each ray to one. This will effectively use the sample values at these
      points as background values.

  Returns:
    An instance of VolumeRenderingResult with the following fields:

    ray_values: The result of accumulating 'sample_values' along the sample
      axis. Leaf values will have the shape [..., N] and are by construction a
      sub-convex combination of the [..., S, N] quantities from 'sample_values'.
    ray_alpha: The sum of per-sample weights along each ray. If this value is
      less than one that indicates a non-zero transmittance from the background
      to the camera, which is equivalent to indicating transparency of the
      foreground scene.
    ray_depth: A pseudo-depth value for the "surface" modelled by the density
      field, computed by accumulating the sample depth values along the ray.
    sample_weights: The per-sample weight values used to compute accumulations.
    sample_intervals: The length of the ray segment associated with each sample.
  """
  # Compute the space between consecutive samples
  intervals = depths[..., 1:] - depths[..., :-1]

  # Assign each sample an interval equal to half the intervals before and after
  before_intervals = jnp.concatenate([intervals[..., :1], intervals], axis=-1)
  after_intervals = jnp.concatenate([intervals, intervals[..., :1]], axis=-1)
  sample_intervals = (before_intervals + after_intervals) / 2

  # Compute the "mass" in each interval as the product of density and length
  sample_mass = sample_density * sample_intervals

  # Compute the cumulative transmittance up to each sample
  transmittance = jnp.exp(-jnp.cumsum(sample_mass[..., :-1], axis=-1))
  # Set transmittance for the first sample to 100%
  transmittance = jnp.concatenate(
      [jnp.ones_like(transmittance[..., :1]), transmittance], axis=-1)

  # Compute the per-sample opacity
  sample_alpha = 1.0 - jnp.exp(-sample_mass)
  if opaque_final_sample:
    sample_alpha = sample_alpha.at[..., -1].set(1.0)

  # Compute the fractional contribution of each sample to the value for the ray
  sample_weights = sample_alpha * transmittance

  # Compute the cumulative opacity and depth of each ray
  ray_alpha = jnp.sum(sample_weights, axis=-1)
  ray_depth = jnp.sum(sample_weights * depths, axis=-1)

  # Accumulate sample quantities along the rays
  def accumulate_values(sample_value):
    return jnp.sum(sample_value * sample_weights[..., None], axis=-2)

  ray_values = jax.tree_map(accumulate_values, sample_values)

  # Combine background values
  def composite_background(ray_value, background_value):
    return ray_value + (1.0 - ray_alpha[..., None]) * background_value

  if background_values is not None:
    ray_values = jax.tree_map(composite_background, ray_values,
                                   background_values)

  return VolumeRenderingResult(
      ray_values=ray_values,
      ray_alpha=ray_alpha,
      ray_depth=ray_depth,
      sample_weights=sample_weights,
      sample_intervals=sample_intervals)


def sample_piecewise_constant_pdf(
    *,
    bin_edges: FloatArray[..., "B+1"],
    weights: FloatArray[..., "B"],
    sample_count: int,
    deterministic: bool,
    rng: Optional[jax.random.KeyArray] = None,
    epsilon: float = 1e-5,
) -> FloatArray[..., "sample_count"]:
  """Draws samples from an empircal 1D PDF using inverse transform sampling.

  Given a 1-dimensional domain divided into a contiguous sequence of bins
  defined by 'bin_edges', this function draws samples from an empirical
  distribution over that domain. The probability density of this distribution is
  defined as constant within each bin such that the total probability mass
  within a bin is equal to the corresponding weight element from the array
  'weights'.

  If the weight values do not sum to one, they will be normalized. For the case
  of all zeros in 'weights', the resulting distribution will have an equal
  probability of sampling within each bin.

  Args:
    bin_edges: A sequence of B+1 points in 1-D space that define the edges of B
      contiguous bins. The bins may have arbitrary leading batch dimensions,
      in which case the sampling will be repeated independently for each batch
      element.
    weights: The fractional probability that a value should be sampled from each
      bin.
    sample_count: The number of samples to be drawn for each batch element.
    deterministic: Whether to use a grid instead of random uniform sampling.
    rng: RNG for the random sampling. Must be set when deterministic=False.
    epsilon: A numerical epsilon for stability.

  Returns:
    A batched array containing the samples.
  """
  dtype = bin_edges.dtype
  batch_shape = bin_edges.shape[:-1]

  # Add a small value to each weight such that the sum is at least epsilon
  weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
  sum_difference = jnp.maximum(0, epsilon - weight_sum)
  weights += sum_difference / weights.shape[-1]
  weight_sum += sum_difference

  # Compute the normalized CDF along the last axis
  normalized_pdf = weights / weight_sum
  cdf = jnp.minimum(1, jnp.cumsum(normalized_pdf[..., :-1], axis=-1))
  cdf = jnp.concatenate(
      [jnp.zeros_like(cdf[..., :1]), cdf,
       jnp.ones_like(cdf[..., :1])], axis=-1)

  # Draw samples from [0, 1)
  if deterministic:
    u = sample_1d_grid(
        sample_count=sample_count, batch_shape=batch_shape, dtype=dtype)
  else:
    u = sample_1d(
        sample_count=sample_count,
        batch_shape=batch_shape,
        strategy=SamplingStrategy.UNIFORM,
        dtype=dtype,
        rng=rng)

  # Find the which bins in the CDF contain the samples
  mask = u[..., None, :] >= cdf[..., :, None]

  def find_bins(x):
    # For values 'x' at the bin edges, find values before and after each sample
    x0 = jnp.max(jnp.where(mask, x[..., None], x[..., :1, None]), -2)
    x1 = jnp.min(jnp.where(~mask, x[..., None], x[..., -1:, None]), -2)
    return x0, x1

  lower_bin_edges, upper_bin_edges = find_bins(bin_edges)
  lower_cdf_value, upper_cdf_value = find_bins(cdf)

  # Compute the sample locations from the bin edges
  denominator = (upper_cdf_value - lower_cdf_value)
  denominator = jnp.where(denominator < epsilon, 1.0, denominator)
  t = (u - lower_cdf_value) / denominator
  samples = lower_bin_edges + t * (upper_bin_edges - lower_bin_edges)

  return samples
