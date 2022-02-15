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

"""Tests for volume_rendering."""

import jax
import jax.numpy as jnp
from jax3d.math import volume_rendering
import pytest


# f64 tests require JAX_ENABLE_X64 which is not set by default
DATA_TYPES = [
    jnp.bfloat16,
    jnp.float32,
]


@pytest.mark.parametrize("strategy", [
    volume_rendering.SamplingStrategy.STRATIFIED,
    volume_rendering.SamplingStrategy.UNIFORM,
])
def test_sample_1d_shape(strategy: volume_rendering.SamplingStrategy) -> None:
  batch_shape = (2, 3, 5, 7, 11, 13)
  sample_count = 17
  rng = jax.random.PRNGKey(0)
  samples = volume_rendering.sample_1d(
      sample_count=sample_count,
      batch_shape=batch_shape,
      strategy=strategy,
      rng=rng)

  # Shape should be batch_shape + (sample_count,)
  assert samples.shape == batch_shape + (sample_count,)


def test_sample_1d_grid_shape() -> None:
  batch_shape = (2, 3, 5, 7, 11, 13)
  sample_count = 17
  samples = volume_rendering.sample_1d_grid(
      sample_count=sample_count,
      batch_shape=batch_shape)

  # Shape should be batch_shape + (sample_count,)
  assert samples.shape == batch_shape + (sample_count,)


@pytest.mark.parametrize("strategy", [
    volume_rendering.SamplingStrategy.STRATIFIED,
    volume_rendering.SamplingStrategy.UNIFORM
])
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_sample_1d_random(strategy: volume_rendering.SamplingStrategy,
                          dtype: jnp.dtype) -> None:
  rng = jax.random.PRNGKey(0)
  batch_shape = (2, 3, 7, 11, 13, 17)
  sample_count = 5
  samples = volume_rendering.sample_1d(
      sample_count=sample_count,
      batch_shape=batch_shape,
      strategy=strategy,
      dtype=dtype,
      rng=rng)

  # All generated samples should be in the range [0, 1)
  assert jnp.min(samples) + jnp.finfo(dtype).eps >= 0.0
  assert jnp.max(samples) - jnp.finfo(dtype).eps < 1.0

  # The samples should conform to a uniform distribution in [0, 1)
  bins = 8
  histogram, _ = jnp.histogram(samples, bins=jnp.linspace(0.0, 1.0, bins + 1))
  expected_count = float(jnp.prod(jnp.array(samples.shape))) / bins
  fraction_of_expected = histogram.astype(jnp.float64) / expected_count
  assert jnp.allclose(
      fraction_of_expected, 1.0, atol=0.01 + 10 * jnp.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_sample_1d_grid(dtype: jnp.dtype) -> None:
  batch_shape = (13, 17)
  sample_count = 23
  samples = volume_rendering.sample_1d_grid(
      sample_count=sample_count,
      batch_shape=batch_shape,
      dtype=dtype)

  # All generated samples should be in the range [0, 1)
  assert jnp.min(samples) + jnp.finfo(dtype).eps >= 0.0
  assert jnp.max(samples) - jnp.finfo(dtype).eps < 1.0

  # The mean of the samples should be 0.5
  assert jnp.allclose(
      jnp.mean(samples, axis=-1), 0.5, atol=jnp.finfo(dtype).eps)

  # The inter-sample distance should equal 1 / sample_count
  assert jnp.allclose(
      samples[..., 1:] - samples[..., :-1],
      1.0 / sample_count,
      atol=jnp.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_sample_1d_sorted(dtype: jnp.dtype) -> None:
  rng = jax.random.PRNGKey(0)
  batch_shape = (2, 3, 5, 7)
  sample_count = 11
  samples = volume_rendering.sample_1d(
      sample_count=sample_count,
      batch_shape=batch_shape,
      strategy=volume_rendering.SamplingStrategy.STRATIFIED,
      dtype=dtype,
      rng=rng)

  # Stratified samples should be strictly in ascending order
  assert jnp.all((samples[..., 1:] - samples[..., :-1]) >= 0.0)


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("use_disparity", [True, False])
@pytest.mark.parametrize("broadcast_near_far", [True, False])
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_sample_along_rays(deterministic: bool,
                           use_disparity: bool,
                           broadcast_near_far: bool,
                           dtype: jnp.dtype) -> None:
  rngs = jax.random.split(jax.random.PRNGKey(42), 4)

  batch_shape = (2, 3, 5)
  ray_origins = jax.random.normal(rngs[0], batch_shape + (3,), dtype=dtype)
  ray_directions = jax.random.normal(rngs[1], batch_shape + (3,), dtype=dtype)

  if broadcast_near_far:
    near = jax.random.uniform(rngs[2], (), dtype=dtype)
    far = near + jax.random.uniform(rngs[3], (), dtype=dtype)
  else:
    near = jax.random.uniform(rngs[2], batch_shape, dtype=dtype)
    far = near + jax.random.uniform(rngs[3], batch_shape, dtype=dtype)

  sample_count = 7

  if deterministic:
    rng = None
  else:
    rng = jax.random.PRNGKey(0)

  depths, points = volume_rendering.sample_along_rays(
      ray_origins=ray_origins,
      ray_directions=ray_directions,
      near=near,
      far=far,
      sample_count=sample_count,
      deterministic=deterministic,
      rng=rng,
      use_linear_disparity=use_disparity)

  # Sample batch shapes should match input
  assert depths.shape == batch_shape + (sample_count,)
  assert points.shape == batch_shape + (sample_count, 3)

  # Depths and points should agree
  computed_depths = jnp.linalg.norm(points - ray_origins[..., None, :], axis=-1)
  assert jnp.allclose(depths, computed_depths, atol=10 * jnp.finfo(dtype).eps)

  # Samples should be between near and far
  scaled_depths = depths / jnp.linalg.norm(ray_directions, axis=-1)[..., None]
  assert jnp.all(scaled_depths + jnp.finfo(dtype).eps >= near[..., None])
  assert jnp.all(scaled_depths - jnp.finfo(dtype).eps < far[..., None])


@pytest.mark.parametrize("value_type", ["pytree", "array", "empty"])
@pytest.mark.parametrize("use_background", [True, False])
@pytest.mark.parametrize("opaque_final_sample", [True, False])
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_volume_rendering(value_type: str, use_background: bool,
                          opaque_final_sample: bool, dtype: jnp.dtype) -> None:
  rngs = jax.random.split(jax.random.PRNGKey(42), 6)

  batch_shape = (2, 3, 5)
  sample_count = 7

  # This test generates per-sample density and depth by starting from ray-level
  # values and working backwards to generate inputs to volume_rendering that
  # should reproduce them.
  ray_alpha = jax.random.uniform(rngs[0], batch_shape, dtype=jnp.float64)
  sample_weights = jax.random.uniform(
      rngs[1], batch_shape + (sample_count,), dtype=jnp.float64)
  sample_weights *= ray_alpha[..., None] / jnp.sum(
      sample_weights, axis=-1, keepdims=True)
  epsilon = jnp.finfo(dtype).eps
  sample_alpha = [sample_weights[..., 0]]
  transmittance = [jnp.ones_like(sample_weights[..., 0])]
  for i in range(1, sample_count):
    transmittance_i = transmittance[i - 1] * (1 - sample_alpha[i - 1] + epsilon)
    sample_alpha_i = sample_weights[..., i] / transmittance_i
    transmittance.append(transmittance_i)
    sample_alpha.append(sample_alpha_i)
  transmittance = jnp.stack(transmittance, axis=-1)
  sample_alpha = jnp.stack(sample_alpha, axis=-1)
  sample_depth = jnp.cumsum(
      jax.random.uniform(rngs[2], sample_alpha.shape) + epsilon, axis=-1)
  intervals = sample_depth[..., 1:] - sample_depth[..., :-1]
  before_intervals = jnp.concatenate([intervals[..., :1], intervals], axis=-1)
  after_intervals = jnp.concatenate([intervals, intervals[..., :1]], axis=-1)
  sample_intervals = (before_intervals + after_intervals) / 2
  sample_density = -jnp.log(1.0 - sample_alpha) / sample_intervals

  if opaque_final_sample:
    ray_alpha = ray_alpha.at[...].set(1.0)
    remainder = 1 - jnp.sum(sample_weights[..., :-1], axis=-1)
    sample_weights = sample_weights.at[..., -1].set(remainder)

  ray_depth = jnp.sum(sample_weights * sample_depth, axis=-1)

  def sample_values_from_ray(ray_value, background_value=None):
    if use_background:
      fg_value = ray_value - background_value * (1 - ray_alpha[..., None])
    else:
      fg_value = ray_value
    sample_values = jax.random.normal(
        rngs[5],  # Reused for simplicity
        batch_shape + (sample_count, ray_value.shape[-1]))
    integrated_value = jnp.sum(
        sample_values * sample_weights[..., None], axis=-2)
    error = (fg_value - integrated_value) / ray_alpha[..., None]
    corrected_sample_values = sample_values + error[..., None, :]
    return corrected_sample_values

  if value_type == "empty":
    ray_values = []
    if use_background:
      background_values = []
    else:
      background_values = None
    sample_values = []
  elif value_type == "array":
    ray_values = jax.random.normal(rngs[3], batch_shape + (11,))
    if use_background:
      background_values = jax.random.normal(rngs[4], batch_shape + (11,))
      sample_values = sample_values_from_ray(ray_values, background_values)
    else:
      background_values = None
      sample_values = sample_values_from_ray(ray_values)
  elif value_type == "pytree":
    fg_rngs = jax.random.split(rngs[3], 3)
    ray_values = {
        "leaf":
            jax.random.normal(fg_rngs[0], batch_shape + (11,)),
        "node": (
            jax.random.normal(fg_rngs[1], batch_shape + (13,)),
            jax.random.normal(fg_rngs[2], batch_shape + (17,)),
        )
    }
    if use_background:
      bg_rngs = jax.random.split(rngs[4], 3)
      background_values = {
          "leaf":
              jax.random.normal(bg_rngs[0], batch_shape + (11,)),
          "node": (
              jax.random.normal(bg_rngs[1], batch_shape + (13,)),
              jax.random.normal(bg_rngs[2], batch_shape + (17,)),
          )
      }
      sample_values = jax.tree_map(sample_values_from_ray, ray_values,
                                   background_values)
    else:
      background_values = None
      sample_values = jax.tree_map(sample_values_from_ray, ray_values)

  cast = lambda x: x.astype(dtype)
  sample_values = jax.tree_map(cast, sample_values)
  if use_background:
    background_values = jax.tree_map(cast, background_values)

  render_result = volume_rendering.volume_rendering(
      sample_values=sample_values,
      sample_density=sample_density.astype(dtype),
      depths=sample_depth.astype(dtype),
      background_values=background_values,
      opaque_final_sample=opaque_final_sample)

  # Accumulated values should be consistent with the generated inputs
  allclose = lambda x, y: jnp.allclose(x, y, atol=200 * jnp.finfo(dtype).eps)
  assert jax.tree_util.tree_all(
      jax.tree_map(allclose, render_result.ray_values, ray_values))

  # Ray alpha values should be consistent with the generated inputs
  assert jax.tree_util.tree_all(
      jax.tree_map(allclose, render_result.ray_alpha, ray_alpha))

  # Ray depth values should be consistent with the generated inputs
  assert jax.tree_util.tree_all(
      jax.tree_map(allclose, render_result.ray_depth, ray_depth))

  # Sample weights should be consistent with the generated inputs
  assert jax.tree_util.tree_all(
      jax.tree_map(allclose, render_result.sample_weights, sample_weights))

  # Sample intervals should be consistent with the generated inputs
  assert jax.tree_util.tree_all(
      jax.tree_map(allclose, render_result.sample_intervals, sample_intervals))


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_sample_piecewise_constant_pdf(
    deterministic: bool, dtype: jnp.dtype) -> None:
  rngs = jax.random.split(jax.random.PRNGKey(42), 2)

  batch_shape = (2, 3)
  sample_count = 50021
  bin_count = 6

  bin_edges = jax.random.uniform(
      rngs[0], batch_shape + (bin_count + 1,), dtype=dtype)
  bin_edges = jnp.cumsum(bin_edges, axis=-1)
  weights = jax.random.uniform(rngs[1], batch_shape + (bin_count,), dtype=dtype)

  if deterministic:
    rng = None
  else:
    rng = jax.random.PRNGKey(0)

  samples = volume_rendering.sample_piecewise_constant_pdf(
      bin_edges=bin_edges,
      weights=weights,
      sample_count=sample_count,
      deterministic=deterministic,
      rng=rng)

  # Shape should match batch_shape and sample_count
  assert samples.shape == (batch_shape + (sample_count,))

  # Sample distribution should match weights
  histogram, _ = jax.vmap(jnp.histogram)(samples.reshape(-1, sample_count),
                                         bin_edges.reshape(-1, bin_count + 1))
  histogram = histogram.astype(jnp.float32) / sample_count
  histogram = histogram.reshape(*batch_shape, bin_count)
  input_probabilities = weights / jnp.sum(weights, axis=-1, keepdims=True)
  assert jnp.allclose(
      histogram, input_probabilities, rtol=0.2, atol=100 * jnp.finfo(dtype).eps)
