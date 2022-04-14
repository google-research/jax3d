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

"""Helper functions/classes for model definition."""

from typing import Optional, Tuple

import chex
from jax import lax
from jax import numpy as jnp
from jax import random
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import PRNGKey, f32  # pylint: disable=g-multiple-import


def cast_rays(z_vals: f32['... 1'],
              origins: f32['... d'],
              directions: f32['... d']) -> f32['... d']:
  return origins[..., None, :] + z_vals[..., None] * directions[..., None, :]


def sample_along_rays(*,
                      key: PRNGKey,
                      origins: f32['...  3'],
                      directions: f32['...  3'],
                      num_samples: int,
                      near: f32['...  1'],
                      far: f32['...  1'],
                      randomized: bool,
                      lindisp: bool) -> Tuple[
                          f32['... num_samples'], f32['... num_samples 3']]:
  """Stratified sampling along the rays.

  Args:
    key: random generator key.
    origins: ray origins.
    directions: ray directions.
    num_samples: number of samples per ray.
    near: near clip.
    far: far clip.
    randomized: use randomized stratified sampling.
    lindisp: sampling linearly in disparity rather than depth.

  Returns:
    z_vals: sampled z values.
    points: sampled points.
  """
  assert not randomized or key is not None
  chex.assert_equal_shape([origins, directions])
  chex.assert_equal_shape([near, far])
  chex.assert_equal_shape_prefix([origins, directions, near, far],
                                 len(origins.shape) - 1)
  batch_shape = list(origins.shape[:-1])

  t_vals = jnp.linspace(0., 1., num_samples)
  if lindisp:
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
  else:
    z_vals = near * (1. - t_vals) + far * t_vals

  if randomized:
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = jnp.concatenate([mids, z_vals[..., -1:]], -1)
    lower = jnp.concatenate([z_vals[..., :1], mids], -1)
    t_rand = random.uniform(key, batch_shape + [num_samples])
    z_vals = lower + (upper - lower) * t_rand
  else:
    # Broadcast z_vals to make the returned shape consistent.
    z_vals = jnp.broadcast_to(z_vals, batch_shape + [num_samples])

  coords = cast_rays(z_vals, origins, directions)
  return z_vals, coords


def posenc(x: f32['... d'], num_octaves: int) -> f32['... d*2*num_octaves']:
  """Compute positional encodings of x with scales 2^num_octaves.

  Instead of computing [sin(x), cos(x)], we use the trig identity
  cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

  Note this code assumes the posencs are ordered as follows:
  x_octave0, y_octave0, z_octave0, x_octave1....

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    num_octaves: int, the maximum (exclusive) degree of the encoding.

  Returns:
    encoded: f32['... d * 2 * num_octaves'], encoded variables.
  """
  if num_octaves == 0:
    return x
  scales = jnp.array([2**i for i in range(num_octaves)])
  xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                   (*x.shape[:-1], x.shape[-1] * num_octaves))
  four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  return four_feat


def volumetric_rendering(
    *,
    rgb: Optional[f32['... num_samples 3']],
    sigma: f32['... num_samples 1'],
    z_vals: f32['... num_samples'],
    dirs: Optional[f32['... 3']],
    semantic: Optional[f32['... num_samples num_classes']],
) -> types.RenderedRays:
  """Volumetric Rendering Function.

  Args:
    rgb: color
    sigma: density
    z_vals: z values along the ray
    dirs: view directions
    semantic: Optional semantic logits.

  Returns:
    comp_rgb: composite rgb values
    disp: disparity
    acc: accumulated opacity
    weights: contribution along the ray
  """
  eps = 1e-10
  dists = jnp.concatenate([
      z_vals[..., 1:] - z_vals[..., :-1],
      jnp.broadcast_to(1e10, z_vals[..., :1].shape)
  ], -1)

  # Note: If norm(dirs) == 0, dists == 0 too.
  if dirs is not None:
    dists = dists * jnp.linalg.norm(dirs[..., None, :], axis=-1)

  # Note that we're quietly turning sigma from [..., 0] to [...].
  alpha = 1.0 - jnp.exp(-sigma[..., 0] * dists)
  accum_prod = jnp.concatenate([
      jnp.ones_like(alpha[..., :1], alpha.dtype),
      jnp.cumprod(1.0 - alpha[..., :-1] + eps, axis=-1)
  ],
                               axis=-1)
  weights = alpha * accum_prod

  comp_rgb = None
  if rgb is not None:
    comp_rgb = (weights[..., None] * rgb).sum(axis=-2)

  # We need to add this stop_gradient in order to prevent gradients from
  # semantic loss to sigma (and thus indirectly to rgb).
  comp_semantic = None
  if semantic is not None:
    semantic_weights = lax.stop_gradient(weights)
    comp_semantic = (semantic_weights[..., None] * semantic).sum(axis=-2)
  depth = (weights * z_vals).sum(axis=-1)
  acc = weights.sum(axis=-1)
  # Equivalent to (but slightly more efficient and stable than):
  #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
  inv_eps = 1 / eps
  disp = jnp.where(depth > 0, acc / depth, 0.)  # in case depth=0.
  disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
  return types.RenderedRays(
      rgb=comp_rgb,
      foreground_rgb=comp_rgb,
      disparity=disp,
      opacity=acc,
      contribution=weights,
      semantic=comp_semantic,
      foreground_semantic=comp_semantic,
  )


def piecewise_constant_pdf(key: PRNGKey,
                           bins: f32['... bins+1'],
                           weights: f32['... bins'],
                           num_samples: int,
                           randomized: bool) -> f32['... num_samples']:
  """Piecewise-Constant PDF sampling.

  Args:
    key: random number generator.
    bins: positions of the bins
    weights: weights per bin
    num_samples: the number of samples.
    randomized: use randomized samples.

  Returns:
    z_samples: the sampled z values.
  """
  assert not randomized or key is not None
  # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
  # avoids NaNs when the input is zeros or small, but has no effect otherwise.
  eps = 1e-5
  weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
  padding = jnp.maximum(0, eps - weight_sum)
  weights += padding / weights.shape[-1]
  weight_sum += padding

  # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
  # starts with exactly 0 and ends with exactly 1.
  pdf = weights / weight_sum
  cdf = jnp.minimum(1, jnp.cumsum(pdf[..., :-1], axis=-1))
  cdf = jnp.concatenate([
      jnp.zeros(list(cdf.shape[:-1]) + [1]), cdf,
      jnp.ones(list(cdf.shape[:-1]) + [1])
  ],
                        axis=-1)

  # Draw uniform samples.
  if randomized:
    # Note that `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u = random.uniform(key, list(cdf.shape[:-1]) + [num_samples])
  else:
    # Match the behavior of random.uniform() by spanning [0, 1-eps].
    u = jnp.linspace(0., 1. - jnp.finfo('float32').eps, num_samples)
    u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

  # Identify the location in `cdf` that corresponds to a random sample.
  # The final `True` index in `mask` will be the start of the sampled interval.
  mask = u[..., None, :] >= cdf[..., :, None]

  def find_interval(x):
    # Grab the value where `mask` switches from True to False, and vice versa.
    # This approach takes advantage of the fact that `x` is sorted.
    x0 = jnp.max(jnp.where(mask, x[..., None], x[..., :1, None]), -2)
    x1 = jnp.min(jnp.where(~mask, x[..., None], x[..., -1:, None]), -2)
    return x0, x1

  bins_g0, bins_g1 = find_interval(bins)
  cdf_g0, cdf_g1 = find_interval(cdf)

  t = jnp.clip(jnp.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
  samples = bins_g0 + t * (bins_g1 - bins_g0)

  # Prevent gradient from backprop-ing through `samples`.
  return lax.stop_gradient(samples)


def sample_pdf(*,
               key,
               bins: f32['... num_bins+1'],
               weights: f32['... num_bins'],
               origins: f32['... 3'],
               directions: f32['... 3'],
               z_vals: f32['... num_coarse_samples'],
               num_samples: int,
               randomized: bool,
               include_original_z_vals: bool = True,) -> Tuple[
                   f32['... num_coarse_samples+num_samples'],
                   f32['... num_coarse_samples+num_samples, 3']]:
  """Hierarchical sampling.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: positions of the bins
    weights: weights per bin
    origins: ray origins.
    directions: ray directions.
    z_vals: coarse samples.
    num_samples: the number of samples.
    randomized: use randomized samples.
    include_original_z_vals: Whether to include the original z_vals in the
      output. If True, the values will be sorted in.

  Returns:
    z_vals: sampled z values
    points: sampled points
  """
  z_samples = piecewise_constant_pdf(key, bins, weights, num_samples,
                                     randomized)

  if include_original_z_vals:
    # Compute united z_vals and sample points
    z_samples = jnp.sort(jnp.concatenate([z_vals, z_samples], axis=-1), axis=-1)
  coords = cast_rays(z_samples, origins, directions)
  return z_samples, coords


def add_gaussian_noise(*,
                       key,
                       raw: f32[...],
                       noise_std: float,
                       randomized: bool) -> f32[...]:
  """Adds gaussian noise to `raw`, which can used to regularize it.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    raw: raw values
    noise_std: The standard deviation of the noise to be added.
    randomized: add noise if randomized is True.

  Returns:
    raw + noise
  """
  assert not randomized or key is not None
  if noise_std and randomized:
    return raw + random.normal(key, raw.shape, dtype=raw.dtype) * noise_std
  else:
    return raw


def valid_rays(rays: types.Rays) -> f32['... 1']:
  """Return a mask which encodes which is 1 if the ray is valid, 0 otherwise."""
  return jnp.sum(jnp.abs(rays.direction), axis=-1, keepdims=True) > 0


def alpha_composite(foreground: f32['... d'], background: f32,
                    alpha: f32['...']) -> f32['... d']:
  # Note that the volume rendered foreground has a max value of alpha.
  return foreground + background * (1.0 - alpha[..., None])


def calculate_near_and_far(rays: types.Rays):
  """Calculates near and far planes."""
  epsilon = 1e-10
  nerf_box = types.BoundingBox3d(min_corner=-jnp.ones(3) + epsilon,
                                 max_corner=jnp.ones(3) - epsilon)

  # Add placeholder rays for invalid rays ||direction|| == 0.
  empty_rays = jnp.linalg.norm(rays.direction, ord=2, axis=-1) == 0
  placeholder_direction = jnp.ones_like(empty_rays, jnp.float32) / 3
  direction = jnp.where(
      empty_rays[..., None], placeholder_direction[..., None], rays.direction)
  rays = rays.replace(direction=direction)

  near, far = nerf_box.intersect_rays(rays)

  near = jnp.maximum(epsilon, near)
  near = jnp.where(near > far, epsilon, near)
  far = jnp.where(near > far, near + epsilon, far)
  return near[..., None], far[..., None]


def _expected_sin(x, x_var):
  """Estimates mean and variance of sin(z), z ~ N(x, var)."""
  # When the variance is wide, shrink sin towards zero.
  y = jnp.exp(-0.5 * x_var) * jnp.sin(x % (2 * jnp.pi))
  return y


def integrated_posenc(points: types.SamplePoints, num_octaves: int):
  """Mip-NeRF posenc: encode points with sinusoids scaled by 2^[:num_octaves].

  Args:
    points: Sample points containing points and covariances. Points should
      be in [-pi, pi].
    num_octaves: int, the maximum (exclusive) degree of the encoding.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if points.covariance is None:
    raise ValueError('Covariance needed for Mip-Nerf integrated posenc.')

  chex.assert_equal_shape([points.position, points.covariance])
  scales = 2**jnp.arange(0, num_octaves)
  shape = points.position.shape[:-1] + (-1,)
  y = jnp.reshape(points.position[..., None, :] * scales[:, None], shape)
  y_var = jnp.reshape(points.covariance[..., None, :] * scales[:, None]**2,
                      shape)

  return _expected_sin(
      jnp.concatenate([y, y + 0.5 * jnp.pi], axis=-1),
      jnp.concatenate([y_var] * 2, axis=-1))
