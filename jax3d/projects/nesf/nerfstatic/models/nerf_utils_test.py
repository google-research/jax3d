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

# Lint as: python3
"""Unit tests for nerf_utils."""
import jax
from jax import random
import jax.numpy as jnp
from jax3d.projects.nesf.nerfstatic.models import nerf_utils
from nerf import _run_nerf_helpers as run_nerf_helpers
from nerf.google.arkadia import run_nerf_helpers as extra_test_helpers
import numpy as np
import pytest
import scipy as sp
import tensorflow as tf


@pytest.mark.parametrize('lindisp', [True, False])
@pytest.mark.parametrize('randomized', [True, False])
def test_sample_along_rays(lindisp: bool, randomized: bool):
  """Test uniform sampling along rays."""
  near_val = np.asarray([[2.]] * 10)
  far_val = np.asarray([[6.]] * 10)
  n_samples = 64
  rays_o = np.random.uniform(-1., 1., (10, 3)).astype(np.float32)
  rays_d = np.random.uniform(-1., 1., (10, 3)).astype(np.float32)

  near = near_val * tf.ones(rays_d.shape[:-1])[..., None]
  far = far_val * tf.ones(rays_d.shape[:-1])[..., None]
  rays_tf = tf.concat([rays_o, rays_d, near, far], axis=-1)
  gt_z_vals, gt_pts = extra_test_helpers.default_ray_sampling(
      rays_tf[..., 0:3], rays_tf[..., 3:6], rays_tf, rays_tf.shape[0],
      n_samples, False, lindisp)

  z_vals, pts = nerf_utils.sample_along_rays(
      key=random.PRNGKey(0),
      origins=rays_o,
      directions=rays_d,
      num_samples=n_samples,
      near=near_val,
      far=far_val,
      randomized=randomized,
      lindisp=lindisp,
  )

  if not randomized:
    np.testing.assert_allclose(gt_z_vals, z_vals, atol=1e-6)
    np.testing.assert_allclose(gt_pts, pts, atol=1e-6)


def test_posenc():
  """Test the positional encoding function."""
  num_dims = 2
  x = np.linspace(0, np.pi, endpoint=True, num=5 * num_dims)
  x = x.reshape((1, 5, num_dims))
  y = nerf_utils.posenc(x, 3)

  # Manually construct expected outputs of posenc for a example=0, sample=0.
  # feature vector. The order of the outputs should be,
  #   [sin(x), sin(y), sin(2x), sin(2y), ...,
  #    cos(x), cos(y), cos(2x), cos(2y), ...]
  expected = np.array([
      # sin()
      np.sin(x[0, 0, 0]), np.sin(x[0, 0, 1]),
      np.sin(2 * x[0, 0, 0]), np.sin(2 * x[0, 0, 1]),
      np.sin(4 * x[0, 0, 0]), np.sin(4 * x[0, 0, 1]),
      # cos()
      np.cos(x[0, 0, 0]), np.cos(x[0, 0, 1]),
      np.cos(2 * x[0, 0, 0]), np.cos(2 * x[0, 0, 1]),
      np.cos(4 * x[0, 0, 0]), np.cos(4 * x[0, 0, 1])])
  np.testing.assert_allclose(y[0, 0], expected, atol=1e-6)

  # Manually construct expected outputs of posenc for a example=0, sample=1.
  expected = np.array([np.sin(x[0, 1, 0]), np.sin(x[0, 1, 1]),
                       np.sin(2*x[0, 1, 0]), np.sin(2*x[0, 1, 1]),
                       np.sin(4*x[0, 1, 0]), np.sin(4*x[0, 1, 1])])
  np.testing.assert_allclose(y[0, 1, 0:6], expected, atol=1e-6)

  # Manually construct expected outputs of posenc for a example=0, sample=2.
  expected = np.array([np.sin(x[0, 2, 0]), np.sin(x[0, 2, 1]),
                       np.sin(2*x[0, 2, 0]), np.sin(2*x[0, 2, 1]),
                       np.sin(4*x[0, 2, 0]), np.sin(4*x[0, 2, 1])])
  np.testing.assert_allclose(y[0, 2, 0:6], expected, atol=1e-6)


def test_volumetric_rendering():
  """Test the volumetric rendering function."""
  white_bkgd = True
  raw = np.random.rand(10, 5, 4)
  z_vals = np.random.uniform(2., 6., (10, 5))
  rays_d = np.random.rand(10, 3)

  raw_tf = tf.cast(raw, tf.float32)
  z_vals_tf = tf.cast(z_vals, tf.float32)
  rays_d_tf = tf.cast(rays_d, tf.float32)
  (gt_rgb, gt_disp, gt_acc, gt_weights,
   unused_depth) = extra_test_helpers.raw2outputs(
       raw_tf, z_vals_tf, rays_d_tf, white_bkgd=white_bkgd)

  raw_rgb_jax = jax.nn.sigmoid(jnp.array(raw[..., :3], dtype=jnp.float32))
  raw_sigma_jax = jax.nn.relu(jnp.array(raw[..., 3:], dtype=jnp.float32))
  raw_semantic_jax = jnp.zeros((*raw_rgb_jax.shape[:-1], 0), dtype=jnp.float32)
  z_vals_jax = jnp.array(z_vals, dtype=jnp.float32)
  rays_d_jax = jnp.array(rays_d, dtype=jnp.float32)
  ray = nerf_utils.volumetric_rendering(
      rgb=raw_rgb_jax,
      sigma=raw_sigma_jax,
      semantic=raw_semantic_jax,
      z_vals=z_vals_jax,
      dirs=rays_d_jax)
  # Adjust color for white background
  ray.rgb = ray.rgb + (1. - ray.opacity[..., None])

  # Normalize disparity maps.
  gt_disp_np = gt_disp.numpy()
  gt_disp_np = ((gt_disp_np - gt_disp_np.min()) /
                (gt_disp_np.max() - gt_disp_np.min()))
  disp = ray.disparity
  disp = ((disp - disp.min()) / (disp.max() - disp.min()))

  # Note that according to
  # https://github.com/tensorflow/tensorflow/issues/5527
  # TF has an numerically issue in reduce_sum whose error can be
  # as much as 2e-5 sometimes compared to numpy. Hence it's possible that
  # this test may fail with numerical differences larger than the bound,
  # especially the gt_disp v.s disp comparison whose numerator and
  # denominator both have this issue.
  np.testing.assert_allclose(gt_rgb, ray.rgb, atol=5e-5, rtol=5e-5)
  np.testing.assert_allclose(gt_disp_np, disp, atol=5e-5, rtol=5e-5)
  np.testing.assert_allclose(gt_acc, ray.opacity, atol=5e-5, rtol=5e-5)
  np.testing.assert_allclose(gt_weights, ray.contribution, atol=5e-5, rtol=5e-5)
  assert ray.semantic.shape == (*gt_rgb.shape[:-1], 0)


def test_volumetric_rendering_semantics():
  """Test the volumetric rendering function."""
  white_bkgd = True
  raw = np.random.rand(10, 5, 4)
  z_vals = np.random.uniform(2., 6., (10, 5))
  rays_d = np.random.rand(10, 3)

  raw_tf = tf.cast(raw, tf.float32)
  z_vals_tf = tf.cast(z_vals, tf.float32)
  rays_d_tf = tf.cast(rays_d, tf.float32)
  gt_rgb = extra_test_helpers.raw2outputs(
      raw_tf, z_vals_tf, rays_d_tf, white_bkgd=white_bkgd)[0]

  raw_sigma_jax = jax.nn.relu(jnp.array(raw[..., 3:], dtype=jnp.float32))
  raw_semantic_jax = jax.nn.sigmoid(jnp.array(raw, dtype=jnp.float32))
  z_vals_jax = jnp.array(z_vals, dtype=jnp.float32)
  ray = nerf_utils.volumetric_rendering(
      rgb=None,
      sigma=raw_sigma_jax,
      semantic=raw_semantic_jax,
      dirs=None,
      z_vals=z_vals_jax)

  assert ray.rgb is None
  assert ray.semantic.shape == (*gt_rgb.shape[:-1], 4)


def test_volumetric_rendering_invalid_direction():
  """Test volumetric_rendering when ray_d == 0."""
  raw = np.random.rand(1, 5, 4)
  z_vals = np.random.uniform(2., 6., (1, 5))
  rays_d = np.zeros((1, 3))  # invalid rays have direction == 0.

  raw_rgb_jax = jax.nn.sigmoid(jnp.array(raw[..., :3], dtype=jnp.float32))
  raw_sigma_jax = jax.nn.relu(jnp.array(raw[..., 3:], dtype=jnp.float32))
  raw_semantic_jax = jnp.zeros((*raw_rgb_jax.shape[:-1], 0), dtype=jnp.float32)
  z_vals_jax = jnp.array(z_vals, dtype=jnp.float32)
  rays_d_jax = jnp.array(rays_d, dtype=jnp.float32)
  ray = nerf_utils.volumetric_rendering(
      rgb=raw_rgb_jax,
      sigma=raw_sigma_jax,
      semantic=raw_semantic_jax,
      z_vals=z_vals_jax,
      dirs=rays_d_jax)

  # If rays_d == 0, you'll get acc == depth == 0 and thus
  # disparity == 0/0 == NaN. Ensure this isn't the case here.
  assert not np.any(np.isnan(ray.disparity))
  np.testing.assert_allclose(ray.contribution, np.zeros_like(ray.contribution))


def test_sample_pdf_eval_mode():
  """Test the sample_pdf function in eval mode, i.e. no randomization."""
  n_samples = 5
  bins = np.array([[0., 1., 2., 4.], [2., 4., 6., 8.]])
  weights = np.array([[1.0, 1.0, 1.0], [0.5, 1.0, 0.5]])
  z_vals = np.random.uniform(0., 2., (2, 5))
  rays = np.random.rand(2, 6)

  bins_tf = tf.cast(bins, tf.float32)
  weights_tf = tf.cast(weights, tf.float32)
  z_samples_tf = run_nerf_helpers.sample_pdf(
      bins_tf, weights_tf, n_samples, det=True)
  z_vals_tf = tf.sort(tf.concat([z_vals, z_samples_tf], -1), -1)
  rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
  pts_tf = (
      rays_o[..., None, :] + rays_d[..., None, :] * z_vals_tf[..., :, None])

  bins_jax = jnp.array(bins, jnp.float32)
  weights_jax = jnp.array(weights, jnp.float32)
  z_vals_jax = jnp.array(z_vals, jnp.float32)
  z_vals_jax, pts_jax = nerf_utils.sample_pdf(
      key=jnp.array([0, 1]),
      bins=bins_jax,
      weights=weights_jax,
      origins=rays_o,
      directions=rays_d,
      z_vals=z_vals_jax,
      num_samples=n_samples,
      randomized=False)

  np.testing.assert_allclose(z_vals_tf, z_vals_jax, atol=1e-5)
  np.testing.assert_allclose(pts_tf, pts_jax, atol=1e-5)


@pytest.mark.parametrize('randomized', [True, False])
def test_piecewise_constant_pdf_train_mode(randomized: bool):
  """Test that piecewise-constant sampling reproduces its distribution."""
  batch_size = 4
  num_bins = 16
  num_samples = 1000000
  precision = 1e5
  rng = random.PRNGKey(20202020)

  # Generate a series of random PDFs to sample from.
  data = []
  for _ in range(batch_size):
    rng, key = random.split(rng)
    # Randomly initialize the distances between bins.
    # We're rolling our own fixed precision here to make cumsum exact.
    bins_delta = jnp.round(precision * jnp.exp(
        random.uniform(key, shape=(num_bins + 1,), minval=-3, maxval=3)))

    # Set some of the bin distances to 0.
    rng, key = random.split(rng)
    bins_delta *= random.uniform(key, shape=bins_delta.shape) < 0.9

    # Integrate the bins.
    bins = jnp.cumsum(bins_delta) / precision
    rng, key = random.split(rng)
    bins += random.normal(key) * num_bins / 2
    rng, key = random.split(rng)

    # Randomly generate weights, allowing some to be zero.
    weights = jnp.maximum(
        0, random.uniform(key, shape=(num_bins,), minval=-0.5, maxval=1.))
    gt_hist = weights / weights.sum()
    data.append((bins, weights, gt_hist))

  # Tack on an "all zeros" weight matrix, which is a common cause of NaNs.
  weights = jnp.zeros_like(weights)
  gt_hist = jnp.ones_like(gt_hist) / num_bins
  data.append((bins, weights, gt_hist))

  bins, weights, gt_hist = [jnp.stack(x) for x in zip(*data)]

  rng, key = random.split(rng)
  # Draw samples from the batch of PDFs.
  samples = nerf_utils.piecewise_constant_pdf(key, bins, weights,
                                              num_samples, randomized)
  assert samples.shape[-1] == num_samples

  # Verify that each set of samples resembles the target distribution.
  for i_samples, i_bins, i_gt_hist in zip(samples, bins, gt_hist):
    i_hist = jnp.float32(jnp.histogram(i_samples, i_bins)[0]) / num_samples
    i_gt_hist = jnp.array(i_gt_hist)

    # Merge any of the zero-span bins until there aren't any left.
    while jnp.any(i_bins[:-1] == i_bins[1:]):
      j = int(jnp.where(i_bins[:-1] == i_bins[1:])[0][0])
      i_hist = jnp.concatenate([
          i_hist[:j],
          jnp.array([i_hist[j] + i_hist[j + 1]]), i_hist[j + 2:]
      ])
      i_gt_hist = jnp.concatenate([
          i_gt_hist[:j],
          jnp.array([i_gt_hist[j] + i_gt_hist[j + 1]]), i_gt_hist[j + 2:]
      ])
      i_bins = jnp.concatenate([i_bins[:j], i_bins[j + 1:]])

    # Angle between the two histograms in degrees.
    angle = 180 / jnp.pi * jnp.arccos(
        jnp.minimum(
            1.,
            jnp.mean(
                (i_hist * i_gt_hist) /
                jnp.sqrt(jnp.mean(i_hist**2) * jnp.mean(i_gt_hist**2)))))
    # Jensen-Shannon divergence.
    m = (i_hist + i_gt_hist) / 2
    js_div = jnp.sum(
        sp.special.kl_div(i_hist, m) + sp.special.kl_div(i_gt_hist, m)) / 2
    assert angle <= 0.5
    assert js_div <= 1e-5


def test_piecewise_constant_pdf_large_flat():
  """Test sampling when given a large flat distribution."""
  num_samples = 100
  num_bins = 100000
  key = random.PRNGKey(0)
  bins = jnp.arange(num_bins)
  weights = np.ones(len(bins) - 1)
  samples = nerf_utils.piecewise_constant_pdf(key, bins[None], weights[None],
                                              num_samples, True)[0]
  # All samples should be within the range of the bins.
  assert jnp.all(samples >= bins[0])
  assert jnp.all(samples <= bins[-1])

  # Samples modded by their bin index should resemble a uniform distribution.
  samples_mod = jnp.mod(samples, 1)
  assert sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic <= 0.2  # pytype: disable=attribute-error

  # All samples should collectively resemble a uniform distribution.
  assert (
      sp.stats.kstest(samples, 'uniform', (bins[0], bins[-1])).statistic <= 0.2)  # pytype: disable=attribute-error


def test_piecewise_constant_pdf_sparse_delta():
  """Test sampling when given a large distribution with a big delta in it."""
  num_samples = 100
  num_bins = 100000
  key = random.PRNGKey(0)
  bins = jnp.arange(num_bins)
  weights = np.ones(len(bins) - 1)
  delta_idx = len(weights) // 2
  weights[delta_idx] = len(weights) - 1
  samples = nerf_utils.piecewise_constant_pdf(key, bins[None], weights[None],
                                              num_samples, True)[0]

  # All samples should be within the range of the bins.
  assert jnp.all(samples >= bins[0])
  assert jnp.all(samples <= bins[-1])

  # Samples modded by their bin index should resemble a uniform distribution.
  samples_mod = jnp.mod(samples, 1)
  assert sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic <= 0.2  # pytype: disable=attribute-error

  # The delta function bin should contain ~half of the samples.
  in_delta = (samples >= bins[delta_idx]) & (samples <= bins[delta_idx + 1])
  np.testing.assert_allclose(jnp.mean(in_delta), 0.5, atol=0.05)


@pytest.mark.parametrize('randomized', [True, False])
def test_piecewise_constant_pdf_single_bin(randomized):
  """Test sampling when given a small `one hot' distribution."""
  num_samples = 625
  key = random.PRNGKey(0)
  bins = jnp.array([0, 1, 3, 6, 10], jnp.float32)
  for i in range(len(bins) - 1):
    weights = np.zeros(len(bins) - 1, jnp.float32)
    weights[i] = 1.
    samples = nerf_utils.piecewise_constant_pdf(key, bins[None],
                                                weights[None], num_samples,
                                                randomized)[0]

    # All samples should be within [bins[i], bins[i+1]].
    assert jnp.all(samples >= bins[i])
    assert jnp.all(samples <= bins[i + 1])
