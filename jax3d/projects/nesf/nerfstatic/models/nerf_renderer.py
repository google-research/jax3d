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

"""A model that renders a scene from spatial latent codes."""

from typing import Optional, Tuple
import chex
from flax import linen as nn
from jax import lax
import jax.numpy as jnp
from jax3d.projects.nesf.nerfstatic.models import nerf_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import ActivationFn, f32  # pylint: disable=g-multiple-import


def _lift_gaussian(d: f32['... num_samples 3'],
                   t_mean: f32,
                   t_var: f32,
                   r_var: f32):
  """Lift a Gaussian defined along ray axes to world-unit coordinates."""
  mean = d[..., None, :] * t_mean[..., None]

  d_mag_sq = jnp.maximum(1e-10, jnp.sum(d**2, axis=-1, keepdims=True))

  d_outer_diag = d**2
  null_outer_diag = 1 - d_outer_diag / d_mag_sq
  t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
  xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
  cov_diag = t_cov_diag + xy_cov_diag
  return mean, cov_diag


def _conical_frustum_to_gaussian(z_vals: f32['... num_samples 1'],
                                 rays: types.Rays):
  """Approximate a conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and rays.base_radius is the
  radius at dist=1. Doesn't assume rays.direction is normalized.

  Note: this function reduces the sample size by one - as it averages z_vals.

  Args:
    z_vals: float array, the distances along the ray.
    rays: rays, containing base_radius: float, the scale of the radius as a
          function of distance.

  Returns:
    a Gaussian (mean and covariance).
  """
  t0 = z_vals[..., :-1]
  t1 = z_vals[..., 1:]
  # Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
  mu = (t0 + t1) / 2  # The average of the two `z` values.
  hw = (t1 - t0) / 2  # The half-width of the two `z` values.
  eps = jnp.finfo(jnp.float32).eps
  t_mean = mu + (2 * mu * hw**2) / jnp.maximum(eps, 3 * mu**2 + hw**2)
  denom = jnp.maximum(eps, 3 * mu**2 + hw**2)
  t_var = (hw**2) / 3 - (4 / 15) * hw**4 * (12 * mu**2 - hw**2) / denom**2
  r_var = (mu**2) / 4 + (5 / 12) * hw**2 - (4 / 15) * (hw**4) / denom
  r_var *= rays.base_radius**2
  means, covs = _lift_gaussian(rays.direction, t_mean, t_var, r_var)
  means = means + rays.origin[..., None, :]
  return means, covs, mu


class NerfRenderer(nn.Module):
  """A Model that renders NeRF.

  The NeRF scene is assumed to be with in the [-1; 1] bounding box.
  Points on rays  will be sampled from along the intersection of the ray with
  this [-1; 1] bounding box.

  Attributes:
    coarse_sample_store: This corresponds to the coarse model in NeRF.
      It will be trained alongside with the fine model and is used to provide
      weights for the importance sampling, if num_fine_samples > 0. Otherwise,
      it is not used.
    fine_sample_store: This corresponds to the fine model in NeRF. This model
      provides data for the final result.
    semantic_sample_store: This corresponds to the fine semantic model in NeRF.
      It will be trained using the predictions of the fine_sample_store.
    num_coarse_samples: Number of coarse samples.
    num_fine_samples: Number of fine samples.
    lindisp: If True, sample linearly in disparity rather than in depth.
    background_params: either NONE or WHITE.
    rgb_activation: Output RGB activation.
    sigma_activation: Output sigma activation.
    enable_mipnerf: Render using Mip-Nerf sampling.
  """
  coarse_sample_store: Optional[types.SampleStoreFn]
  fine_sample_store: types.SampleStoreFn
  semantic_sample_store: Optional[types.SemanticSampleStoreFn]
  num_coarse_samples: int
  num_fine_samples: int
  lindisp: bool
  background_params: types.BackgroundType
  rgb_activation: ActivationFn = nn.sigmoid
  sigma_activation: ActivationFn = nn.relu
  noise_std: float = 0.0
  # If specified, use those static values for near and far plane.
  static_near_far: Optional[Tuple[float, float]] = None
  # By default this class will drop all aux data. This is usally the right thing
  # to do. Only enable for debugging or if you know what you are doing.
  preserve_aux: bool = False
  # By default this class will drop contribution values. This is usally the
  # right thing to do.
  # Only enable for debugging or if you know what you are doing.
  preserve_contribution: bool = False
  # By default this will drop sigma grid values. Only enable if you want
  # to save the sigma grid to disk.
  preserve_sigma_grid: bool = False
  enable_mipnerf: bool = False
  # Set to True when computing ray regularization. Only applicable if
  # enable_mipnerf is True.
  enable_ray_regularization: bool = False

  # This is a workaround for models implementing coarse and fine models when
  # environment variable FLAX_PROFILE=1 is set. Without @nn.nowrap, an internal
  # error is raised from within Flax.
  #
  # Context: https://chat.google.com/room/AAAA3ooiXg8/XJE2xoL45gQ
  @nn.nowrap
  def render_single_stage(self,
                          *,
                          sample_store: types.SampleStoreFn,
                          semantic_sample_store: Optional[
                              types.SemanticSampleStoreFn],
                          z_vals: f32['... num_samples 1'],
                          sample_positions: f32['... num_samples 3'],
                          rays: types.Rays,
                          randomized_sampling: bool,
                          deterministic: bool,
                          name: str,
                          ) -> types.RenderedRays:
    """Renders a single stage (coarse or fine) of NeRF.

    Args:
      sample_store: Callable function given SamplePoints and weight vectors
        returns SampleResults.
      semantic_sample_store: Callable function given SamplePoints and embeddings
        returns semantic predictions. If None, semantics are taken straight from
          the sample_store model.
      z_vals: Sampled z-values.
      sample_positions: Sampled points.
      rays: Rays through a scene.
      randomized_sampling: Boolean, if True random noise is added to the density
        predictions.
      deterministic: Whether to run the model in a deterministic fashion.
      name: String appended to the Background MLP name.

    Returns:
      An instance of RenderedRays.
    """
    if self.enable_mipnerf:
      sample_positions, covariances, z_vals = _conical_frustum_to_gaussian(
          z_vals, rays)
      assert z_vals.shape[-1] == sample_positions.shape[-2]
    else:
      covariances = None
    sample_points = types.SamplePoints(scene_id=rays.scene_id,
                                       position=sample_positions,
                                       covariance=covariances,
                                       direction=rays.direction)
    features = sample_store(sample_points,
                            deterministic)
    # Add noises to regularize the density predictions if needed
    sigma = nerf_utils.add_gaussian_noise(
        key=self.make_rng('sampling') if randomized_sampling else None,
        raw=features.sigma,
        noise_std=self.noise_std,
        randomized=randomized_sampling)
    rgb = self.rgb_activation(features.rgb)
    sigma = self.sigma_activation(sigma)
    # Take semantics from the main model.
    semantic = features.semantic
    # If semantic_sample_store is set, replace semantics with its outputs.
    if semantic_sample_store is not None:
      semantic = semantic_sample_store(
          lax.stop_gradient(sample_points),
          lax.stop_gradient(features.sigma_grid),
          lax.stop_gradient(features.sigma_penultimate_embeddings))

    # Volumetric rendering.
    rendered_result = nerf_utils.volumetric_rendering(
        rgb=rgb,
        sigma=sigma,
        z_vals=z_vals,
        dirs=rays.direction,
        semantic=semantic)

    if self.preserve_aux:
      rendered_result.aux = features.aux

    if self.preserve_sigma_grid:
      rendered_result.sigma_grid = features.sigma_grid

    # Mask results for invalid rays, by replacing the results with zeros in
    # those locations.
    mask = nerf_utils.valid_rays(rays)
    rendered_result.rgb = jnp.where(mask, rendered_result.rgb, 0)
    rendered_result.disparity = jnp.where(mask[..., 0],
                                          rendered_result.disparity, 0)
    if rendered_result.semantic is not None:
      rendered_result.semantic = jnp.where(mask, rendered_result.semantic, 0)
    # Background modeling.
    if self.background_params == types.BackgroundType.NONE:
      return rendered_result

    elif self.background_params in [types.BackgroundType.WHITE,
                                    types.BackgroundType.GREY]:
      background_color = {
          # Pure white.
          types.BackgroundType.WHITE: 1.0,

          types.BackgroundType.GREY: 205./255.
      }[self.background_params]
      rendered_result.rgb = nerf_utils.alpha_composite(
          rendered_result.rgb, background_color, rendered_result.opacity)
      return rendered_result

    else:
      raise NotImplementedError(f'{self.background_params}')

  @nn.compact
  def __call__(
      self,
      rays: Optional[types.Rays],
      randomized_sampling: bool,
      deterministic: bool = True,
      points: Optional[types.SamplePoints] = None,
  ) -> Tuple[Optional[types.RenderResult], Optional[f32['... num_classes']]]:
    """Nerf Model.

    Note querying 3D semantics in mipnerf mode is currently NOT supported!

    Args:
      rays: the Rays to render
      randomized_sampling: Whether or not to randomize sample positions and add
        gaussian noise to sigmas.
      deterministic: Whether to run the model in a deterministic fashion.
      points: Optional set of 3D points to predict semantic class for. If not
        specified, only 'rendered_rays' is returned.

    Returns:
      render results for coarse and fine stage. If points is specified,
      the 3D semantic results for querying the model at the points is also
      returned.
    """
    if rays is None and points is None:
      raise ValueError('Either rays or points needs to be not None!')

    if rays is not None:
      chex.assert_equal_shape((rays.origin, rays.direction))

      if self.static_near_far:
        base = jnp.ones_like(rays.direction[..., :1])
        near = base * self.static_near_far[0]
        far = base * self.static_near_far[1]
      else:
        near, far = nerf_utils.calculate_near_and_far(rays)

      # Stratified sampling along rays
      num_samples = self.num_coarse_samples
      if self.enable_mipnerf:
        # Draw one more sample which later gets reduced in render_single_stage.
        num_samples += 1
      z_vals, sample_positions = nerf_utils.sample_along_rays(
          key=self.make_rng('sampling') if randomized_sampling else None,
          origins=rays.origin,
          directions=rays.direction,
          num_samples=num_samples,
          near=near,
          far=far,
          randomized=randomized_sampling,
          lindisp=self.lindisp)
      chex.assert_shape(sample_positions, (..., num_samples, 3))

      # Hierarchical sampling based on coarse predictions
      coarse_result = None
      if self.num_fine_samples > 0:
        coarse_result = self.render_single_stage(
            sample_store=self.coarse_sample_store,
            semantic_sample_store=None,
            z_vals=z_vals,
            sample_positions=sample_positions,
            rays=rays,
            randomized_sampling=randomized_sampling,
            deterministic=deterministic,
            name='coarse')

        # Mip-NeRF actually samples inbetween the given z_vals.
        if self.enable_mipnerf:
          z_vals = .5 * (z_vals[..., 1:] + z_vals[..., :-1])

        weights = coarse_result.contribution

        num_fine_samples = self.num_fine_samples
        if self.enable_mipnerf:
          num_fine_samples += 1

        # bin_posts marks the end-points for each interval. For each bin as
        # defined by these bin-posts (note that n bin_posts define n-1 bins), we
        # have a weight.
        bin_posts = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        bin_weights = weights[..., 1:-1]

        z_vals, sample_positions = nerf_utils.sample_pdf(
            key=self.make_rng('sampling') if randomized_sampling else None,
            bins=bin_posts,
            weights=bin_weights,
            origins=rays.origin,
            directions=rays.direction,
            z_vals=z_vals,
            num_samples=num_fine_samples,
            randomized=randomized_sampling,
            include_original_z_vals=not self.enable_mipnerf,
        )
        expected_total_num_fine_samples = num_fine_samples
        if not self.enable_mipnerf:
          expected_total_num_fine_samples += self.num_coarse_samples
        chex.assert_shape(
            sample_positions,
            (..., expected_total_num_fine_samples, 3))

      fine_result = self.render_single_stage(
          sample_store=self.fine_sample_store,
          semantic_sample_store=self.semantic_sample_store,
          z_vals=z_vals,
          sample_positions=sample_positions,
          rays=rays,
          randomized_sampling=randomized_sampling,
          deterministic=deterministic,
          name='fine')

      if self.enable_mipnerf and self.enable_ray_regularization:
        fine_result.ray_z_vals = z_vals
      render_result = types.RenderResult(coarse=coarse_result, fine=fine_result)
    else:
      render_result = None

    if points is not None:
      if self.enable_mipnerf:
        semantic_predictions = None
      else:
        assert self.semantic_sample_store is not None, (
            'Cannot evaluate semantics without a semantic store.')
        fine_features = self.fine_sample_store(points, deterministic)
        semantic_predictions = self.semantic_sample_store(
            lax.stop_gradient(points),
            lax.stop_gradient(fine_features.sigma_grid),
            lax.stop_gradient(fine_features.sigma_penultimate_embeddings))
    else:
      semantic_predictions = None
    return render_result, semantic_predictions
