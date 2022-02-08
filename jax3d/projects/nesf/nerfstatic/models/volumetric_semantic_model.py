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

"""Volumetric Semantic Model."""

from typing import Optional, Tuple, Union

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp

from jax3d.projects.nesf.nerfstatic.models import grid_interpolator
from jax3d.projects.nesf.nerfstatic.models import mlp
from jax3d.projects.nesf.nerfstatic.models import model_utils
from jax3d.projects.nesf.nerfstatic.models import nerf_utils
from jax3d.projects.nesf.nerfstatic.models import semantic_model
from jax3d.projects.nesf.nerfstatic.models import unet3d
from jax3d.projects.nesf.nerfstatic.models import vanilla_nerf_mlp
from jax3d.projects.nesf.nerfstatic.utils import geometry_utils as geom
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import f32, Tree  # pylint: disable=g-multiple-import


class VolumetricSemanticModel(nn.Module):
  """Volumetric Semantic Module.


  Computes the semantic predictions for input SamplePoints and sigma grid
  features from the Sigma MLP then calls the volumetric rendering equation to
  return an instance of RenderedRays.

  The module contains a nerf_model, a UNet and semantic decoder.

  Attributes:
    nerf_model: vanille_nerf_mlp.
    interpolator: grid interpolator for the semantic model.
    semantic_decoder_params: semantic decoder params.
    unet_params: UNet params.
    num_posencs: (ignored)
    num_samples: Number of samples along the ray.
    lindisp: If true, sample linearly in disparity otherwise in depth.
    static_near_far: Optional fixed values for the near and far planes. If left
      empty, the near and far planes are calculated from the input rays.
    threshold: if set, f(x) = max(x, threshold) is applied to sigma_grid.
    binarization: if set, f(x) = where(x > threshold, 1, 0) is applied to
      sigma_grid. Has no effect if threshold is not set.
    apply_random_scene_rotations: if set, apply random scene rotations at
      train time.
    random_scene_rotation_max_angle: maximum angle for random scene rotations
      in radians.
  """
  nerf_model: vanilla_nerf_mlp.VanillaNerfMLP
  interpolator: grid_interpolator.GridInterpolator
  semantic_decoder_params: mlp.MlpParams
  unet_params: unet3d.UNetParams
  num_posencs: int
  num_samples: int
  lindisp: bool
  static_near_far: Optional[Tuple[float, float]] = None
  threshold: Optional[int] = None
  binarization: bool = False
  apply_random_scene_rotations: bool = False
  random_scene_rotation_max_angle: float = 2 * jnp.pi

  @nn.compact
  def __call__(
      self,
      rays: types.Rays,
      randomized_sampling: bool,
      is_train: bool,
      sigma_grid: f32['1 x y z c'],
      nerf_model_weights: Tree[jnp.ndarray],
      points: Optional[types.SamplePoints] = None,
  ) -> Union[
      types.RenderedRays,
      Tuple[types.RenderedRays, f32['k num_classes']]]:
    """Apply VolumetricSemanticModel.

    Args:
      rays: Camera rays. Model will predict semantic logits for each ray.
      randomized_sampling: bool. If True, add random jittering when sampling
        points along each camera ray.
      is_train: bool. If True, model is training in training mode and various
        randomized behavior will be enabled.
      sigma_grid: density grid describing scene's 3D geometry.
      nerf_model_weights: parameters for a single NeRF model. Used in
        volumetric rendering.
      points: Optional set of 3D points to predict semantic class for. If not
        specified, only 'rendered_rays' is returned.

    Returns:
      rendered_rays: Predicted RGB, opacity, and semantic logits for each
        camera ray.
      semantic_3d_predictions: Predicted semantic logits for each 3D point.
        Only returned when 'points' is defined.
    """
    chex.assert_equal_shape((rays.origin, rays.direction))

    # Construct outer_to_inner and inner_to_outer transforms.
    outer_to_inner = geom.Identity()
    if self.apply_random_scene_rotations:
      if is_train:
        angle = jax.random.uniform(
            self.make_rng('data_augmentation'),
            minval=0.0, maxval=self.random_scene_rotation_max_angle)
      else:
        angle = 0.0
      outer_to_inner = _create_outer_to_inner(angle)
      del angle
    inner_to_outer = geom.Inverse(transform=outer_to_inner)

    # Construct ray representation for the inside coordinate system.
    rays_in = outer_to_inner.forward(rays)
    chex.assert_shape(rays_in.scene_id, rays.batch_shape + (1,))
    chex.assert_shape(rays_in.origin, rays.batch_shape + (3,))
    chex.assert_shape(rays_in.direction, rays.batch_shape + (3,))

    if self.static_near_far:
      base = jnp.ones_like(rays.direction[..., :1])
      near = base * self.static_near_far[0]
      far = base * self.static_near_far[1]
    else:
      # Intersect ray with the [-1, 1]^3. Use the inner coordiante system.
      # Output is independent of the choice of coordinate system.
      near, far = nerf_utils.calculate_near_and_far(rays_in)

    # Stratified sampling along rays
    #
    # Note: we can only do the stratified sampling as we do not have knowledge
    # of weights for the bins where we can divide the samples across in space.
    #
    z_vals, sample_positions_in = nerf_utils.sample_along_rays(
        key=self.make_rng('sampling') if randomized_sampling else None,
        origins=rays_in.origin,
        directions=rays_in.direction,
        num_samples=self.num_samples,
        near=near,
        far=far,
        randomized=randomized_sampling,
        lindisp=self.lindisp)
    chex.assert_shape(z_vals, (*rays.batch_shape, self.num_samples))
    chex.assert_shape(sample_positions_in,
                      (*rays.batch_shape, self.num_samples, 3))

    # Construct sample points for the inside and outside coordinate systems.
    sample_points_in = types.SamplePoints(scene_id=rays_in.scene_id,
                                          position=sample_positions_in,
                                          direction=rays_in.direction)
    sample_points = inner_to_outer.forward(sample_points_in)

    # Construct SemanticModel.
    #
    # NOTE: SemanticModel operates on the "inner" coordinate system. If you
    # have points or rays in the "outer" coordinate system, apply
    # "inner_to_outer" first!!
    semantic_net = semantic_model.SemanticModel(
        interpolator=self.interpolator,
        decoder_params=self.semantic_decoder_params,
        unet_params=self.unet_params,
        num_posencs=self.num_posencs,
        enable_sigma_semantic=True)

    # Recompute sigma grid (if necessary).
    if self.apply_random_scene_rotations:
      sigma_grid = self._recompute_sigma_grid(nerf_model_weights,
                                              sigma_grid.shape,
                                              inner_to_outer=inner_to_outer)

    # Apply thresholding and binarization.
    if self.threshold:
      sigma_grid = self._apply_threshold(sigma_grid)

    # 2D inference. Operation is applied in the inside coordinate system.
    semantic_point_predictions = semantic_net(
        sample_points_in, sigma_grid=sigma_grid,
        sigma_penultimate_features=None)

    # Calculate density, RGB. Operation is applied in the outside coordinate
    # system.
    nerf_features = self.nerf_model.apply(nerf_model_weights, sample_points)
    rgb = nn.sigmoid(nerf_features.rgb)
    sigma = nn.relu(nerf_features.sigma)

    # Verify shapes.
    batch_shape = sample_points.batch_shape
    chex.assert_shape(rgb, batch_shape + (3,))
    chex.assert_shape(sigma, batch_shape + (1,))
    chex.assert_shape(z_vals, batch_shape)
    chex.assert_shape(
        semantic_point_predictions,
        batch_shape + (self.semantic_decoder_params.num_outputs,))

    rendered_rays = nerf_utils.volumetric_rendering(
        rgb=rgb, sigma=sigma, z_vals=z_vals, dirs=None,
        semantic=semantic_point_predictions)

    # 3D inference.
    if points is None:
      return rendered_rays

    # This operation is applied in the inside coordinate system.
    semantic_3d_predictions = semantic_net(
        outer_to_inner.forward(points), sigma_grid=sigma_grid,
        sigma_penultimate_features=None)

    return rendered_rays, semantic_3d_predictions

  def _apply_threshold(self,
                       sigma_grid: f32['1 x y z c'],
                       ) -> f32['1 x y z c']:
    """Apply (optional) thresholding and binarization to sigma_grid."""
    if self.binarization:
      return jnp.where(sigma_grid > self.threshold, 1, 0)
    return jnp.where(sigma_grid > self.threshold, sigma_grid, 0)

  def _recompute_sigma_grid(self,
                            variables: Tree[jnp.ndarray],
                            sigma_grid_shape: Tuple[int, int, int, int, int],
                            inner_to_outer: geom.Transform,
                            ) -> f32['1 x y z c']:
    """Reconstructs NeRF sigma grid.

    Args:
      variables: Variables for self.nerf_model.
      sigma_grid_shape: Desired shape of sigma grid.
      inner_to_outer: Transform describing the transformation from the inner
        to outer coordinate systems.

    Returns:
      sigma_grid: Sigma grid for use with SemanticModel.
    """
    # The logic below only handles a single scene.
    assert len(sigma_grid_shape) == 5
    assert sigma_grid_shape[0] == 1

    # Construct evenly-spaced XYZ positions in the [-1, 1] grid.
    spatial_shape = sigma_grid_shape[1:-1]
    sigma_grid_positions = model_utils.generate_grid(num_scenes=1,
                                                     grid_size=spatial_shape)
    sample_points_in = types.SamplePoints(
        scene_id=jnp.asarray([[0]]),  # arbitrary value
        position=sigma_grid_positions,
        direction=jnp.zeros((1, 3)))  # arbitrary value

    # shape=[1, n, 1]. This operation is applied in the outside coordinate
    # system.
    sample_points = inner_to_outer.forward(sample_points_in)
    sample_results = self.nerf_model.apply(variables, sample_points)

    # Apply sigma-to-density transform.
    sigma_grid = 1. - jnp.exp(-1 * jax.nn.relu(sample_results.sigma))

    # shape=[1, x, y, z, k]
    sigma_grid = jnp.reshape(sigma_grid, [1, *spatial_shape, 1])

    return sigma_grid


def _create_outer_to_inner(angle: f32['']) -> geom.Transform:
  """Constructs an "outside" box to "inside" box transform.

  Consider an "outside" box of shape [-1, 1]^3. We define an "inside" box to
  be a box that sits inside of the "outside" box with a rotation about the
  z-axis. The inside box is defined such that it will always lie within the
  outside box, regardless of the amount of rotation.

  The inside box has its own coordinate system such that all points in
  [-1, 1]^3 in the inside box's coordinate system lies within the inside box.
  That is, if you sample a point p = [x, y, z] ~ Uniform(-1, 1), then apply
  p' = outer_to_inner.backward(p), then p' will lie within the inside box
  according to the outside box's coordinate system.

  This function produces a coordinate transform from the outside box's
  coordinate system to the inside box's.

  For example,
  ```
    # Sample a ray in the inside box's coordinate system.
    ray_inside = Rays(origin=np.random.uniform(-1, 1, size=(1, 3)),
                      direction=normalize(np.random.randn(1, 3)))

    # Construct transform.
    outer_to_inner = outer_box_to_rotated_inner_box_transform(angle=np.pi/4)

    # Calculate origin, direction of the ray in the outside box's coordinate
    # system.
    ray_outside = outer_to_inner.backward(rays_inside)
  ```

  Args:
    angle: angle to rotate the inner box about the z-axis. Measured in radians.

  Returns:
    outer_to_inner: Transform such that outer_to_inner.forward(rays) transforms
      rays from the outer box's coordinate system to the inner box's.
  """
  # Imagine we have a square of shape [-1, 1]^2. Imagine the 2D point (1, 1)
  # in the inside square's coordinate system. What is this point in the outside
  # square's coordinate system? Answer: 1/sqrt(2).
  scale = geom.Scale(scale=jnp.array([1./jnp.sqrt(2), 1./jnp.sqrt(2), 1.]))

  # We rotate -1 * angle because we are constructing the inside-to-outside
  # transform. The outside-to-inside transform will apply a rotation in the
  # opposite direction.
  rotate = geom.Rotate(axis=jnp.array([0, 0, 1]), radians=(-1 * angle))

  return geom.Inverse(transform=geom.Compose(transforms=[scale, rotate]))
