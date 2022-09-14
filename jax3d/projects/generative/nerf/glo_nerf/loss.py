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

"""Main loss function for the GLO NeRF model."""

from typing import Any, Dict, Tuple

from etils.array_types import FloatArray, PRNGKey  # pylint: disable=g-multiple-import
import gin
import jax
import jax.numpy as jnp
from jax3d.projects.generative.common import image_utility
from jax3d.projects.generative.nerf import camera as jax_camera
from jax3d.projects.generative.nerf import losses
from jax3d.projects.generative.nerf import metrics
from jax3d.projects.generative.nerf.glo_nerf import models


@gin.configurable(allowlist=["mask_mode"])
def transformer_nerf_loss_fn(
    model_parameters: models.ModelParameters,
    inputs: models.ModelInputs,
    data: Dict[str, Any],
    rng: PRNGKey,
    step: jnp.ndarray,
    mask_mode: str = "alpha_supervision",
    reconstruct_gamma_rgb: bool = True
) -> Tuple[FloatArray, Dict[str, FloatArray]]:
  """The main GLO NeRF loss function including all terms.

  Args:
    model_parameters: Network weights needed to invoke the model.
    inputs: Model inputs.
    data: Dictionary of training data.
    rng: The RNG key for the current training step.
    step: The current step number for scheduling.
    mask_mode: One of the following modes:
      'alpha_supervision': Masks are used to supervise the density field.
      'reconstruction_weight': Masks are used to only train the foreground.
      'none': Masks are ignored in photometric reconstruction.
    reconstruct_gamma_rgb: Whether photometric reconstruction should operate on
      gamma-encoded RGB values. Otherwise, reconstruction will be done on linear
      radiance values.

  Returns:
    total_loss: The weighted sum of all loss terms.
    loss_terms: A dictionary of unweighted loss terms for logging.
  """

  # Combine identity and view dimensions as the model does not use them.
  def flatten_views(t):
    return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])

  data = jax.tree_map(flatten_views, data)
  latent_tokens = inputs.latent_tokens
  latent_tokens = jax.tree_map(flatten_views, latent_tokens)
  inputs = inputs.replace(latent_tokens=latent_tokens)

  origins, directions = jax.vmap(jax_camera.pixels_to_rays)(
      data["camera"], data["pixel_coordinates"])
  rays = jnp.concatenate([origins, directions], axis=-1)

  render = models.Model().apply(
      model_parameters,
      inputs,
      rays,
      rng=rng,
      step=step,
      return_additional_sample_data=True,
      is_training=True)

  loss_terms = {}
  total_loss = 0.0

  foreground_mask = data["weight"]

  if mask_mode == "reconstruction_weight":
    photometric_reconstruction_mask = foreground_mask
  else:
    photometric_reconstruction_mask = jnp.ones_like(foreground_mask)

  if reconstruct_gamma_rgb:
    # Gamma encoding is generally regarded as a good approximation of human
    # brightness perception in images. Note if other NeRF literature is not
    # gamma decoding their images this is effectively the loss they are using.
    # (https://en.wikipedia.org/wiki/Gamma_correction).
    gt_rgb = data["gamma_rgb"]
    predicted_rgb = render["gamma_rgb"]
  else:
    gt_rgb = image_utility.srgb_gamma_to_linear(data["gamma_rgb"])
    predicted_rgb = render["linear_rgb"]

  if mask_mode == "multiply":
    gt_rgb *= foreground_mask

  with gin.config_scope("volumetric_rgb"):
    reconstruction_loss, reconstruction_loss_weight = losses.reconstruction(
        gt_rgb, predicted_rgb, photometric_reconstruction_mask)
  if reconstruction_loss_weight != 0.0:
    loss_terms["Reconstruction"] = reconstruction_loss
    total_loss += reconstruction_loss * reconstruction_loss_weight

  # Always compute PSNR on gamma values for consistency
  gt_rgb = data["gamma_rgb"]
  if mask_mode == "multiply":
    gt_rgb *= foreground_mask
  psnr = metrics.psnr(render["gamma_rgb"], gt_rgb)
  loss_terms["Training PSNR"] = psnr

  if mask_mode == "alpha_supervision":
    with gin.config_scope("alpha"):
      alpha_loss, alpha_loss_weight = losses.reconstruction(
          foreground_mask, render["alpha"])
    if alpha_loss_weight != 0.0:
      loss_terms["Alpha"] = alpha_loss
      total_loss += alpha_loss_weight * alpha_loss

  hard_surface_loss, hard_surface_loss_weight = losses.hard_surface(
      render["sample_weights"])
  loss_terms["Hard Surface"] = hard_surface_loss
  total_loss += hard_surface_loss_weight * hard_surface_loss

  return total_loss, loss_terms
