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

"""Main loss function for the 2D autoencoder model."""

from typing import Any, Dict, Tuple

import einops
from etils.array_types import FloatArray, PRNGKey  # pylint: disable=g-multiple-import
import gin
import jax
import jax.numpy as jnp
from jax3d.projects.generative.nerf import losses
from jax3d.projects.generative.nerf import metrics
from jax3d.projects.generative.nerf.autoencoder import models


@gin.configurable(allowlist=["pixel_batch_size"])
def transformer_loss_fn(
    model_parameters: models.ModelParameters,
    data: Dict[str, Any],
    rng: PRNGKey,
    step: jnp.ndarray,
    pixel_batch_size=512,
) -> Tuple[FloatArray, Dict[str, FloatArray]]:
  """The main autoencoder loss function.

  Args:
    model_parameters: Network weights needed to invoke the model.
    data: Dictionary of training data.
    rng: The RNG key for the current training step.
    step: The current step number for scheduling.
    pixel_batch_size: number of pixels to sample per image.

  Returns:
    total_loss: The weighted sum of all loss terms.
    loss_terms: A dictionary of unweighted loss terms for logging.
  """

  # Combine identity and view dimensions as the model does not use them.
  def flatten_views(t):
    return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])

  data = jax.tree_map(flatten_views, data)

  gt_rgb = data["image_data"]["image"]

  x = jnp.linspace(0.0, 1.0, gt_rgb.shape[2])
  y = jnp.linspace(0.0, 1.0, gt_rgb.shape[1])
  pixels = jnp.stack(jnp.meshgrid(x, y, indexing="xy"), axis=-1)
  pixels = jnp.tile(pixels[None], (gt_rgb.shape[0], 1, 1, 1))
  pixels = einops.rearrange(pixels, "B H W uv -> B (H W) uv")

  inds = jax.random.randint(rng, (gt_rgb.shape[0], pixel_batch_size), 0,
                            pixels.shape[1])

  @jax.vmap
  def take(pixels, inds):
    return pixels[inds]

  pixel_batch = take(pixels, inds)

  predicted_rgb = models.Model().apply(
      model_parameters,
      gt_rgb,
      pixel_batch,
      step=step,
      is_training=True)

  gt_rgb = einops.rearrange(gt_rgb, "B H W rgb -> B (H W) rgb")
  gt_rgb = take(gt_rgb, inds)

  loss_terms = {}
  total_loss = 0.0

  with gin.config_scope("rgb"):
    reconstruction_loss, reconstruction_loss_weight = losses.reconstruction(
        gt_rgb, predicted_rgb)
  if reconstruction_loss_weight != 0.0:
    loss_terms["Reconstruction"] = reconstruction_loss
    total_loss += reconstruction_loss * reconstruction_loss_weight

  # Always compute PSNR on gamma values for consistency
  psnr = metrics.psnr(predicted_rgb, gt_rgb)
  loss_terms["Training PSNR"] = psnr

  return total_loss, loss_terms
