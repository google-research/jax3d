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

"""Evaluation functionality for GLO NeRF Model."""

from typing import Any, Iterator

from absl import logging
import einops
from flax import jax_utils
import gin

import jax
import jax.numpy as jnp
from jax3d.projects.generative.nerf import camera as jax_camera
from jax3d.projects.generative.nerf import metrics
from jax3d.projects.generative.nerf import visualization
from jax3d.projects.generative.nerf.glo_nerf import models
import numpy as np

_MAX_ANIMATION_RESOLUTION = 128
_EVAL_BATCHES_PER_CHECKPOINT = 400
EVAL_IDS_PER_BATCH = 16

RENDER_RESULT_TO_LABEL = {
    "linear_rgb": "Volumetric Reconstruction",
    "bg_gamma_rgb": "Background Reconstruction",
    "fg_gamma_rgb": "Foreground Volumetric Reconstruction",
    "gamma_rgb": "Volumetric Reconstruction (gamma)",
    "shading_radiance": "Surface Reconstruction (no exposure correction)",
    "shading_linear_rgb": "Surface Reconstruction",
    "shading_gamma_rgb": "Surface Reconstruction (gamma)",
    "alpha": "Volumetric Alpha",
    "depth": "Expected Depth",
    "analytic_normal": "Analytic Normal at Expected Depth",
    "predicted_normal": "Predicted Normal at Expected Depth",
    "uv": "UV Coordinates",
}


def _pad_and_center_image(image, width, height):
  """Pads image(s) to desired width and height, while centering.

  Args:
    image: Input image(s) to pad. Assumed to be in [..., H, W, C] format.
    width: Output image width
    height: Output image height

  Returns:
    A Tensor of the same dimensionality of the input (preserves any batching
    dimensions), but with images padded to the desired width and height.
  """
  assert len(image.shape) >= 3
  batch_dimensions = len(image.shape) - 3
  width_pad = width - image.shape[-2]
  assert width_pad >= 0
  height_pad = height - image.shape[-3]
  assert height_pad >= 0

  # Center image in the padding
  image = jnp.pad(image, ((0, 0),) * batch_dimensions + (
      (height_pad // 2, height_pad - (height_pad // 2)),
      (width_pad // 2, width_pad - (width_pad // 2)),
      (0, 0),
  ))
  return image


def render_id_view_grid(image_renderer, summary_data, model_parameters, step):
  """Render images of each summary identity from multiple viewpoints.

  Returns a composite image grid of identities (rows) and views (columns) for
  all render results.

  Args:
    image_renderer: Renderer function from Model.create_image_renderer.
    summary_data: Summary data from load_summary_data(). Images are assumed to
      have dimension [identity, view, height, width, channels].
    model_parameters: Model parameters from checkpoint.
    step: Training step (needed to set scheduled values correctly).

  Returns:
    A dictionary containing gamma-encoded RGB images visualizing the rendered
    outputs of the model for each identity and view.
  """
  summary_images = summary_data["image"]
  id_count = summary_images.shape[0]
  view_count = summary_images.shape[1]
  # summary_data is padded to the maximum image resolution from
  # tf.Dataset.padded_batch() in data.load_summary_data().
  max_width = summary_images.shape[3]
  max_height = summary_images.shape[2]
  logging.info("Rendering %d ID X %d view image grid (%dx%d resolution each).",
               id_count, view_count, max_width, max_height)

  # Render an image for each view in each identity
  multi_id_multi_view_images = {}
  for id_idx in range(id_count):
    multi_view_images = {}
    for view_idx in range(view_count):
      single_view_latents = summary_data["latents"][id_idx, view_idx]
      logging.info("Rendering identity:%d view_subindex:%d", id_idx, view_idx)
      # pylint: disable=cell-var-from-loop
      camera = jax.tree_map(lambda t: t[id_idx, view_idx],
                            summary_data["camera"])

      inputs = models.ModelInputs(latent_tokens=single_view_latents)

      render_results = image_renderer(
          camera, inputs, model_parameters=model_parameters, step=step)

      rgb_results = visualization.convert_results_to_rgb(render_results)

      for name in rgb_results:
        image = _pad_and_center_image(rgb_results[name], max_width, max_height)
        multi_view_images.setdefault(name, [])
        multi_view_images[name].append(image)

    for name in multi_view_images:
      multi_id_multi_view_images.setdefault(name, [])
      # Concatenate views along H axis (0)
      multi_id_multi_view_images[name].append(
          np.concatenate(multi_view_images[name], axis=0))

  image_grids = {}
  for name in multi_id_multi_view_images:
    # Concatenate IDs along W axis (1)
    image_grids[name] = np.concatenate(multi_id_multi_view_images[name], axis=1)

  return image_grids


@gin.configurable(allowlist=["apply_mask"])
def compute_batch_psnr(model_parameters: models.ModelParameters,
                       latents: np.ndarray,
                       data: ...,
                       step: int,
                       apply_mask: bool = False) -> float:
  """Computes the reconstruction PSNR for a batch of data.

  Args:
    model_parameters: Model parameters from checkpoint.
    latents: ConditionVariables object of all latents required for model.
    data: A batch of data to evaluate.
    step: Training step (needed to set scheduled values correctly).
    apply_mask: Use masked data for PSNR.

  Returns:
    The computed scalar PSNR value.
  """

  # Combine identity and view dimensions as the model does not use them.
  def flatten_views(t):
    return einops.rearrange(t, "V I ... -> (V I) ...")

  data_flat = jax.tree_map(flatten_views, data)
  latents = jax.tree_map(flatten_views, latents)
  origins, directions = jax.vmap(jax_camera.pixels_to_rays)(
      data_flat["camera"], data_flat["pixel_coordinates"])
  rays = jnp.concatenate([origins, directions], axis=-1)
  # Use a constant RNG as randomness is not required
  rng = jax.random.PRNGKey(step)

  inputs = models.ModelInputs(latent_tokens=latents)

  render = models.Model().apply(
      model_parameters, inputs, rays, rng=rng, step=step)

  pred = render["gamma_rgb"]
  gt = data_flat["gamma_rgb"]
  if apply_mask:
    pred *= data_flat["weight"]
    gt *= data_flat["weight"]
  psnr = metrics.psnr(pred, gt)

  return psnr


def compute_eval_psnr(model_parameters: models.ModelParameters,
                      latent_table: np.ndarray, data_iterator: Iterator[Any],
                      psnr_function: ..., step: int) -> float:
  """Computes eval PSNR for one loaded checkpoint.

  Args:
    model_parameters: Model parameters from checkpoint.
    latent_table: ConditionVariables object from checkpoint.
    data_iterator: An iterator yielding batches of data to evaluate.
    psnr_function: A pre-pmap'ed copy of 'compute_batch_psnr'.
    step: Training step (needed to set scheduled values correctly).

  Returns:
    The computed scalar PSNR value.
  """
  model_parameters_replicated = jax_utils.replicate(model_parameters)
  step_replicated = jax_utils.replicate(step)

  mean_psnr = 0.0

  for i in range(_EVAL_BATCHES_PER_CHECKPOINT):
    logging.info("Computing PSNR of Eval data - batch %d/%d", i + 1,
                 _EVAL_BATCHES_PER_CHECKPOINT)
    batch = next(data_iterator)
    batch_latents = latent_table[batch["identity_index"]]
    batch_psnr = psnr_function(model_parameters_replicated, batch_latents,
                               batch, step_replicated)
    mean_psnr += jnp.mean(batch_psnr)

  return mean_psnr / _EVAL_BATCHES_PER_CHECKPOINT
