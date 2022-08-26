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

"""Evaluation script for 2D autoencoder transformer model."""

import einops
import jax
import jax.numpy as jnp
from jax3d.projects.generative.nerf import metrics
from jax3d.projects.generative.nerf.autoencoder import models


EVAL_BATCHES_PER_CHECKPOINT = 256
EVAL_IDS_PER_BATCH = 8


def render_frames(model_parameters: models.ModelParameters, data: ...,
                  step: int):
  """Render summary images for TensorBoard."""
  # Combine identity and view dimensions as the model does not use them.

  data_flat = data

  gt_rgb = data_flat["image_data"]["image"]

  frames = []
  for i in range(gt_rgb.shape[0]):
    predicted_rgb = models.Model().apply(
        model_parameters, gt_rgb[i:i+1], step=step)
    frames.append(predicted_rgb[0])

  return jnp.concatenate(frames, axis=1)


def psnr(model_parameters: models.ModelParameters, data: ..., step: int):
  """Compute PSNR for test images."""
  # Combine identity and view dimensions as the model does not use them.
  def flatten_views(t):
    return einops.rearrange(t, "V I ... -> (V I) ...")

  data_flat = jax.tree_map(flatten_views, data)

  gt_rgb = data_flat["image_data"]["image"]

  frames = []
  for i in range(gt_rgb.shape[0]):
    predicted_rgb = models.Model().apply(
        model_parameters, gt_rgb[i:i+1], step=step)
    frames.append(predicted_rgb[0])

  return metrics.psnr(gt_rgb, jnp.stack(frames, axis=0))
