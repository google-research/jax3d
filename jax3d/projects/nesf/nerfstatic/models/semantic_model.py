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

"""Semantic Add-on Model."""

from typing import Optional
from absl import logging
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax3d.projects.nesf.nerfstatic.models import grid_interpolator
from jax3d.projects.nesf.nerfstatic.models import mlp
from jax3d.projects.nesf.nerfstatic.models import unet3d
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import f32


class SemanticModel(nn.Module):
  """Semantic Module.

  Computes the semantic predictions for input SamplePoints, optional sigma grid
  and optional penultimate features from the Sigma MLP.
  Model contains a semantic decoder, UNet and GridInterpolator. Operates with
  both NeRF and PSF models. Note that setting the flag enable_sigma_semantic to
  True gives the 3D4F behavior, whereas False gives the SemanticNerf behavior.

  Attributes:
    grid: the latent grid.
    decoder_params: MlpParams for the decoder.
    unet_params: UNetParams.
    num_posencs: Number of positional encoding dimensions.
    enable_unet: Boolean, whether to pre-apply a 3D UNet on the inputs.
  """
  interpolator: grid_interpolator.GridInterpolator
  decoder_params: mlp.MlpParams
  unet_params: unet3d.UNetParams
  num_posencs: int
  enable_sigma_semantic: bool = True

  @nn.compact
  def __call__(
      self, points: types.SamplePoints,
      sigma_grid: Optional[f32['1 x y z c']],
      sigma_penultimate_features: Optional[f32['...']],
  ) -> f32['... num_semantic_classes']:
    """Evaluate the semantic model.

    Args:
      points: Sample points.
      sigma_grid: Density field.
      sigma_penultimate_features: Penaltimate feature activations from the sigma
        MLP.

    Returns:
      Semantic predictions of shape [.... num_semantic_classes].
    """
    scene_id = jnp.broadcast_to(points.scene_id[..., None, :],
                                points.position.shape[:-1] + (1,))

    semantic_mlp = mlp.MLP(self.decoder_params, name='semantic_mlp')
    if self.enable_sigma_semantic:
      assert sigma_grid is not None
      logging.info('Applying UNet to sigma_grid.')
      embeddings = sigma_grid
      unet = unet3d.UNet3D(params=self.unet_params, name='unet3d')
      embeddings = unet(embeddings)

      # To save HBM, we use pmap and vmap to ensure that only one scene is
      # processed by this function at a time. As a result, the scene_id stored
      # 'points' is incorrect, and should not be used for indexing into
      # 'embeddings'. We guarantee this by ensuring that embeddings describes
      # only one scene and that the lookup key 'scene_id' is set to zero.
      assert embeddings.shape[0] == 1
      scene_id = jax.tree_map(lambda x: x * 0, scene_id)

      latent = self.interpolator(embeddings, scene_id, points.position)
    else:
      assert sigma_penultimate_features is not None
      latent = sigma_penultimate_features

    if self.decoder_params.num_outputs > 0:
      return semantic_mlp(latent).predictions
    else:
      return jnp.zeros((*points.position.shape[:-1], 0),
                       dtype=jnp.float32)

