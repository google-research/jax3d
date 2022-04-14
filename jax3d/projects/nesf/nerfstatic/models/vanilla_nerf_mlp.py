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

"""Vanilla Nerf MLP implementation."""

import functools
from typing import Optional, Tuple

from flax import linen as nn
import jax
from jax import numpy as jnp

from jax3d.projects.nesf.nerfstatic.models import mlp
from jax3d.projects.nesf.nerfstatic.models import model_utils
from jax3d.projects.nesf.nerfstatic.models import nerf_utils
from jax3d.projects.nesf.nerfstatic.utils import types


class VanillaNerfMLP(nn.Module):
  """The vanilla NeRF MLP, including positional encoding."""
  net_params: mlp.MlpParams
  viewdir_net_params: mlp.MlpParams
  num_posencs: int  # Positional encoding for the spatial position.
  viewdir_num_posencs: int  # Positional encoding for the view direction.
  use_viewdirs: bool = True  # Whether or not to use view directions.
  num_scenes: int = 1  # Number of scene embeddings.
  num_scene_features: int = 0  # Number of per scene features.
  enable_sigma_semantic: bool = False
  sigma_grid_size: Optional[Tuple[int, int, int]] = None
  enable_mipnerf: bool = False

  @nn.compact
  def __call__(self, points: types.SamplePoints,
               input_views: Optional[types.Views] = None,
               deterministic: bool = True,
               ) -> types.SampleResults:
    """Apply positional encoding and evaluate the MLP.

    Points are assumed to be within the [-1; 1] bounding box.

    Args:
      points: Sample points.
      input_views: Optional input views. Unused.
      deterministic: Whether to run the model in a deterministic fashion.
        Unused.

    Returns:
      Sample results. Note that semantics is filled with a placeholder value
      of 0 which will be later replaced in nerf_renderer.py. Further note
      that embeddings is a dummy variable that is needed for the semantic_model.
      It is not used with NeRF model.
    """
    del input_views, deterministic  # Unused.
    positions = points.position
    if self.enable_mipnerf:
      point_features = nerf_utils.integrated_posenc(points, self.num_posencs)
    else:
      point_features = nerf_utils.posenc(positions, self.num_posencs)
    if self.num_scene_features > 0:
      scenes_embeddings = nn.Embed(self.num_scenes,
                                   self.num_scene_features,
                                   name="scenes")(points.scene_id)
      scenes_embeddings = jnp.broadcast_to(scenes_embeddings,
                                           point_features.shape[:-1] +
                                           scenes_embeddings.shape[-1:])
      point_features = jnp.concatenate([point_features, scenes_embeddings],
                                       axis=-1)
    dense_layer = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
    sigma_mlp = mlp.MLP(params=self.net_params, name="sigma_mlp")
    sigma_output = sigma_mlp(point_features)
    raw_sigma = sigma_output.predictions
    latent_layer = sigma_output.penultimate_features

    sigma_grid = None
    # In case of end-to-end training of semantics with reconstruction, it is
    # reasonable to generate a new sigma grid at every method call. However,
    # in the case of splitting the code into separate stages with semantics and
    # reconstruction, the variable `enable_sigma_grid` should be restructured to
    # be part of the method call to save on memory.
    if self.enable_sigma_semantic:
      sigma_grid = model_utils.generate_sigma_grid(
          num_scenes=self.num_scenes,
          grid_size=self.sigma_grid_size,
          embeddings=None,
          grid=None,
          num_posencs=self.num_posencs,
          sigma_decoder=sigma_mlp,
      )

    if self.use_viewdirs:
      # Output of the first part of MLP.
      bottleneck = dense_layer(self.viewdir_net_params.width,
                               name="bottleneck")(latent_layer)
      # Broadcast viewdir from [..., 3] to [..., num_sample_points, 3] since all
      # the sample points for the same ray have the same viewdir.
      view_dirs = jnp.broadcast_to(points.direction[..., None, :],
                                   positions.shape)
      view_dirs = nerf_utils.posenc(
          view_dirs, self.viewdir_num_posencs)
      rgb_latent = jnp.concatenate([bottleneck, view_dirs], axis=-1)
      rgb_mlp = mlp.MLP(self.viewdir_net_params, name="rgb_mlp")
      raw_rgb = rgb_mlp(rgb_latent).predictions
    else:
      raw_rgb = dense_layer(3, name="rgb_out")(latent_layer)

    raw_semantics = jnp.zeros((*raw_rgb.shape[:-1], 0), dtype=jnp.float32)

    return types.SampleResults(
        rgb=raw_rgb, sigma=raw_sigma,
        semantic=raw_semantics,
        sigma_penultimate_embeddings=sigma_output.penultimate_features,
        sigma_grid=sigma_grid)
