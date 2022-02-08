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

"""A UNet-3D model."""

from dataclasses import dataclass
from typing import Tuple

from flax import linen as nn
import jax.numpy as jnp
from jax3d.projects.nesf.utils.typing import ActivationFn, f32  # pylint: disable=g-multiple-import


@dataclass
class UNetParams:
  """Parameters for the UNet model."""
  feature_size: Tuple[int, int, int, int]
  output_dims: int
  depth: int = 3
  activation_fn: ActivationFn = nn.relu


class UNet3D(nn.Module):
  """A UNet3D model."""
  params: UNetParams  # Network parameters.
  num_spatial_dims: int = 3

  @nn.compact
  def __call__(self,
               input_features: f32['... input_dims']) -> f32['... feat_dims']:
    """Applies the convnet on the input features.

    Args:
      input_features: grid-like structure.

    Returns:
      convnet output.
    """

    def apply_convblock(feature: f32['... dims'],
                        feature_sizes: Tuple[int, int]):
      latent = feature
      for feature_size in feature_sizes:
        latent = nn.Conv(features=feature_size,
                         kernel_size=(3,)*self.num_spatial_dims,
                         strides=(1,)*self.num_spatial_dims)(latent)
        # latent = nn.BatchNorm(latent)
        latent = self.params.activation_fn(latent)
      return latent

    def apply_transpose_convblock(feature: f32['... m_dims'],
                                  skip_feature: f32['... n_dims'],
                                  feature_sizes: Tuple[int, int]):
      latent = feature
      for i, feature_size in enumerate(feature_sizes):
        if i == 0:
          latent = nn.ConvTranspose(
              features=feature_size,
              kernel_size=(2,)*self.num_spatial_dims,
              strides=(2,)*self.num_spatial_dims)(latent)
          latent = jnp.concatenate([latent, skip_feature], axis=-1)
        else:
          latent = nn.Conv(features=feature_size,
                           kernel_size=(3,)*self.num_spatial_dims,
                           strides=(1,)*self.num_spatial_dims)(latent)
        return self.params.activation_fn(latent)

    output_convblocks = []
    features = input_features
    for i in range(self.params.depth):
      output_convblocks.append(apply_convblock(
          feature=features,
          feature_sizes=self.params.feature_size[i:i+2]))
      if i < self.params.depth - 1:
        features = nn.max_pool(output_convblocks[i],
                               window_shape=(2,)*self.num_spatial_dims,
                               strides=(2,)*self.num_spatial_dims)
      else:
        features = output_convblocks[i]

    for i in range(self.params.depth - 1):
      features = apply_transpose_convblock(
          feature=features,
          skip_feature=output_convblocks[self.params.depth - i - 2],
          feature_sizes=self.params.feature_size[self.params.depth -
                                                 i:self.params.depth - i -
                                                 2:-1])

    features = nn.Conv(features=self.params.output_dims,
                       kernel_size=(1,)*self.num_spatial_dims)(features)

    return features
