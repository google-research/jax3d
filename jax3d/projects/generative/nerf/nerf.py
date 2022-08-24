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

"""Flax module to provide a basic MLP NeRF model implementation."""
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import gin
import jax
import jax.numpy as jnp
from jax3d.projects.generative.nerf import positional_encoding


@gin.configurable
class PositionalEncoding(nn.Module):
  """Positional encoding module.

  Attributes:
    min_frequency_degree: log2 of the minimum frequency of the encoding.
    max_frequency_degree: log2 of the maximum frequency of the encoding.
    use_identity: whether to include identity channels in the encoding.
  """
  min_frequency_degree: int = 0
  max_frequency_degree: int = 4
  use_identity: bool = True

  @nn.compact
  def __call__(self, coordinates):
    """Encodes the input coordinates using an axis-aligned sinusoidal encoding.

    Args:
      coordinates: Input tensor with coordinates along the last dimension.

    Returns:
      Positional encoding of the input coordinates.
    """
    return positional_encoding.sinusoidal(
        coordinates,
        self.min_frequency_degree,
        self.max_frequency_degree,
        include_identity=self.use_identity)


@gin.configurable
class NeRF(nn.Module):
  """Basic MLP NeRF model implementation as a flax module.

  Attributes:
    use_viewdirs: Whether to condition color branch on view direction.
    use_feature_skip: Whether to include sample features in skip connections.
    noise_std: Standard deviation of noise used to regularize density.
    nerf_trunk_depth: Number of layers in the network trunk.
    nerf_trunk_width: Width of layers in the network trunk.
    nerf_density_branch_depth: Number of layers in the density network.
    nerf_density_branch_width: Width of layers in the density network.
    nerf_rgb_branch_depth: Number of layers in the rgb network.
    nerf_rgb_branch_width: Width of layers in the rgb network.
    nerf_skips: Indices of layers in the trunk to receive skip connections.
    dtype: Floating point format to use for network weights and outputs. All
      density and radiance values will be converted to float32 before being
      returned from the module.
    activation: Activation function to use for intermediate layers.
    sigma_activation: Activation function to use for density values.
    rgb_activation: Activation function to use for color values.
  """

  # NeRF architecture.
  use_viewdirs: bool = False
  use_feature_skip: bool = True
  noise_std: Optional[float] = None
  nerf_trunk_depth: int = 8
  nerf_trunk_width: int = 256
  nerf_density_branch_depth: int = 1
  nerf_density_branch_width: int = 128
  nerf_rgb_branch_depth: int = 1
  nerf_rgb_branch_width: int = 128
  nerf_skips: Tuple[int] = (4,)

  dtype: Any = "float32"
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  sigma_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
  rgb_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.sigmoid

  @nn.compact
  def __call__(self,
               sample_positions,
               sample_features,
               sample_directions=None,
               noise_rng=None):
    """Sample the radiance and density fields at sample points.

    Args:
      sample_positions: spatial positions of sample points.
      sample_features: per-sample feature vectors to condition the field.
      sample_directions: direction vector associated with each sample.
      noise_rng: PRNGKey used to generate noise for density predictions.

    Returns:
      rgb: per-sample RGB value. Can be treated as either gamma encoded or
        linear.
      density: per-sample scalar density value.
    """
    with gin.config_scope("spatial_encoding"):
      posenc = PositionalEncoding()(sample_positions)
      posenc = posenc.astype(self.dtype)

    sample_features = sample_features.astype(self.dtype)
    net = jnp.concatenate((posenc, sample_features), axis=-1)
    for i in range(self.nerf_trunk_depth):
      if i in tuple(self.nerf_skips):
        if self.use_feature_skip:
          net = jnp.concatenate((net, posenc, sample_features), axis=-1)
        else:
          net = jnp.concatenate((net, posenc), axis=-1)

      net = nn.Dense(self.nerf_trunk_width, dtype=self.dtype)(net)
      net = self.activation(net)
    common_feature = net

    for i in range(self.nerf_density_branch_depth):
      net = nn.Dense(self.nerf_density_branch_width, dtype=self.dtype)(net)
      net = self.activation(net)
    density = nn.Dense(1, dtype=self.dtype)(net).astype(jnp.float32)
    if self.noise_std is not None and noise_rng is not None:
      density += jax.random.normal(noise_rng, density.shape) * self.noise_std
    density = self.sigma_activation(density)[..., 0]

    net = common_feature
    if self.use_viewdirs:
      with gin.config_scope("view_encoding"):
        viewenc = PositionalEncoding()(sample_directions)
        viewenc = viewenc.astype(self.dtype)
      net = jnp.concatenate((net, viewenc), axis=-1)

    for i in range(self.nerf_rgb_branch_depth):
      net = nn.Dense(self.nerf_rgb_branch_width, dtype=self.dtype)(net)
      net = self.activation(net)
    net = nn.Dense(3, dtype=self.dtype)(net).astype(jnp.float32)
    rgb = self.rgb_activation(net)

    return rgb, density
