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

"""A simple MLP which can be used to decode latents into densities or rgbs."""

import dataclasses
import functools
from typing import Optional

from flax import linen as nn
import jax
from jax import numpy as jnp

from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import ActivationFn
from jax3d.projects.nesf.utils.typing import f32


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class MlpParams:
  """MLP model parameters."""

  depth: int
  width: int
  num_outputs: int
  skip_layer: Optional[int] = 0
  activation: ActivationFn = nn.relu


class MLP(nn.Module):
  """A simple MLP."""
  params: MlpParams  # Network parameters.

  @nn.compact
  def __call__(self, input_feats: f32['... feature']) -> types.MlpOutputs:
    """Evaluate the MLP.

    Args:
      input_feats: jnp.ndarray(float32), [batchdims, feature]

    Returns:
      An instance of types.MlpOutputs.
    """
    dense_layer = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
    x = input_feats
    for i in range(self.params.depth):
      x = dense_layer(self.params.width)(x)
      x = self.params.activation(x)
      if self.params.skip_layer and i % self.params.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, input_feats], axis=-1)
    penultimate_features = x
    predictions = dense_layer(self.params.num_outputs)(x)
    return types.MlpOutputs(predictions=predictions,
                            penultimate_features=penultimate_features)
