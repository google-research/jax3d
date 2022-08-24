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

"""Common utilities for visualizing NeRF outputs."""

from typing import Dict, Union

import einops
import jax.numpy as jnp
from jax3d.projects.generative.common import image_utility
from matplotlib.pyplot import cm as colormap
import numpy as np


def _mono_to_srgb_gamma(rgb):
  # Useful for visualizing alpha maps. "Why gamma correction on alpha?" you may
  # ask. Think of the visualization as viewing an X-ray slide, where the alhpa
  # map "gates" a white background. You then need gamma encoding to store the
  # proper radiance level.
  rgb = einops.repeat(rgb, "... 1 -> ... rgb", rgb=3)
  return image_utility.linear_to_srgb_gamma(rgb)


def _depth_to_rgb(depth):
  depth -= np.min(depth)
  depth /= np.max(depth)
  return colormap.turbo(depth[..., 0])[..., :3]


def _normal_to_rgb(normal):
  return (normal + 1.0) / 2.0


_RENDER_RESULT_TO_RGB_FN = {
    "linear_rgb": image_utility.linear_to_srgb_gamma,
    "fg_gamma_rgb": lambda x: x,
    "gamma_rgb": lambda x: x,
    "shading_radiance": image_utility.linear_to_srgb_gamma,
    "shading_linear_rgb": image_utility.linear_to_srgb_gamma,
    "shading_gamma_rgb": lambda x: x,
    "depth": _depth_to_rgb,
    "alpha": _mono_to_srgb_gamma,
    "analytic_normal": _normal_to_rgb,
    "predicted_normal": _normal_to_rgb,
}


Results = Dict[str, Union[np.ndarray, jnp.ndarray]]


def convert_results_to_rgb(results: Results) -> Results:
  """Convert a results dictionary to gamma-encoded RGB images."""
  rgb_images = {}
  for key in results:
    if key not in _RENDER_RESULT_TO_RGB_FN:
      continue

    # Download to CPU first for efficiency
    result_cpu = np.array(results[key])
    rgb_images[key] = _RENDER_RESULT_TO_RGB_FN[key](result_cpu)

  return rgb_images
