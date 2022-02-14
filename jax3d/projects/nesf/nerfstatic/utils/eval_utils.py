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

"""Helper functions for the eval libraries."""


from typing import Optional

import chex
import jax
from jax import numpy as jnp

from jax3d.projects.nesf.nerfstatic.models import volumetric_semantic_model
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import PRNGKey, Tree, f32  # pylint: disable=g-multiple-import


def predict_fn_3d(
    rng: PRNGKey,
    points: types.SamplePoints,
    nerf_variables: Tree[jnp.ndarray],
    nerf_sigma_grid: f32["1 x y z c"],
    *,
    semantic_variables: Tree[jnp.ndarray],
    semantic_model: volumetric_semantic_model.VolumetricSemanticModel,
) -> f32["D n k"]:
  """Predict semantic logits for a set of 3D points.

  Args:
    rng: jax3d random state.
    points: 3D points to evaluate. Batch size is 'n'.
    nerf_variables: NeRF Model's variables
    nerf_sigma_grid: NeRF sigma grid.
    semantic_variables: Semantic model variables.
    semantic_model: Semantic model for rendering.

  Returns:
    semantic_logits: Array of shape [D, n, k]. Contains logits for
      semantic predictions for each point in 'points' from all devices
      participating in this computation. The return value of this
      function's dimensions correspond to,
        D - number of total devices
        n - number of points per device.
        k - number of semantic classes.
  """
  rng_names = ["params", "sampling", "data_augmentation"]
  rng, *rng_keys = jax.random.split(rng, len(rng_names) + 1)

  # Construct dummy rays to render. The current implementation of
  # VolumetricSemanticModel requires a set of rays to be provided.
  #
  # TODO(duckworthd): Find a way round this extra logic.
  normalize_fn = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
  n = jax.local_device_count() or 8
  dummy_rays = types.Rays(scene_id=jnp.zeros((n, 1), dtype=jnp.int32),
                          origin=jnp.zeros((n, 3)),
                          direction=normalize_fn(jnp.ones((n, 3))))

  _, predictions = semantic_model.apply(
      semantic_variables,
      rngs=dict(zip(rng_names, rng_keys)),
      rays=dummy_rays,
      sigma_grid=nerf_sigma_grid,
      randomized_sampling=True,
      is_train=False,
      nerf_model_weights=nerf_variables,
      points=points)

  assert predictions.shape[1] == 1
  return jax.lax.all_gather(predictions[:, 0, :], axis_name="batch")


def create_sample_points_for_3d_semantics(
    view: types.Views,
    ) -> types.SamplePoints:
  """Construct SamplePoints object for 3D inference."""
  # [n, 3]
  point_cloud = view.point_cloud
  num_points, _ = point_cloud.points.shape
  points = jnp.reshape(point_cloud.points, (num_points, 1, 3))
  dummy_direction = jnp.zeros((num_points, 3))

  return types.SamplePoints(scene_id=point_cloud.scene_id,
                            position=points,
                            direction=dummy_direction)


def markdown(s: str) -> str:
  """Wraps a string into a markdown code block."""
  return f"```\n{s}\n```"


def get_image_key_from_image_ids(image_ids: jnp.ndarray) -> Optional[str]:
  """Get image key from View.image_ids."""
  if image_ids is None:
    return None
  chex.assert_shape(image_ids, ())
  return str(image_ids.flatten()[0])
