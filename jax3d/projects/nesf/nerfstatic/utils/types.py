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

"""Type definitions for Nerfstatic."""

import enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import chex
import flax
from flax import linen as nn
import jax.numpy as jnp
from jax3d.projects.nesf.utils.typing import Array, StrArray, f32, i32  # pylint: disable=g-multiple-import
import numpy as np

# Nested dictionary of additional data
AuxiliaryDict = Dict[str, Union[Array[...], 'AuxiliaryDict']]  # pytype: disable=not-supported-yet


@chex.dataclass
class LabeledPointCloud:
  """A semantic-labeled 3D point cloud.

  Contains 'n' points, each of which has a 3D location and a semantic class.

  """
  scene_id: i32['n 1']
  points: f32['n 3']
  semantics: i32['n 1']

  @property
  def size(self):
    """Number of points in the point cloud."""
    chex.assert_rank(self.points, 2)
    chex.assert_rank(self.semantics, 2)
    chex.assert_equal_shape_prefix([self.points, self.semantics], 1)
    chex.assert_shape(self.points, (..., 3))
    chex.assert_shape(self.semantics, (..., 1))
    return self.points.shape[0]


@chex.dataclass
class Rays:
  """Rays through a scene.

  Leading dimension (rays.batch_shape) can be:

  * `num_devices batch_size_per_device`: Before broadcasting to `pmap`
  * `batch_size`: Input of the model
  * `height width`: To render an image during eval

  """
  # The id of the current scene.
  # We keep a channnel dimension to allow for simpler transformations of the
  # whole dataclass.
  scene_id: Optional[i32['... 1']]
  origin: f32['... 3']  # The origin of the ray.
  direction: f32['... 3']  # The normalized direction of the ray.
  base_radius: Optional[f32['... 1']] = None  # Used for Mip-Nerf.

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Returns the leading dimensions of the rays."""
    return self.origin.shape[:-1]


@chex.dataclass
class Views:
  """Views for a scene.

  Leading dimension (views.batch_shape) can be:

  * `num_devices batch_size_per_device`: Before broadcasting to `pmap`
  * `batch_size`: Input of the model
  * `height width`: Spatial image dimensions for input views or for evaluation

  """
  rays: Rays
  # Depth can either be a dummy value (0) or the ground truth depth.
  depth: f32['... 1']
  rgb: Optional[f32['... 3']] = None
  semantics: Optional[i32['... 1']] = None
  # Image ID matching the on-disk image identifier
  # * In RAY mode: Is `None`
  # * In IMAGE mode: Is a singleton scalar str (so `rgb.shape=(h, w, c)` but
  #   `image_ids.shape=()`)
  image_ids: Optional[StrArray['...']] = None
  semantic_mask: Optional[i32['... 1']] = None

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Returns the leading dimensions of the batch."""
    return self.rays.batch_shape

  @property
  def point_cloud(self) -> LabeledPointCloud:
    """Semantic labeled point cloud derived from depth.

    Only includes points within a [-1, 1]^3 bounding box.

    """
    if self.semantics is None:
      raise ValueError('semantics must be defined to construct labeled '
                       'point cloud.')
    ray_o = np.reshape(self.rays.origin, (-1, 3))
    ray_d = np.reshape(self.rays.direction, (-1, 3))
    depth = np.reshape(self.depth, (-1, 1))

    points = ray_o + depth * ray_d
    mask = np.all((points >= -1) & (points <= 1), axis=-1)

    select_points = points[mask]

    semantics = np.reshape(self.semantics, (-1, 1))
    select_semantics = semantics[mask]

    scene_id = np.reshape(self.rays.scene_id, (-1, 1))
    select_scene_id = scene_id[mask]

    return LabeledPointCloud(scene_id=select_scene_id,
                             points=select_points,
                             semantics=select_semantics)


@chex.dataclass
class Batch:
  """Single example batch.

  Leading dimension can be:

  * `num_devices batch_size_per_device`: Before broadcasting to `pmap`
  * `batch_size`: Input of the model
  * `height width`: To render an image during eval

  """
  target_view: Views

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Returns the leading dimensions of the target batch."""
    return self.target_view.batch_shape

  @classmethod
  def as_types(
      cls,
      target_batch_shape: Tuple[int, ...],
      scene_id: bool = True,
      image_id: bool = False,
      semantic_mask: bool = False,
      enable_base_radii: bool = False,
  ) -> 'Batch':
    """`Batch` factory returning a batch containing `j3d.typing` types.

    ```
    assert Batch.as_types(target_batch_shape=(2,)) == Batch(rgb=f32[2, 3], ...)
    ```

    Args:
      target_batch_shape: The inner shape to add to the target view.
      scene_id: Wether `batch.rays.scene_id` is set or `None`.
      image_id: Whether `batch.image_id` is set or `None`.
      semantic_mask: Whether 'batch.semantic_mask' is set or `None`.
      enable_base_radii: Whether to compute base radii, e.g. for MipNerf.

    Returns:
      batch: The `Batch` containing
    """
    base_radius = f32[(*target_batch_shape, 1)] if enable_base_radii else None
    placeholder_target_views = Views(
        rays=Rays(
            scene_id=i32[(*target_batch_shape, 1)] if scene_id else None,
            origin=f32[(*target_batch_shape, 3)],
            direction=f32[(*target_batch_shape, 3)],
            base_radius=base_radius,
        ),
        depth=f32[(*target_batch_shape, 1)],
        rgb=f32[(*target_batch_shape, 3)],
        semantics=i32[(*target_batch_shape, 1)],
        image_ids=StrArray[tuple(target_batch_shape[:-2])]
        if image_id else None,
        semantic_mask=i32[(*target_batch_shape, 1)] if semantic_mask else None,
    )
    return cls(
        target_view=placeholder_target_views,
    )

  def pop_image_id_stateless(self) -> Tuple['Batch', Optional[i32['... 1']]]:
    """Returns a copy of the batch without the image_ids."""
    image_ids = self.target_view.image_ids
    batch = self.replace(
        target_view=self.target_view.replace(image_ids=None),
    )
    return batch, image_ids


@chex.dataclass
class SamplePoints:
  """Sample points along a ray inside the NeRF renderer."""
  scene_id: f32['... 1']
  position: f32['... num_samples 3']
  direction: f32['... 3']
  covariance: Optional[f32['... 3']] = None  # Covariance for Mip-NeRF.

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Returns the leading dimensions of the sample points."""
    return self.position.shape[:-1]


@chex.dataclass
class SampleResults:
  """Values along the ray inside the NeRF Renderer."""
  rgb: f32['b num_samples 3']
  sigma: f32['b num_samples 1']
  semantic: f32['b num_samples num_classes']
  sigma_penultimate_embeddings: f32['b num_samples num_embeddings']
  sigma_grid: Optional[f32['... num_features']] = None
  # aux dict, which contains selected activations for debugging.
  # If those are not used, XLA optimization will ensure they cause no overhead.
  aux: Optional[AuxiliaryDict] = None


SampleStoreFn = Callable[[
    # Query sample points.
    SamplePoints,
    # deterministic: Whether to run the model in deterministic mode or not.
    # Used for modules such as dropout whose behavior varies between training
    # and evaluation.
    bool,
],
                         # Output: sample results.
                         SampleResults]

SemanticSampleStoreFn = Callable[[
    # Query sample points.
    SamplePoints,
    # sigma grid: Dense grid containing the sigma predictions.
    f32['...'],
    # sigma_penultimate_features: Penaltimate feature activations from the sigma
    # MLP.
    f32['...']
],
                                 # Output: semantic predictions.
                                 f32['...'],]


@chex.dataclass
class RenderedRays:
  """Rendered rays.

  Leading dimension can be:
  * `batch_size`: Output of the model.
  * `batch_size height width`: Output of a rendered image during eval.
  """
  rgb: f32['... 3']
  foreground_rgb: f32['... 3']
  disparity: f32['... 1']
  opacity: f32['... 1']
  contribution: Optional[f32['... num_samples']]
  semantic: f32['... num_classes']
  foreground_semantic: f32['... num_classes']
  # aux dict, which contains selected activations for debugging.
  # If those are not used, XLA optimization will ensure they cause no overhead.
  aux: Optional[Dict[str, Any]] = None
  sigma_grid: Optional[f32['...']] = None
  # Optional ray history. Used in mip-nerf-360 for computing the ray
  # regularization.
  ray_z_vals: Optional[f32['...']] = None


@chex.dataclass
class RenderResult:
  """Similar to the above, but with separate coarse and fine stage."""
  coarse: Optional[RenderedRays]
  fine: RenderedRays


@chex.dataclass
class InitializedModel:
  """Initialized model ready for train/eval."""

  # flax.linen Module representing a model.
  model: nn.Module

  # Model's variables. First argument when using model.__call__().
  variables: flax.core.scope.FrozenVariableDict


@chex.dataclass
class LossTerm:
  """A single loss object with a multiplier."""

  # Loss. May include be a non-scalar array when used within vmap.
  loss: f32['...']

  # Weight multiplier associated with this loss term.
  weight: f32['...']

  @property
  def value(self) -> float:
    result = jnp.mean(self.weight * self.loss)
    assert not result.shape, result.shape
    return result


@chex.dataclass
class ReconstructionModelStats():
  """Loss and stats for reconstruction based models."""
  reconstruction_loss: LossTerm
  psnr: float
  semantic_loss: Optional[LossTerm] = None
  mean_iou: Optional[float] = None

  @property
  def total(self) -> float:
    if self.semantic_loss is None:
      semantic_loss = 0
    else:
      semantic_loss = self.semantic_loss.value
    return self.reconstruction_loss.value + semantic_loss


@chex.dataclass
class SemanticModelStats():
  """Loss and stats for semantic based models."""
  semantic_loss: LossTerm
  smoothness_regularization_loss: LossTerm
  mean_iou: float
  percent_matching_scene_ids: float

  @property
  def total(self) -> float:
    value1 = jnp.array(self.semantic_loss.value)
    value2 = jnp.array(self.smoothness_regularization_loss.value)
    assert (not value1.shape) and (not value2.shape)
    return value1 + value2


@chex.dataclass
class IOU:
  """IOU metrics."""
  mean_iou: f32['']
  per_class_iou: f32['num_classes']


@chex.dataclass
class MlpOutputs:
  """Outputs of the MLP model."""
  predictions: f32['... num_outputs']  # Network predictions.
  penultimate_features: f32['...']  # Penultimate features.


@chex.dataclass
class BoundingBox3d:
  """3D axis aligned bounding box."""
  min_corner: f32[3]
  max_corner: f32[3]

  @property
  def size(self):
    return self.max_corner - self.min_corner

  def intersect_rays(self, rays: Rays) -> Tuple[f32['...'], f32['...']]:
    """Calculates the intersection of rays with this bbox.

    This method will return the near and far intersection of the rays with the
    bounding box. Those near and far intersection values are multipliers of
    rays.direction. E.g. the ray intersects the bbox at:
      rays.origin + rays.direction * (near|far)_intersection
    Note that both near and far intersections can be negative (if
    the intersection is "behind" the origin).
    If the ray does not intersect the bbox, far will be smaller then near.
    If the ray only touches the bbox, near == far.

    Args:
      rays: The rays to intersect the bbox with.

    Returns:
      Tuple[near_intersection, far_intersection]. Both values are of shape
        rays.origins.shape[:-1].
    """
    corners = jnp.stack([self.min_corner, self.max_corner], axis=-1)
    corners -= rays.origin[..., None]
    intersections = corners / rays.direction[..., None]
    min_intersections = jnp.amax(jnp.amin(intersections, axis=-1), axis=-1)
    max_intersections = jnp.amin(jnp.amax(intersections, axis=-1), axis=-1)
    return min_intersections, max_intersections


class BackgroundType(enum.Enum):
  NONE = 'NONE'
  WHITE = 'WHITE'
  GREY = 'GREY'


class InterpolationType(enum.Enum):
  TRILINEAR = enum.auto()
