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

"""Utility functions."""

from typing import Callable, List, Optional

import chex
import flax
import flax.optim
import jax
import jax.numpy as jnp
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.metrics import metrics
from jax3d.projects.nesf.nerfstatic.nerf import colormap
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import PathLike
from jax3d.projects.nesf.utils.typing import Tree, f32, i32  # pylint: disable=g-multiple-import
import numpy as np
from PIL import Image


# This cannot be a chex.dataclass, b/c this is not supported by msgpack.
@flax.struct.dataclass
class TrainState:
  optimizer: flax.optim.Optimizer


@chex.dataclass
class ReconstructionStats:
  """Reconstruction model statistics for the losses and metrics."""
  coarse_model: types.ReconstructionModelStats
  fine_model: types.ReconstructionModelStats
  regularization: types.LossTerm
  ray_regularization: types.LossTerm

  @property
  def total(self) -> float:
    return (self.coarse_model.total + self.fine_model.total +
            self.regularization.value + self.ray_regularization.value)


@chex.dataclass
class SemanticStats:
  """Semantic Model statistics for the losses and metrics."""
  semantic_model: types.SemanticModelStats
  regularization: types.LossTerm

  @property
  def total(self) -> float:
    semantic_model_total = jnp.array(self.semantic_model.total)
    assert not semantic_model_total.shape, semantic_model_total.shape

    regularization_value = self.regularization.value
    assert not regularization_value.shape, regularization_value.shape

    return semantic_model_total + regularization_value


def render_image(render_fn: Callable[[int, types.Rays], types.RenderResult],
                 rays: types.Rays,
                 normalize_disp,
                 chunk) -> types.RenderedRays:
  """Render all the pixels of an image (in test mode).

  This function ensures that 'rays' is evenly split across all hosts if using
  a multi-host evaluation setup. While each host receives
  the same super batch, only a subset of rays are evaluated by devices attached
  to this host. This only works if 'rays' is identical on each host.

  This function ensures a maximum of 'chunk' rays are processed at one time.
  This upper bounds the memory usage of the underlying devices.

  WARNING: !! HERE THERE BE DRAGONS !!

  The following chunk of code efficiently uses a TPU Pod slice for inference.
  Let 'num_local_devices' be the number of TPU cores attached to this host and
  'num_total_devices' be the total number of TPU cores across all hosts. This
  computation depends on the following assumptions,

    1) Each TPU host has the same value for 'rays'.
    2) render_fn: $INPUT -> $OUTPUT is such that $INPUT is of shape
       [num_local_devices, batch_size_per_device, ...] and $OUTPUT is of
       shape [num_local_devices, num_total_devices, batch_size_per_device].
    3) Each $INPUT[i, j] is operated upon independently of $INPUT[i', j']
       for all choices of (i, j), (i', j'). Communication within the same
       batch, such as via BatchNorm, is not allowed.
    4) $OUTPUT[0] == ... == $OUTPUT[num_local_devices-1].
    5) $OUTPUT fits in host RAM.

  If any of the above are violated, this code will do The Wrong Thing (TM).

  TODO(duckworthd): Consider migrating pmap(all_gather(..)) inside of this
  function. Doing so will ensure that the assumptions above are hidden from
  the caller.

  Args:
    render_fn: function, jit-ed render function.
    rays: a `Rays` namedtuple, the rays to be rendered. The shape of this objext
      *must* be [H W C], i.e. it contains rays for a single image.
    normalize_disp: bool, if true then normalize `disp` to [0, 1].
    chunk: int, the size of chunks to render sequentially.

  Returns:
    A types.RenderedRays object.
  """
  if len(rays.batch_shape) != 2:
    raise ValueError("rays must have exactly rank 3 (i.e. 2 batch dimensions). "
                     "Expected batch shape: [H W] "
                     f"Actual batch shape: {rays.batch_shape}")

  # Transform 'rays' into a bag-of-rays representation.
  height, width = rays.batch_shape
  num_rays = height * width
  rays = jax.tree_map(lambda r: r.reshape((num_rays, -1)), rays)

  process_index = jax.process_index()
  results = []
  for i in range(0, num_rays, chunk):
    # pylint: disable=cell-var-from-loop

    # Split 'rays' into smaller, bite-sized bits that fit in device memory.
    # Result is of shape,
    #   [chunk, ...]
    # Requires Assumption (3): no cross-batch communication.
    chunk_rays = jax.tree_map(lambda r: r[i:i + chunk], rays)
    chunk_size = chunk_rays.origin.shape[0]

    # Pad pseudo-examples at the end of this 'chunk_rays' to ensure that each
    # device is operating on an ndarray of the same shape.
    #
    # Requires Assumption (3): no cross-batch communication.
    rays_remaining = chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining
      chunk_rays = jax.tree_map(
          lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode="edge"), chunk_rays)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # process_count.
    rays_per_process = chunk_rays.origin.shape[0] // jax.process_count()

    # Identify the subset of 'chunk_rays' that this host is responsible for.
    # If each host is processing the same value for 'chunk_rays', then we can
    # reconstruct the full 'render_fn(chunk_rays)' at the end.
    start = process_index * rays_per_process
    stop = start + rays_per_process

    # Reshape chunk_rays to shape,
    #   [num_local_devices, batch_size_per_device, ...]
    # Requires Assumption (1): Each host has same input 'rays'.
    chunk_rays = jax.tree_map(lambda r: shard(r[start:stop]), chunk_rays)

    # Apply inference function. Result is of shape,
    #   [num_local_devices, num_total_devices, batch_size_per_device, ...]
    # Requires Assumption (1): Each host has same input 'rays'.
    # Requires Assumption (2): Input/Output shapes.
    fine = render_fn(chunk_rays)[0].fine

    sigma_grid = fine.sigma_grid

    # Remove sigma_grid from fine to avoid error when reshapping.
    fine.sigma_grid = None

    # Undo sharding operation. We select $OUTPUT[0] for each result as
    #   $OUTPUT[0] == ... == $OUTPUT[num_local_devices-1].
    # After unshard(), result is of shape,
    #   [chunk, ...]
    # Requires Assumption (4): Each device returns the same value.
    # Requires Assumption (5): Output fits in RAM.
    fine = jax.tree_map(lambda x: unshard(x[0], padding=padding), fine)
    results.append(fine)
    # pylint: enable=cell-var-from-loop

  # Merge each list of ndarrays into one ndarray of shape [num_rays, ...].
  # Signature: `tree_multimap(fn, *arrs: RenderedRays) -> RenderedRays`
  # fn has signature `fn(*arrs: Array) -> Array`
  results = jax.tree_map(
      lambda *arrs: jnp.concatenate(arrs, axis=0),
      *results,
  )

  # Reshape each ndarray to [height, width, num_channels].
  results = jax.tree_map(lambda x: x.reshape(height, width, -1), results)

  # Normalize disp for visualization for scenes that aren't scaled in [0, 1].
  if normalize_disp:
    disp = results.disparity
    # Zero disparity is ignored for normalization since those are only produced
    # by masked padding pixels.
    min_disp = jnp.min(disp[jnp.nonzero(disp)])
    max_disp = jnp.max(disp)
    disp = (disp - min_disp) / (max_disp - min_disp)
    results = results.replace(disparity=disp)

  # Sigma_grid is the same for every ray in this batch as the rays belong to
  # a single image from a single scene.
  if sigma_grid is not None:
    # the first 3 axes are batch, height, width which are repeated across all
    # sigma_grids.
    results.sigma_grid = sigma_grid[0, 0, 0]

  return results


# TODO(svora): Merge predict_2d_semantic with render_image when finalized.
def predict_2d_semantic(
    render_pfn: Callable[..., types.RenderResult],
    rng: j3d.RandomState,
    rays: types.Rays,
    nerf_variables: Tree[jnp.ndarray],
    nerf_sigma_grid: f32["1 x y z c"],
    chunk: int) -> types.RenderedRays:
  """Render all the pixels of an image (in test mode) for semantic decoder.

  Args:
    render_pfn: Rendering function. Arguments must be the following,
      rng - jax.random.PRNGKey. Random key.
      rays - types.Rays. Rays to predict semantics for.
      nerf_variables - Tree[jnp.ndarray]. NeRF variables.
      nerf_sigma_grid - f32["1 x y z c"]. NeRF density grid.
    rng: Random number generator.
    rays: a `Rays` object. The rays to be rendered. The shape of this objext
      *must* be [H W C], i.e. it contains rays for a single image.
    nerf_variables: NeRF model variables.
    nerf_sigma_grid: NeRF sigma grid.
    chunk: Maximum number of rays to process at a time.

  Returns:
    A types.RenderedRays object.
  """
  if len(rays.batch_shape) != 2:
    raise ValueError("rays must have exactly rank 3 (i.e. 2 batch dimensions). "
                     "Expected batch shape: [H W] "
                     f"Actual batch shape: {rays.batch_shape}")

  # Transform 'rays' into a bag-of-rays representation.
  height, width = rays.batch_shape
  num_rays = height * width
  rays = jax.tree_map(lambda r: r.reshape((num_rays, -1)), rays)

  process_index = jax.process_index()
  results = []
  for i in range(0, num_rays, chunk):
    # pylint: disable=cell-var-from-loop

    # Split 'rays' into smaller, bite-sized bits that fit in device memory.
    # Result is of shape,
    #   [chunk, ...]
    # Requires Assumption (3): no cross-batch communication.
    chunk_rays = jax.tree_map(lambda r: r[i:i + chunk], rays)
    chunk_size = chunk_rays.origin.shape[0]

    # Pad pseudo-examples at the end of this 'chunk_rays' to ensure that each
    # device is operating on an ndarray of the same shape.
    #
    # Requires Assumption (3): no cross-batch communication.
    rays_remaining = chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining
      chunk_rays = jax.tree_map(
          lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode="edge"), chunk_rays)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # process_count.
    rays_per_process = chunk_rays.origin.shape[0] // jax.process_count()

    # Identify the subset of 'chunk_rays' that this host is responsible for.
    # If each host is processing the same value for 'chunk_rays', then we can
    # reconstruct the full 'render_pfn(chunk_rays)' at the end.
    start = process_index * rays_per_process
    stop = start + rays_per_process

    # Reshape chunk_rays to shape,
    #   [num_local_devices, batch_size_per_device, ...]
    # Requires Assumption (1): Each host has same input 'rays'.
    chunk_rays = jax.tree_map(lambda r: shard(r[start:stop]), chunk_rays)

    # Apply inference function. Result is of shape,
    #   [num_local_devices, num_total_devices, batch_size_per_device, ...]
    # Requires Assumption (1): Each host has same input 'rays'.
    # Requires Assumption (2): Input/Output shapes.
    fine = render_pfn(rng.next(), chunk_rays, nerf_variables, nerf_sigma_grid)

    # Undo sharding operation. We select $OUTPUT[0] for each result as
    #   $OUTPUT[0] == ... == $OUTPUT[num_local_devices-1].
    # After unshard(), result is of shape,
    #   [chunk, ...]
    # Requires Assumption (4): Each device returns the same value.
    # Requires Assumption (5): Output fits in RAM.
    fine = jax.tree_map(lambda x: unshard(x[0], padding=padding), fine)
    results.append(fine)
    # pylint: enable=cell-var-from-loop

  # Merge each list of ndarrays into one ndarray of shape [num_rays, ...].
  # Signature: `tree_multimap(fn, *arrs: RenderedRays) -> RenderedRays`
  # fn has signature `fn(*arrs: Array) -> Array`
  results = jax.tree_map(
      lambda *arrs: jnp.concatenate(arrs, axis=0),
      *results,
  )

  # Reshape each ndarray to [height, width, num_channels].
  results = jax.tree_map(lambda x: x.reshape(height, width, -1), results)

  return results


def predict_3d_semantic(
    render_pfn: Callable[..., f32["d D n k"]],
    rng: j3d.RandomState,
    sample_points: types.SamplePoints,
    nerf_variables: Tree[jnp.ndarray],
    nerf_sigma_grid: f32["1 x y z c"],
    chunk: int,
    num_semantic_classes: int,
    ) -> f32["N k"]:
  """Predict semantic categories for 3D points.

  Args:
    render_pfn: a function resulting from jax.pmap(f), where f returns the
      result of an jax.lax.all_gather(). The arguments to this function must
      be,
        rng - jax.random.PRNGKey. Random key.
        sample_points - types.SamplePoints. Points to predict semantics for.
          batch_shape will be [d n k].
        nerf_variables - Tree[jnp.ndarray]. NeRF variables.
        nerf_sigma_grid - f32["1 x y z c"]. NeRF density grid.
      The return value of this function's dimensions correspond to,
        d - number of local devices
        D - number of total devices
        n  - number of points per device
        k  - number of semantic categories
    rng: Random number generator.
    sample_points: Query points to predict semantic categories for. Batch
      shape must be [N 1].
    nerf_variables: NeRF variables for a single scene.
    nerf_sigma_grid: NeRF sigma grid for a single scene.
    chunk: maximum number of points per call to render_pfn.
    num_semantic_classes: number of expected semantic classes.

  Returns:
    Semantic logits for all points in 'sample_points'.
  """
  if not (len(sample_points.batch_shape) == 2 and
          sample_points.batch_shape[-1] == 1):
    raise ValueError(
        f"sample_points must have exactly rank 3 (i.e. 2 batch dimensions). "
        f"Expected batch shape: [N 1] "
        f"Actual batch shape: {sample_points.batch_shape}")

  # Helper functions for readability.
  def pad_first_dimension(x, n):
    padding = [(0, n)] + [(0, 0)] * (len(x.shape)-1)
    return jnp.pad(x, padding, mode="edge")

  num_points, _ = sample_points.batch_shape

  process_index = jax.process_index()
  results = []
  for i in range(0, num_points, chunk):
    # pylint: disable=cell-var-from-loop

    # Split 'sample_points' into smaller, bite-sized bits that fit in device
    # memory. Result is of shape,
    #   [chunk, ...]
    chunk_sample_points: types.SamplePoints = (
        jax.tree_map(lambda r: r[i:i+chunk], sample_points))
    chunk_size = chunk_sample_points.batch_shape[0]

    # Pad pseudo-examples at the end of this 'chunk_sample_points' to ensure
    # that each device is operating on an ndarray of the same shape.
    sample_points_remaining = chunk_size % jax.device_count()
    if sample_points_remaining != 0:
      padding = jax.device_count() - sample_points_remaining
      chunk_sample_points = jax.tree_map(
          lambda r: pad_first_dimension(r, padding),
          chunk_sample_points)
    else:
      padding = 0
    # After padding the number of chunk_sample_points is always divisible by
    # process_count.
    sample_points_per_process = (
        chunk_sample_points.batch_shape[0] // jax.process_count())

    # Identify the subset of 'chunk_sample_points' that this host is
    # responsible for. If each host is processing the same value for
    # 'chunk_sample_points', then we can reconstruct the full
    # 'render_pfn(chunk_sample_points)' at the end.
    start = process_index * sample_points_per_process
    stop = start + sample_points_per_process

    # Reshape chunk_sample_points to shape,
    #   [num_local_devices, batch_size_per_device, ...]
    chunk_sample_points = jax.tree_map(lambda r: shard(r[start:stop]),
                                       chunk_sample_points)

    # Apply inference function. Result is of shape,
    #   [num_local_devices, num_total_devices, batch_size_per_device, ...]
    predictions = render_pfn(rng.next(), chunk_sample_points,
                             nerf_variables, nerf_sigma_grid)

    # Undo sharding operation. We select $OUTPUT[0] for each result as
    #   $OUTPUT[0] == ... == $OUTPUT[num_local_devices-1].
    # After unshard(), result is of shape,
    #   [chunk, ...]
    predictions = jax.tree_map(lambda x: unshard(x[0], padding=padding),
                               predictions)
    results.append(predictions)
    # pylint: enable=cell-var-from-loop

  # In case there aren't any points to render, return something of a meaningful
  # shape.
  if not results:
    return jnp.zeros((0, num_semantic_classes))

  # Merge each list of ndarray into one ndarray of shape
  # [num_sample_points, ...].
  results = jax.tree_map(lambda *arrs: jnp.concatenate(arrs, axis=0),
                              *results)

  return results


def save_img(img, path: PathLike):
  """Save an image to disk.

  Note clipping and rescaling is only performed if the image type is float.
  Otherwise the image is saved to disk as is.

  Args:
    img: jnp.ndarry, [height, width, channels], float32 images will be clipped
      to [0, 1] before saved to path.
    path: string, path to save the image to.
  """
  with j3d.Path(path).open("wb") as imgout:
    if img.dtype == np.float32:
      img = np.array((np.clip(img, 0., 1.) * 255.).astype(jnp.uint8))
    else:
      img = np.array(img)
    assert np.amax(img) <= 255, f"Saved image has max {np.amax(img)} > 255."
    Image.fromarray(img).save(imgout, format="PNG")


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.

  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.

  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.

  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  t = np.clip(step / max_steps, 0, 1)
  log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
  return delay_rate * log_lerp


def shard(xs):
  """Split data into shards for multiple devices along the first dimension."""
  return jax.tree_map(
      lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)


def to_device(xs):
  """Transfer data to devices (GPU/TPU)."""
  return jax.tree_map(jnp.array, xs)


def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  if padding > 0:
    y = y[:-padding]
  return y


def compute_iou_from_preds(
    logits: Optional[f32["b ... num_classes"]],
    labels: Optional[f32["b ... 1"]],
    num_classes: int) -> types.IOU:
  """Computes the IoU given the predictions and labels.

  Helper function which first computes the activation over the predictions
  before computing the confusion matrix and returning the per class IoU and the
  mean IoU.

  Args:
    logits: nd-array with the predicted model logits.
    labels: nd-array with the ground-truth labels.
    num_classes: number of classes.

  Returns:
    A types.IOU object.
  """
  if num_classes == 0:
    return types.IOU(mean_iou=0, per_class_iou=jnp.zeros((0, 0)))
  # Take argmax over the predictions.
  predictions = jnp.expand_dims(jnp.argmax(logits, axis=-1), axis=-1)
  confusion_matrix = metrics.compute_confusion_matrix(
      y_true=labels, y_pred=predictions, num_classes=num_classes)
  mean_iou, per_class_iou = metrics.compute_iou(confusion_matrix)
  return types.IOU(mean_iou=mean_iou, per_class_iou=per_class_iou)


def compute_conf_mat_from_preds(
    logits: Optional[f32["b ... num_classes"]],
    labels: Optional[f32["b ... 1"]],
    num_classes: int) -> f32["num_classes num_classes"]:
  """Computes the confusion matrix given the predictions and labels.

  Helper function which first computes the activation over the predictions
  before computing the confusion matrix and returning the per class IoU and the
  mean IoU.

  Args:
    logits: nd-array with the predicted model logits.
    labels: nd-array with the ground-truth labels.
    num_classes: number of classes.

  Returns:
    A nd-array containing the confusion matrix.
  """
  if num_classes == 0:
    return types.IOU(mean_iou=0, per_class_iou=jnp.zeros((0, 0)))
  # Take argmax over the predictions.
  predictions = jnp.expand_dims(jnp.argmax(logits, axis=-1), axis=-1)
  return metrics.compute_confusion_matrix(
      y_true=labels, y_pred=predictions, num_classes=num_classes)


def compute_iou_from_con_mat(
    conf_mat: f32["num_classes num_classes"]) -> types.IOU:
  """Computes the IoU given a confusion matrix.

  Args:
    conf_mat: A nd-array confusion matrix.

  Returns:
    A types.IOU object.
  """

  if conf_mat.shape[0] == 0:
    return types.IOU(mean_iou=0, per_class_iou=jnp.zeros((0, 0)))
  mean_iou, per_class_iou = metrics.compute_iou(conf_mat)
  return types.IOU(mean_iou=mean_iou, per_class_iou=per_class_iou)


def fmt_confmat(x: f32["num_classes num_classes"],
                labels: Optional[List[str]] = None) -> str:
  """String format for a confusion matrix.

  Args:
    x: Confusion matrix. x[i, j] contains the number of predictions where the
      true label was labels[i] and prediction was labels[j].
    labels: Optional list of label names. If None, a list of names will be
      generated for you. If labels are provided, there must be one label per
      row/column in the confusion matrix.

  Returns:
    A human-readable string representing the confusion matrix. Each row in the
      printed matrix will sum to 1.0
  """
  n = x.shape[0]
  if x.shape != (n, n):
    raise ValueError(f"Confusion matrix is not square. Shape: {x.shape}")

  if not labels:  # None or empty list
    labels = [f"cls{i:02d}" for i in range(n)]
  if len(labels) != n:
    raise ValueError(
        f"List of labels does not match number of categories in the confusion "
        f"matrix. Found {len(labels)} labels for {n} categories.")
  assert len(labels) == n

  # Normalize along each row
  x = x / np.sum(x, axis=1, keepdims=True)

  # Construct result.
  result = []

  # Print top row with names. Names will be concatenated if they are longer
  # than 6 characters.
  entries = [f"{label:>6.6s}" for label in labels]
  result.append("{:^10.10s}".format(" ") + " | ".join(entries))

  # Print values. Each entry will be of the form "0.xxx". Label names will have
  # up to 10 characters.
  fmt = "{:^10.10s}" + " | ".join(["{:6.3f}"]*n)
  for i in range(n):
    result.append(fmt.format(labels[i], *x[i]))
  return "\n".join(result)


def get_color_coded_semantics_image(
    semantics: i32["b h w"]) -> f32["b h w 3"]:
  """Creates a color coded semantic image using the passed color_mapping.

  Args:
    semantics: Array with semantic values in range [0, num_classes].

  Returns:
    Array with the same shape as semantics with RGB values for each class.
  """
  color_mapping = colormap.get_color_map() / 255.0
  return color_mapping[semantics, :]


def predict_3d_semanticnerf(
    render_pfn: Callable[..., f32["d D n k"]],
    sample_points: types.SamplePoints,
    chunk: int,
    num_semantic_classes: int,
    ) -> f32["N k"]:
  """Predict semantic categories for 3D points.

  Args:
    render_pfn: a function resulting from jax.pmap(f), where f returns the
      result of an jax.lax.all_gather(). The arguments to this function must
      be,
        sample_points - types.SamplePoints. Points to predict semantics for.
          batch_shape will be [d n k].
      The return value of this function's dimensions correspond to,
        d - number of local devices
        D - number of total devices
        n  - number of points per device
        k  - number of semantic categories
    sample_points: Query points to predict semantic categories for. Batch
      shape must be [N 1].
    chunk: maximum number of points per call to render_pfn.
    num_semantic_classes: number of expected semantic classes.

  Returns:
    Semantic logits for all points in 'sample_points'.
  """
  if not (len(sample_points.batch_shape) == 2 and
          sample_points.batch_shape[-1] == 1):
    raise ValueError(
        f"sample_points must have exactly rank 3 (i.e. 2 batch dimensions). "
        f"Expected batch shape: [N 1] "
        f"Actual batch shape: {sample_points.batch_shape}")

  # Helper functions for readability.
  def pad_first_dimension(x, n):
    padding = [(0, n)] + [(0, 0)] * (len(x.shape)-1)
    return jnp.pad(x, padding, mode="edge")

  num_points, _ = sample_points.batch_shape

  process_index = jax.process_index()
  results = []
  for i in range(0, num_points, chunk):
    # pylint: disable=cell-var-from-loop

    # Split 'sample_points' into smaller, bite-sized bits that fit in device
    # memory. Result is of shape,
    #   [chunk, ...]
    chunk_sample_points: types.SamplePoints = (
        jax.tree_map(lambda r: r[i:i+chunk], sample_points))
    chunk_size = chunk_sample_points.batch_shape[0]

    # Pad pseudo-examples at the end of this 'chunk_sample_points' to ensure
    # that each device is operating on an ndarray of the same shape.
    sample_points_remaining = chunk_size % jax.device_count()
    if sample_points_remaining != 0:
      padding = jax.device_count() - sample_points_remaining
      chunk_sample_points = jax.tree_map(
          lambda r: pad_first_dimension(r, padding),
          chunk_sample_points)
    else:
      padding = 0
    # After padding the number of chunk_sample_points is always divisible by
    # process_count.
    sample_points_per_process = (
        chunk_sample_points.batch_shape[0] // jax.process_count())

    # Identify the subset of 'chunk_sample_points' that this host is
    # responsible for. If each host is processing the same value for
    # 'chunk_sample_points', then we can reconstruct the full
    # 'render_pfn(chunk_sample_points)' at the end.
    start = process_index * sample_points_per_process
    stop = start + sample_points_per_process

    # Reshape chunk_sample_points to shape,
    #   [num_local_devices, batch_size_per_device, ...]
    chunk_sample_points = jax.tree_map(lambda r: shard(r[start:stop]),
                                       chunk_sample_points)

    # Apply inference function. Result is of shape,
    #   [num_local_devices, num_total_devices, batch_size_per_device, ...]
    predictions = render_pfn(chunk_sample_points)[1]

    # Undo sharding operation. We select $OUTPUT[0] for each result as
    #   $OUTPUT[0] == ... == $OUTPUT[num_local_devices-1].
    # After unshard(), result is of shape,
    #   [chunk, ...]
    predictions = jax.tree_map(lambda x: unshard(x[0], padding=padding),
                               predictions)
    results.append(predictions)
    # pylint: enable=cell-var-from-loop

  # In case there aren't any points to render, return something of a meaningful
  # shape.
  if not results:
    return jnp.zeros((0, num_semantic_classes))

  # Merge each list of ndarray into one ndarray of shape
  # [num_sample_points, ...].
  results = jax.tree_map(lambda *arrs: jnp.concatenate(arrs, axis=0),
                              *results)

  # TODO(someone): Double check why results has an extra dimension here.
  # Expected [num_sample_points, num_classes]
  # Actual [num_sample_points, 1, num_classes]
  results = jnp.squeeze(results, axis=1)
  return results
