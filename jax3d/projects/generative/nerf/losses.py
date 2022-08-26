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

"""Common NeRF loss functions.

Loss functions defined in this module return a scalar weight in addition to the
raw loss value. This enables both logging of the raw values as well as weight
scheduling defined by the caller.
"""

from typing import Callable, Optional, Tuple

from etils.array_types import FloatArray
import gin
import jax
import jax.numpy as jnp


def _get_norm_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Maps the name of a reconstruction norm to a function."""
  if name.lower() == "l1":
    return jnp.abs
  elif name.lower() == "l2":
    return lambda a: a**2
  else:
    raise ValueError(f"Unknown norm function {name}.")


@gin.configurable(allowlist=["low_threshold", "high_threshold"])
def tri_mode_clipping(ground_truth: FloatArray[..., "C"],
                      predicted: FloatArray[..., "C"],
                      low_threshold: float = 0.0,
                      high_threshold: float = 1.0) -> FloatArray[..., "C"]:
  """An error clipping scheme for data saturated outside a given range.

  For ground truth pixels outside this range, predicted pixels only affect the
  loss if they are higher than the low ground truth pixel, or lower than the
  high ground truth pixel. In this case, the predicted values are compared to
  the threshold rather than the pixel value.

  Args:
    ground_truth: The ground truth RGB pixel values to be reconstructed.
    predicted: Estimated RGB pixel values to be evaluated.
    low_threshold: The lower edge of the range.
    high_threshold: The upper end of the range.

  Returns:
    The clipped error value.
  """
  assert low_threshold < high_threshold

  # If groundtruth is above the high limit, only penalize predictions below it.
  gt_high_mask = ground_truth > high_threshold
  gt_high_error = jnp.maximum(high_threshold - predicted,
                              0.0)
  gt_high_error *= gt_high_mask.astype(jnp.float32)

  # If groundtruth is below the low limit only penalize predictions above it.
  gt_low_mask = ground_truth < low_threshold
  gt_low_error = jnp.minimum(low_threshold - predicted, 0.0)
  gt_low_error *= gt_low_mask.astype(jnp.float32)

  # Normal loss for in-range groundtruth.
  in_range_mask = jnp.invert(gt_high_mask ^ gt_low_mask)
  in_range_error = ground_truth - predicted
  in_range_error *= in_range_mask.astype(jnp.float32)

  return gt_high_error + gt_low_error + in_range_error


@gin.configurable(
    "reconstruction_loss", allowlist=["weight", "norm", "use_tri_mode"])
def reconstruction(ground_truth: FloatArray[..., "C"],
                   predicted: FloatArray[..., "C"],
                   mask: Optional[FloatArray[...]] = None,
                   weight: float = 1.0,
                   norm: str = "l2",
                   use_tri_mode: bool = False) -> Tuple[FloatArray, float]:
  """A photometric reconstruction loss.

  Args:
    ground_truth: The ground truth RGB pixel values to be reconstructed.
    predicted: Estimated RGB pixel values to be evaluated.
    mask: Optional per-pixel weights for masking the contribution of certain
      pixels to the loss.
    weight: The scalar weight controlling the strength of the loss.
    norm: Either 'l1' or 'l2' to set the reconstruction norm to be used.
    use_tri_mode: If true, use tri-mode clipping on the error values.

  Returns:
    loss: The scalar loss value with no weight applied.
    weight: The scalar weight controlling the strength of the loss.
  """
  if use_tri_mode:
    error = tri_mode_clipping(ground_truth, predicted)
  else:
    error = ground_truth - predicted

  normed_error = _get_norm_fn(norm)(error)
  if mask is not None:
    normed_error *= mask

  loss = jnp.mean(normed_error)
  return loss, weight


@gin.configurable(
    "normal_consistency_loss",
    allowlist=["weight", "mode", "hold_analytic_normals_constant"])
def normal_consistency(analytic_normals: FloatArray[..., 3],
                       predicted_normals: FloatArray[..., 3],
                       mask: Optional[FloatArray[...]] = None,
                       weight: float = 0.0,
                       hold_analytic_normals_constant: bool = True,
                       mode: str = "error") -> Tuple[FloatArray, float]:
  """Loss for enforcing consistency between predicted and analytic normals.

  Args:
    analytic_normals: Normal vectors derived from a density field.
    predicted_normals: Directly predicted normals to be supervised.
    mask: Optional per-pixel weights for masking the contribution of certain
      pixels to the loss.
    weight: The scalar weight controlling the strength of the loss.
    hold_analytic_normals_constant: If true, treat analytic normals as fixed
      input instead of learnable output by applying stop_gradient.
    mode: Either 'error' or 'cosine' to control whether L2 or cosine
      distance is penalized.

  Returns:
    loss: The scalar loss value with no weight applied.
    weight: The scalar weight controlling the strength of the loss.
  """
  if hold_analytic_normals_constant:
    analytic_normals = jax.lax.stop_gradient(analytic_normals)

  if mode == "error":
    error = (analytic_normals - predicted_normals)**2
  elif mode == "cosine":
    error = 1.0 - jnp.sum(analytic_normals * predicted_normals, axis=-1)
  else:
    raise ValueError(f"Unknown normal consistency loss mode {mode}.")

  if mask is not None:
    error *= mask
  loss = jnp.mean(error)
  return loss, weight


@gin.configurable("color_correction_regularization", allowlist=["weight"])
def color_correction_regularization(error: FloatArray[...],
                                    weight: float = 0.0
                                   ) -> Tuple[FloatArray, float]:
  """Color correction regularization.

  Args:
    error: Color correction error values to be penalized.
    weight: The scalar weight controlling the strength of the loss.

  Returns:
    loss: The scalar value with no weight applied.
    weight: The scalar weight controlling the strength of the loss.
  """
  return jnp.mean(error), weight


@gin.configurable("hard_surface_loss", allowlist=["weight"])
def hard_surface(sample_weights: FloatArray[...],
                 weight: float = 0.0) -> Tuple[FloatArray, float]:
  """Hard surface density regularizer loss.

  Args:
    sample_weights: Per-sample contribution weights from volume rendering.
    weight: The scalar weight controlling the strength of the loss.

  Returns:
    loss: The scalar loss value with no weight applied.
    weight: The scalar weight controlling the strength of the loss.
  """
  loss = jnp.mean(-jnp.log(
      jnp.exp(-jnp.abs(sample_weights)) +
      jnp.exp(-jnp.abs(1.0 - sample_weights))))
  return loss, weight
