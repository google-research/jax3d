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

"""Metrics."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax3d.projects.nesf.utils.typing import f32, i32  # pylint: disable=g-multiple-import


def compute_psnr(mse):
  """Compute psnr value given mse (we assume the maximum pixel value is 1).

  Args:
    mse: float, mean square error of pixels.

  Returns:
    psnr: float, the psnr value.
  """
  return -10. * jnp.log(mse) / jnp.log(10.)


def compute_ssim(img0,
                 img1,
                 max_val,
                 filter_size=11,
                 filter_sigma=1.5,
                 k1=0.01,
                 k2=0.03,
                 return_map=False):
  """Computes SSIM from two images.

  This function was modeled after tf.image.ssim, and should produce comparable
  output.

  Args:
    img0: array. An image of size [..., width, height, num_channels].
    img1: array. An image of size [..., width, height, num_channels].
    max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
    filter_size: int >= 1. Window size.
    filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
    k1: float > 0. One of the SSIM dampening parameters.
    k2: float > 0. One of the SSIM dampening parameters.
    return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

  Returns:
    Each image's mean SSIM, or a tensor of individual values if `return_map`.
  """
  # Construct a 1D Gaussian blur filter.
  hw = filter_size // 2
  shift = (2 * hw - filter_size + 1) / 2
  f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma)**2
  filt = jnp.exp(-0.5 * f_i)
  filt /= jnp.sum(filt)

  # Blur in x and y (faster than the 2D convolution).
  filt_fn1 = lambda z: jsp.signal.convolve2d(z, filt[:, None], mode="valid")
  filt_fn2 = lambda z: jsp.signal.convolve2d(z, filt[None, :], mode="valid")

  # Vmap the blurs to the tensor size, and then compose them.
  num_dims = len(img0.shape)
  map_axes = tuple(list(range(num_dims - 3)) + [num_dims - 1])
  for d in map_axes:
    filt_fn1 = jax.vmap(filt_fn1, in_axes=d, out_axes=d)
    filt_fn2 = jax.vmap(filt_fn2, in_axes=d, out_axes=d)
  filt_fn = lambda z: filt_fn1(filt_fn2(z))

  mu0 = filt_fn(img0)
  mu1 = filt_fn(img1)
  mu00 = mu0 * mu0
  mu11 = mu1 * mu1
  mu01 = mu0 * mu1
  sigma00 = filt_fn(img0**2) - mu00
  sigma11 = filt_fn(img1**2) - mu11
  sigma01 = filt_fn(img0 * img1) - mu01

  # Clip the variances and covariances to valid values.
  # Variance must be non-negative:
  sigma00 = jnp.maximum(0., sigma00)
  sigma11 = jnp.maximum(0., sigma11)
  sigma01 = jnp.sign(sigma01) * jnp.minimum(
      jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01))

  c1 = (k1 * max_val)**2
  c2 = (k2 * max_val)**2
  numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
  denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
  ssim_map = numer / denom
  ssim = jnp.mean(ssim_map, list(range(num_dims - 3, num_dims)))
  return ssim_map if return_map else ssim


def compute_confusion_matrix(y_true: i32["..."],
                             y_pred: i32["..."],
                             num_classes: int,
                             weights: f32["..."] = None
                             ) -> f32["num_classes num_classes"]:
  """Computes the confusion matrix between y_true and y_pred.

  Args:
    y_true: nd-array; Array of true labels.
    y_pred: nd-array; Array of predicted labels.
    num_classes: int; Number of classes.
    weights: nd-array, Weight of each datapoint (e.g. for masking).

  Returns:
    A [num_classes, num_classes] confusion matrix, normalized by the number of
      elements in y_true/y_pred.
  """
  chex.assert_equal_shape([y_true, y_pred])
  if weights is None:
    weights = jnp.ones_like(y_true)
  else:
    chex.assert_equal_shape([y_true, weights])

  # If weights are all zero, histogram2d returns NaN. To avoid this, set weights
  # to 1 and then set output to zero below:
  weights_all_zero = 1.0 - jnp.any(weights).astype(jnp.float32)
  weights = weights + weights_all_zero

  cm, *_ = jnp.histogram2d(
      y_true.ravel(),
      y_pred.ravel(),
      bins=jnp.arange(num_classes + 1),
      weights=weights.ravel())

  # If weights are all zero, set the confusion matrix to zero:
  cm = cm * (1.0 - weights_all_zero)
  return cm


def compute_iou(confusion_matrix: f32["num_classes num_classes"]
                ) -> Tuple[f32, f32["num_classes"]]:
  """Computes the mean intersection-over-union, given a confusion matrix.

  Args:
    confusion_matrix: confusion matrix with shape [num_classes, num_classes].

  Returns:
    A tuple where the first element is the mean IoU and the second an
      array of num_classes containing the per-class IoU
  """
  sum_over_row = jnp.sum(confusion_matrix, axis=0)
  sum_over_col = jnp.sum(confusion_matrix, axis=1)
  true_positives = jnp.diag(confusion_matrix)

  # sum_over_row + sum_over_col =
  #     2 * true_positives + false_positives + false_negatives.
  denominator = sum_over_row + sum_over_col - true_positives

  # The mean is only computed over classes that appear in the
  # label or prediction tensor. If the denominator is 0, we need to
  # ignore the class.
  iou_per_class = true_positives / denominator
  mean_iou = jnp.nan_to_num(jnp.nanmean(iou_per_class))
  iou_per_class = jnp.nan_to_num(iou_per_class)
  return (mean_iou, iou_per_class)
