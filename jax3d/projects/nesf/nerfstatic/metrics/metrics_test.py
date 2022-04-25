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

"""Unit tests for metrics."""
import functools

from absl.testing import absltest
import jax
from jax import random
from jax3d.projects.nesf.nerfstatic.metrics import metrics
import numpy as np
import pandas as pd
import tensorflow as tf


class MetricsTest(absltest.TestCase):

  def test_ssim_golden(self):
    """Test our SSIM implementation against the Tensorflow version."""
    rng = random.PRNGKey(0)
    shape = (2, 12, 12, 3)
    for _ in range(4):
      rng, key = random.split(rng)
      max_val = random.uniform(key, minval=0.1, maxval=3.)
      rng, key = random.split(rng)
      img0 = max_val * random.uniform(key, shape=shape, minval=-1, maxval=1)
      rng, key = random.split(rng)
      img1 = max_val * random.uniform(key, shape=shape, minval=-1, maxval=1)
      rng, key = random.split(rng)
      filter_size = random.randint(key, shape=(), minval=1, maxval=10)
      rng, key = random.split(rng)
      filter_sigma = random.uniform(key, shape=(), minval=0.1, maxval=10.)
      rng, key = random.split(rng)
      k1 = random.uniform(key, shape=(), minval=0.001, maxval=0.1)
      rng, key = random.split(rng)
      k2 = random.uniform(key, shape=(), minval=0.001, maxval=0.1)

      ssim_gt = tf.image.ssim(
          img0,
          img1,
          max_val,
          filter_size=filter_size,
          filter_sigma=filter_sigma,
          k1=k1,
          k2=k2).numpy()
      for return_map in [False, True]:
        ssim_fn = jax.jit(
            functools.partial(
                metrics.compute_ssim,
                max_val=max_val,
                filter_size=filter_size,
                filter_sigma=filter_sigma,
                k1=k1,
                k2=k2,
                return_map=return_map))
        ssim = ssim_fn(img0, img1)
        if not return_map:
          np.testing.assert_allclose(ssim, ssim_gt, atol=1E-5, rtol=1E-5)
        else:
          np.testing.assert_allclose(
              np.mean(ssim, [1, 2, 3]), ssim_gt, atol=1E-5, rtol=1E-5)
        self.assertLessEqual(np.max(ssim), 1.)
        self.assertGreaterEqual(np.min(ssim), -1.)

  def test_ssim_lowerbound(self):
    """Test the unusual corner case where SSIM is -1."""
    sz = 11
    img = np.meshgrid(*([np.linspace(-1, 1, sz)] * 2))[0][None, ..., None]
    eps = 1e-5
    ssim = metrics.compute_ssim(
        img, -img, 1., filter_size=sz, filter_sigma=1.5, k1=eps, k2=eps)
    np.testing.assert_allclose(ssim, -np.ones_like(ssim), atol=1E-5, rtol=1E-5)

  def test_compute_confusion_matrix(self):
    labels = np.array(
        [2, 0, 2, 2, 0, 1, 1])
    predictions = np.array(
        [0, 0, 2, 1, 0, 2, 1])
    predicted_conf_mat = metrics.compute_confusion_matrix(
        y_true=labels, y_pred=predictions, num_classes=3)

    # Confusion matrix:
    # Pred / True:   0    1    2
    #
    # 0              2    0    1
    # 1              0    1    1
    # 2              0    1    1

    expected_conf_mat = pd.crosstab(labels, predictions)
    assert np.allclose(predicted_conf_mat, expected_conf_mat)

  def test_compute_weighted_confusion_matrix(self):
    labels = np.array(
        [2, 0, 2, 1])
    predictions = np.array(
        [0, 0, 2, 1])
    weights = np.array(
        [0.5, 1, 1, 0.3])
    predicted_conf_mat = metrics.compute_confusion_matrix(
        y_true=labels, y_pred=predictions, num_classes=3, weights=weights)

    # Confusion matrix:
    # Pred / True:   0    1    2
    #
    # 0              1    0    0.5
    # 1              0   0.3    0
    # 2              0    0     1

    expected_conf_mat = pd.crosstab(
        labels, predictions, weights, aggfunc=sum).fillna(0)
    assert np.allclose(predicted_conf_mat, expected_conf_mat)

  def test_compute_iou(self):
    conf_mat = np.array([[2, 1, 1], [1, 3, 0], [0, 1, 1]])
    pred_mean_iou, pred_per_class_iou = metrics.compute_iou(conf_mat)
    expected_per_class_iou = [2/5., 3/6., 1/3.]
    expected_mean_iou = np.mean(expected_per_class_iou)
    assert np.allclose(pred_per_class_iou, expected_per_class_iou)
    assert np.isclose(pred_mean_iou, expected_mean_iou)


if __name__ == '__main__':
  absltest.main()
