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

"""Produces a dataset of the same shape as KLEVR but using fake data.

This code is intended to be a drop-in replacement for datasets/klevr.py.
"""

import dataclasses
from typing import Tuple

import jax
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.datasets import dataset_utils
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.nerfstatic.utils import types
import numpy as np


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class FakeKlevrParams:
  """Fake KLEVR parameters."""

  # Number of semantic classes. Must be >0.
  num_semantic_classes: int = 4

  # Number of images in the full scene.
  num_images: int = 50

  # Number of images in the train split. Must be <= num_images.
  num_train_images: int = 40

  # Height of each image in pixels.
  height: int = 20

  # Width of each image in pixels.
  width: int = 20

  @property
  def num_test_images(self):
    """Number of images in the test set."""
    assert self.num_images >= self.num_train_images
    return self.num_images - self.num_train_images


def make_examples(
    data_dir: j3d.Path,
    *,
    split: str,
    params: FakeKlevrParams,
    **kwargs,
) -> Tuple[types.Batch, dataset_utils.DatasetMetadata]:
  """Generate examples for a single fake scene.

  Args:
    data_dir: Fake path for data. Unused.
    split: Dataset split. One of "train" or "test".
    params: Additional hyperparameters.
    **kwargs: Unused.

  Returns:
    batch: A fake dataset representing a single scene. Values are randomly
      generated but have the proper shape and dtype.
    metadata: ...
  """
  del kwargs
  assert params.num_semantic_classes > 0

  N, H, W = params.num_images, params.height, params.width  # pylint: disable=invalid-name
  scene_id = data_dir.parts[-1]

  batch = types.Batch(
      target_view=types.Views(
          rays=_random_rays((N, H, W)),
          rgb=_random_rgb((N, H, W)),
          depth=_random_depth((N, H, W)),
          semantics=_random_semantics((N, H, W), params.num_semantic_classes),
          image_ids=_fake_image_ids(scene_id, N)),)

  semantic_labels = (
      ['background'] +
      [f'class{i}' for i in range(params.num_semantic_classes - 1)])
  metadata = dataset_utils.DatasetMetadata(labels=semantic_labels,
                                           cameras=None,
                                           scene_name=scene_id)

  if split == 'train':
    batch = _nested_gather(np.arange(0, params.num_train_images), batch)
  elif split == 'test':
    batch = _nested_gather(
        np.arange(params.num_train_images,
                  params.num_train_images + params.num_test_images), batch)
  else:
    raise ValueError(split)

  assert params.num_semantic_classes == len(metadata.labels)
  return batch, metadata


################################################################################
# Internal helper methods


def _random_rays(batch_dims):
  return types.Rays(
      scene_id=None,
      origin=_random_ray_origin(batch_dims),
      direction=_random_ray_direction(batch_dims),
      base_radius=None)


def _random_ray_origin(batch_dims):
  return _random_floats((*batch_dims, 3), bounds=(-1, 1))


def _random_ray_direction(batch_dims):
  d = _random_floats((*batch_dims, 3))
  d = d / np.linalg.norm(d, axis=-1, keepdims=True)
  return d


def _random_rgb(batch_dims):
  return _random_floats((*batch_dims, 3), bounds=(0, 1))


def _random_depth(batch_dims):
  return _random_floats((*batch_dims, 1), bounds=(0, 100))


def _random_semantics(batch_dims, num_semantic_classes):
  return _random_ints((*batch_dims, 1), bounds=(0, num_semantic_classes))


def _fake_image_ids(scene_id, N):  # pylint: disable=invalid-name
  result = [f'{scene_id}_{image_id}' for image_id in range(N)]
  return np.asarray(result)


def _random_floats(shape, bounds=None):
  result = np.random.randn(*shape)
  if bounds:
    lower, upper = bounds
    result = np.asarray(jax.nn.sigmoid(result))  # [0, 1]
    result = result * (upper - lower) - lower  # [lower, upper]
  return result


def _random_ints(shape, bounds=None):
  assert bounds
  lower, upper = bounds
  return np.random.randint(lower, upper, size=shape)


def _nested_gather(indices, values):

  def _maybe_gather(v):
    if v.ndim:
      return v[indices]
    return v  # Don't gather a scalar.

  return jax.tree_map(_maybe_gather, values)


################################################################################
# Dataset registration

dataset_utils.register_dataset(
    dataset_utils.RegisteredDataset(
        name='fake_klevr',
        make_examples_fn=make_examples,
        config_cls=FakeKlevrParams,
    ))
