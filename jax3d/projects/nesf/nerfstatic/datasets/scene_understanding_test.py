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

"""Tests for scene_understanding."""

# TODO(noharadwan): Decide if we wanna keep or remove before code release.

import jax
from jax3d.projects.nesf.nerfstatic import datasets
import numpy as np
import sunds
from sunds.conftest import lego_builder  # pylint: disable=unused-import


def test_sunds_ray(lego_builder: sunds.core.DatasetBuilder):  # pylint: disable=redefined-outer-name
  args = datasets.DatasetParams(
      dataset='sunds',
      data_dir=lego_builder.data_dir,
      specific_params=datasets.scene_understanding.SundsParams(
          name='nerf_synthetic/lego',
      ),
      ray_shuffle_buffer_size=13,
  )

  ds = datasets.get_dataset(
      split='train',
      args=args,
      model_args=None,
      example_type=datasets.ExampleType.RAY,
  )

  # Check that the shape is fully defined
  _, dummy_ex = ds.peek()
  assert dummy_ex.target_view.batch_shape == (
      jax.local_device_count(), args.batch_size.per_device)

  _, ex = next(iter(ds))
  assert ex.target_view.batch_shape == (
      jax.local_device_count(), args.batch_size.per_device)
  assert ex.target_view.image_ids is None


def test_sunds_image(lego_builder: sunds.core.DatasetBuilder):  # pylint: disable=redefined-outer-name
  args = datasets.DatasetParams(
      dataset='sunds',
      data_dir=lego_builder.data_dir,
      specific_params=datasets.scene_understanding.SundsParams(
          name='nerf_synthetic/lego',
      ),
  )

  ds = datasets.get_dataset(
      split='train',
      args=args,
      model_args=None,
      example_type=datasets.ExampleType.IMAGE,
  )

  _, ex = next(iter(ds))
  assert ex.target_view.batch_shape == (800, 800)
  assert ex.target_view.image_ids.shape == ()  # pylint: disable=g-explicit-bool-comparison
  assert ex.target_view.image_ids == 'lego-lego_train_frame0000-default_camera'


def test_sunds_scene_id(lego_builder: sunds.core.DatasetBuilder):  # pylint: disable=redefined-outer-name
  args = datasets.DatasetParams(
      dataset='sunds',
      data_dir=lego_builder.data_dir,
      specific_params=datasets.scene_understanding.SundsParams(
          name='nerf_synthetic/lego',),
      ray_shuffle_buffer_size=13,
  )

  ds = datasets.get_dataset(
      split='train',
      args=args,
      model_args=None,
      example_type=datasets.ExampleType.RAY,
  )

  _, ex = next(iter(ds))
  np.testing.assert_allclose(ex.target_view.rays.scene_id,
                             np.zeros((1, 4096, 1), dtype=np.int32))

  # TODO(duckworthd): Write a variant of this test for a dataset where
  # _SceneIdMapping.lookup() emits non-trivial values.
