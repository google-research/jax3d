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

"""Tests for fake_klevr."""

import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.datasets import fake_klevr
from jax3d.projects.nesf.nerfstatic.utils import types


def test_make_examples():
  data_path = j3d.Path('/fake/123')
  params = fake_klevr.FakeKlevrParams()

  exs, _ = fake_klevr.make_examples(data_path, split='train', params=params)
  assert isinstance(exs, types.Batch)
  assert j3d.types_like(exs) == types.Batch.as_types(
      target_batch_shape=(params.num_train_images, params.height, params.width),
      scene_id=False,
      image_id=True)
  assert exs.target_view.image_ids[0] == '123_0'

  exs, _ = fake_klevr.make_examples(data_path, split='test', params=params)
  assert j3d.types_like(exs) == types.Batch.as_types(
      target_batch_shape=(params.num_test_images, params.height, params.width),
      scene_id=False,
      image_id=True)
  assert exs.target_view.image_ids[0] == f'123_{params.num_train_images}'
