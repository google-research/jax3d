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

"""Tests for jax3d.projects.nesf.nerfstatic.datasets.klevr."""

from typing import List

import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.datasets import klevr
from jax3d.projects.nesf.nerfstatic.utils import types

_LABELS = ['background', 'Cube', 'Cylinder', 'Sphere', 'Torus', 'Gear']


def test_make_examples():
  data_path = j3d.nf_dir() / 'datasets/test_data/klevr/0'

  exs, metadata = klevr.make_examples(data_path, split='train',
                                      enable_sqrt2_buffer=False,
                                      image_idxs=(0, 1))
  assert isinstance(exs, types.Batch)
  assert metadata.labels == _LABELS
  assert j3d.types_like(exs) == types.Batch.as_types(
      target_batch_shape=(2, 256, 256), scene_id=False, image_id=True)
  assert _get_image_ids(exs) == [0, 0]

  exs, metadata = klevr.make_examples(data_path, split='test',
                                      enable_sqrt2_buffer=False,
                                      image_idxs=None)
  assert metadata.labels == _LABELS
  assert j3d.types_like(exs) == types.Batch.as_types(
      target_batch_shape=(1, 256, 256), scene_id=False, image_id=True)
  assert _get_image_ids(exs) == [0]


def _get_image_ids(exs: types.Batch) -> List[int]:
  """Returns the image ids from the examples."""
  return [s[0, 0, 0] for s in exs.target_view.semantics]
