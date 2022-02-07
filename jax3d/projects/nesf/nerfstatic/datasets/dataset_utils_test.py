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

"""Tests for dataset_utils."""

from jax3d.projects.nesf.nerfstatic.datasets import dataset_utils
import pytest


def test_prefetch():

  loaded_elems = []

  def _iterator(num_iter):
    for x in range(num_iter):
      loaded_elems.append(x)
      yield x

  ds = dataset_utils.prefetch(_iterator(20), buffer_size=3)
  assert next(ds) == 0
  # First element yield, but 3 elements loaded
  assert loaded_elems == [0, 1, 2]
  assert next(ds) == 1  # Next element loaded
  assert loaded_elems == [0, 1, 2, 3]
  assert next(ds) == 2  # Next element loaded
  assert loaded_elems == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    'iterator', [
        (),  # Empty iterator
        range(1),
        range(2),
        range(3),
        range(4),
        range(40),
    ],
)
def test_prefetch_all_values_yield(iterator):
  all_values = list(iterator)
  ds = dataset_utils.prefetch(iterator, buffer_size=3)
  assert all_values == list(ds)
