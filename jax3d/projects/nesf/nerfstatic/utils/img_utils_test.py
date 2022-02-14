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

"""Tests for jax3d.projects.nesf.nerfstatic.utils.img_utils."""

import pathlib

from jax3d.projects.nesf.nerfstatic.utils import img_utils
import numpy as np
import pytest


@pytest.mark.parametrize(
    'fname, dtype, arr',
    [
        ('a.tif', np.float32, [[0, 1e+10, 1e-10], [125.5, 0.00123, 10.01]]),
        ('a.tiff', np.int32, [[0, 1, 2], [10_000_000, 256, 60000]]),
        ('a.png', np.uint8, [[0, 1, 2], [0, 255, 200]]),
        ('a.png', np.uint8, [  # 1-channels
            [[0], [2]],
            [[255], [200]],
        ]),
        ('a.png', np.uint8, [  # 3-channels
            [[0, 1, 2], [0, 1, 2]],
            [[0, 255, 200], [0, 255, 200]],
        ]),
        ('a.png', np.uint8, [  # 4-channels
            [[0, 1, 2, 100], [0, 1, 2, 200]],
            [[0, 255, 200, 200], [0, 255, 200, 200]],
        ]),
        ('a.png', np.uint16, [[0, 1, 2], [256, 255, 20_000]]),
        ('a.png', np.uint16, [[0, 1, 2], [5, 5, 5]]),
    ]
)
def test_imread_imwrite(
    tmp_path: pathlib.Path,
    fname: str,
    dtype,
    arr,
):
  tmp_path = tmp_path / fname
  arr = np.array(arr, dtype=dtype)

  if arr.ndim == 2:  # Single channels images are loaded as (h, w, 1)
    arr = arr[..., None]

  img_utils.imwrite(tmp_path, arr)
  new_arr = img_utils.imread(tmp_path)
  assert np.allclose(arr, new_arr)
  assert new_arr.dtype == dtype


def test_imwrite_bad(tmp_path: pathlib.Path):
  arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
  with pytest.raises(ValueError, match='dtype should be uint8/uint16'):
    img_utils.imwrite(tmp_path / 'x.png', arr)

  arr = np.zeros((2, 2, 3), dtype=np.uint16)
  with pytest.raises(ValueError, match='single-channel'):
    img_utils.imwrite(tmp_path / 'x.png', arr)
