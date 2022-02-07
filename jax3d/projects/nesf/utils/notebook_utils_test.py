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

"""Tests for jax3d.projects.nesf.utils.notebook_utils."""

import sys
from unittest import mock

from jax import numpy as jnp
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.utils import notebook_utils
import pytest


def test_display_array_as_image():
  # IPython is only available on Colab, so mock the import
  assert 'IPython' not in sys.modules
  ipython_mock = mock.MagicMock()
  sys.modules['IPython'] = ipython_mock

  j3d.utils.display_array_as_img()
  assert ipython_mock.get_ipython.call_count == 1

  # Cleanup import
  del sys.modules['IPython']


@pytest.mark.parametrize('valid_shape', [
    (28, 28),
    (28, 28, 1),
    (28, 28, 3),
])
def test_array_repr_html_valid(valid_shape):
  # 2D images are displayed as images
  assert '<img' in notebook_utils._array_repr_html(jnp.zeros(valid_shape))


@pytest.mark.parametrize('invalid_shape', [
    (7, 7),
    (28, 7),  # Only one dimension bigger than the threshold
    (28, 28, 4),  # Invalid number of dimension
    (28, 28, 0),
    (2, 28, 28),
])
def test_array_repr_html_invalid(invalid_shape):
  assert notebook_utils._array_repr_html(jnp.zeros(invalid_shape)) is None
