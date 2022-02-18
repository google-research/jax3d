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

"""Test utils."""

from typing import Any

from etils.etree import jax as etree
from etils.etree import Tree
import numpy as np


# TODO(epot): Should use `chex.assert_xyz` once dataclasses support DM `tree`
def assert_trees(assert_fn, x: Tree[Any], y: Tree[Any]) -> None:
  """Compare all values."""
  etree.backend.assert_same_structure(x, y)
  # TODO(epot): Better error messages
  etree.backend.map(assert_fn, x, y)


def assert_allclose(x: Tree[Any], y: Tree[Any]) -> None:
  """Assert the trees are close."""
  assert_trees(np.testing.assert_allclose, x, y)
