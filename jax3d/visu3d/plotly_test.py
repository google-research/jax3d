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

"""Tests for plotly."""

from __future__ import annotations

import chex
from etils import enp
from jax3d import visu3d as v3d
from jax3d.visu3d.lazy_imports import plotly_base
from jax3d.visu3d.lazy_imports import plotly_go as go
import numpy as np


@enp.testing.parametrize_xnp()
def test_to_xyz_dict(xnp: enp.NpModule):
  chex.assert_tree_all_close(
      v3d.plotly.to_xyz_dict([
          [0, 1, 2],
          [0, 10, 20],
          [0, 100, 200],
      ]),
      {
          'x': xnp.array([0, 0, 0]),
          'y': xnp.array([1, 10, 100]),
          'z': xnp.array([2, 20, 200]),
      },
  )

  chex.assert_tree_all_close(
      v3d.plotly.to_xyz_dict(
          [
              [0, 1, 2],
              [0, 10, 20],
              [0, 100, 200],
          ],
          pattern='axis_{}',
          names='uvw',
          axis=0,  # pytype: disable=wrong-arg-types
      ),
      {
          'axis_u': xnp.array([0, 1, 2]),
          'axis_v': xnp.array([0, 10, 20]),
          'axis_w': xnp.array([0, 100, 200]),
      },
  )


class VisuObj(v3d.plotly.Visualizable):
  """Test object."""

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    """Construct the traces of the given object."""
    return [
        go.Scatter3d(
            x=[0, 1, 2],
            y=[0, 1, 2],
            z=[0, 1, 2],
        ),
    ]


def test_make_fig():
  x_vig = VisuObj()
  x_trace = go.Scatter3d(
      x=[0, 1, 2],
      y=[0, 1, 2],
      z=[0, 1, 2],
  )
  x_array = np.ones((4, 3))
  fig = v3d.make_fig([
      x_vig,
      x_trace,
      x_array,
  ])
  assert isinstance(fig, go.Figure)
