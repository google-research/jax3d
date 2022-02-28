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

"""Point utils."""

from __future__ import annotations

import dataclasses

from etils import edc
from etils.array_types import FloatArray, IntArray  # pylint: disable=g-multiple-import
from jax3d.visu3d import array_dataclass
from jax3d.visu3d import plotly
from jax3d.visu3d import transformation
from jax3d.visu3d.lazy_imports import plotly_base
import numpy as np


@edc.dataclass(kw_only=True)
@dataclasses.dataclass(frozen=True)
class Point(array_dataclass.DataclassArray, plotly.Visualizable):
  """3d point cloud.

  Attributes:
    p: Points
    rgb: uint8 color
  """
  p: FloatArray['*shape 3'] = array_dataclass.array_field(shape=(3,))
  rgb: IntArray['*shape 3'] = array_dataclass.array_field(
      shape=(3,),
      dtype=np.uint8,
      default=None,
  )

  # Protocols (inherited)

  def apply_transform(self, tr: transformation.Transform) -> Point:
    # No `color` modification
    return self.replace(p=tr.apply_to_pos(self.p))

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    return plotly.make_points(self.p, rgb=self.rgb)
