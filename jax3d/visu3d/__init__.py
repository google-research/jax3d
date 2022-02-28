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

"""Visu3d API."""

from __future__ import annotations

import sys

from jax3d.visu3d import lazy_imports
from jax3d.visu3d import plotly
from jax3d.visu3d import typing
from jax3d.visu3d.array_dataclass import array_field
from jax3d.visu3d.array_dataclass import DataclassArray
from jax3d.visu3d.array_dataclass import stack
from jax3d.visu3d.camera import Camera
from jax3d.visu3d.camera_spec import CameraSpec
from jax3d.visu3d.camera_spec import PinholeCamera
from jax3d.visu3d.plotly import make_fig
from jax3d.visu3d.ray import Ray
from jax3d.visu3d.transformation import Transform

# Inside tests, can use `v3d.testing`
if 'pytest' in sys.modules:  # < Ensure open source does not trigger import
  try:
    from jax3d.visu3d import testing  # pylint: disable=g-import-not-at-top
  except ImportError:
    pass

del sys
