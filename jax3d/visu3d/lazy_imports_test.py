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

"""Tests for lazy_imports."""

import sys

from jax3d import visu3d as v3d


def test_lazy():
  assert 'plotly' not in sys.modules
  go = v3d.lazy_imports.plotly_go
  plotly_base = v3d.lazy_imports.plotly_base
  assert isinstance(go.Scatter3d(), plotly_base.BaseTraceType)
