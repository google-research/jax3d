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

"""Tests for point."""

from etils import enp
from jax3d import visu3d as v3d
import pytest

# Activate the fixture
set_tnp = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('shape', [(), (2,), (2, 3)])
@pytest.mark.parametrize('with_color', [False, True])
def test_point(
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
    with_color: bool,
):
  if with_color:
    rgb_kwargs = {'rgb': xnp.ones(shape=shape + (3,))}
  else:
    rgb_kwargs = {}
  p = v3d.Point(p=xnp.ones(shape=shape + (3,)), **rgb_kwargs)

  tr = v3d.Transform(R=xnp.eye(3), t=xnp.zeros((3,)))
  p2 = tr @ p
  v3d.testing.assert_array_equal(p, p2)

  # Display should works
  _ = p.fig
