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

"""Tests for vectorization."""

from etils import enp
from jax3d import visu3d as v3d
from jax3d.visu3d import np_utils
from jax3d.visu3d import vectorization
from jax3d.visu3d.utils import inspect_utils
import pytest

H = 2
W = 3
X0 = 4
X1 = 5

# Activate the fixture
set_tnp = enp.testing.set_tnp


@pytest.mark.parametrize(
    [
        'self_shape',
        'arg_shape',
        'expected_arg_shape',
    ],
    [
        ((H,), (H,), (H,)),
        ((1,), (H,), (H,)),
        ((H,), (1,), (H,)),
        ((H,), (H, X0, X1), (H, X0, X1)),
        ((1,), (H, X0, X1), (H, X0, X1)),
        ((H,), (1, X0, X1), (H, X0, X1)),
        ((H, W), (H, W), (H * W,)),
        ((1, 1), (H, W), (H * W,)),
        ((H, W), (1, 1), (H * W,)),
        ((1, W), (H, 1), (H * W,)),
        ((H, W), (H, W, X0, X1), (H * W, X0, X1)),
        ((1, 1), (H, W, X0, X1), (H * W, X0, X1)),
        ((H, W), (1, 1, X0, X1), (H * W, X0, X1)),
        ((1, W), (H, 1, X0, X1), (H * W, X0, X1)),
    ],
)
@enp.testing.parametrize_xnp()
def test_broadcast_args(
    self_shape: v3d.typing.Shape,
    arg_shape: v3d.typing.Shape,
    expected_arg_shape: v3d.typing.Shape,
    xnp: enp.NpModule,
):

  def fn(self, arg_dc, arg_array):
    assert isinstance(self, v3d.Ray)
    assert isinstance(arg_dc, v3d.Ray)
    assert isinstance(arg_array, xnp.ndarray)
    assert self.shape == ()  # pylint: disable=g-explicit-bool-comparison
    assert arg_dc.shape == expected_arg_shape[1:]
    assert arg_array.shape == expected_arg_shape[1:] + (3,)

  self = v3d.Ray(pos=[0, 0, 0], dir=[0, 0, 0])
  self = self.as_xnp(xnp)
  self = self.broadcast_to(self_shape)

  arg_dc = v3d.Ray(pos=[0, 0, 0], dir=[0, 0, 0])
  arg_dc = arg_dc.as_xnp(xnp)
  arg_dc = arg_dc.broadcast_to(arg_shape)

  arg_array = xnp.zeros(arg_shape + (3,))

  bound_args = inspect_utils.Signature(fn).bind(self, arg_dc, arg_array)
  bound_args, batch_shape = vectorization._broadcast_and_flatten_args(
      bound_args,
      map_non_static=lambda fn, args: args.map(fn),
  )

  assert len(bound_args) == 3
  new_self, new_dc, new_array = bound_args
  new_self = new_self.value
  new_dc = new_dc.value
  new_array = new_array.value

  # Self is flatten
  flat_batch_shape = (np_utils.size_of(batch_shape),)
  assert new_self.shape == flat_batch_shape
  assert expected_arg_shape[:1] == flat_batch_shape

  # Other are broadcasted to a self.flatten compatible size
  assert new_dc.shape == expected_arg_shape
  assert new_array.shape == expected_arg_shape + (3,)


@pytest.mark.parametrize(
    [
        'self_shape',
        'arg_shape',
    ],
    [
        ((H, W), ()),
        ((H, W), (H,)),
        ((H, W), (W,)),
        ((H, W), (H, X0,)),
        ((H, W), (X0, W,)),
    ],
)
@enp.testing.parametrize_xnp()
def test_broadcast_args_failure(
    self_shape: v3d.typing.Shape,
    arg_shape: v3d.typing.Shape,
    xnp: enp.NpModule,
):

  def fn(self, arg):
    del self, arg

  self = v3d.Ray(pos=[0, 0, 0], dir=[0, 0, 0])
  self = self.as_xnp(xnp)
  self = self.broadcast_to(self_shape)

  arg_dc = v3d.Ray(pos=[0, 0, 0], dir=[0, 0, 0])
  arg_dc = arg_dc.as_xnp(xnp)
  arg_dc = arg_dc.broadcast_to(arg_shape)

  bound_args = inspect_utils.Signature(fn).bind(self, arg_dc)

  with pytest.raises(ValueError, match='Cannot vectorize shapes'):
    vectorization._broadcast_and_flatten_args(
        bound_args,
        map_non_static=lambda fn, args: args.map(fn),
    )
