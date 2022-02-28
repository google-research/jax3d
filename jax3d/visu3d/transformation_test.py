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

"""Tests for transformation."""

import dataclasses

from etils import enp
from etils.array_types import FloatArray
from jax3d import visu3d as v3d
from jax3d.visu3d import testing
import numpy as np
import pytest

# Activate the fixture
set_tnp = enp.testing.set_tnp


@dataclasses.dataclass
class TransformExpectedValue:
  """Tests values."""
  # Transformation params
  R: FloatArray[3, 3]  # pylint: disable=invalid-name
  t: FloatArray[3]

  # Expected rays values after transformation
  expected_pos: FloatArray[..., 3]
  expected_dir: FloatArray[..., 3]

  # Expected transformation values after composition with other tr
  expected_r: FloatArray[..., 3, 3]
  expected_t: FloatArray[..., 3]


# Transformation values
_RAY_POS = np.array([1, 3, 5])
_RAY_DIR = np.array([2, 1, 4])
_TR_R = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
])
_TR_T = np.array([4, 3, 7])

_TR_EXPECTED_VALUES = [
    # Test identity
    TransformExpectedValue(
        R=np.eye(3),
        t=np.zeros((3,)),
        # Identity should be a no-op
        expected_pos=_RAY_POS,
        expected_dir=_RAY_DIR,
        expected_r=_TR_R,
        expected_t=_TR_T,
    ),
    # Test translation only
    TransformExpectedValue(
        R=np.eye(3),
        t=[3, -1, 2],
        # Only position translated
        expected_pos=_RAY_POS + [3, -1, 2],
        expected_dir=_RAY_DIR,
        expected_r=_TR_R,
        expected_t=_TR_T + [3, -1, 2],
    ),
    # Test rotation only
    TransformExpectedValue(
        R=[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        t=[0, 0, 0],
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        expected_pos=[5, 1, 3],
        expected_dir=[4, 2, 1],
        # Rotation invert axis `(1, 2, 3)` -> `(2, 1, 3)`
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        expected_r=[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        expected_t=[7, 4, 3],
    ),
    # Test translation + rotation
    TransformExpectedValue(
        R=[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        t=[3, -1, 2],
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        # And translation applied
        expected_pos=np.array([5, 1, 3]) + [3, -1, 2],
        expected_dir=[4, 2, 1],
        # Rotation invert axis `(1, 2, 3)` -> `(2, 1, 3)`
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        expected_r=[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        expected_t=np.array([7, 4, 3]) + [3, -1, 2],
    ),
]


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('shape', [(), (4, 3)])
@pytest.mark.parametrize('test_values', _TR_EXPECTED_VALUES)
def test_transformation(
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
    test_values: TransformExpectedValue,
):
  tr = v3d.Transform(R=xnp.array(test_values.R), t=xnp.array(test_values.t))

  _assert_tr_common(tr)

  _assert_ray_transformed(tr, shape=shape, test_values=test_values)

  _assert_point_transformed(tr, shape=shape, test_values=test_values)

  _assert_tr_transformed(tr, test_values=test_values)


def _assert_tr_common(tr: v3d.Transform):
  """Generic rules applied to all transformation."""

  identity_tr = v3d.Transform(R=tr.xnp.eye(3), t=tr.xnp.zeros((3,)))

  # Inverting the matrix is equivalent to the matrix of the invert transform
  _assert_equal(tr.inv.matrix4x4, enp.compat.inv(tr.matrix4x4))

  # Inverting twice the transformation should be a no-op
  _assert_equal(tr.inv.inv, tr)

  # Composing the transformation with the inverse should be identity
  _assert_equal(tr.inv @ tr, identity_tr)
  _assert_equal(tr @ tr.inv, identity_tr)

  # Exporting/importing matrix from 4x4 should be a no-op
  _assert_equal(v3d.Transform.from_matrix(tr.matrix4x4), tr)

  # Figure should work
  _ = tr.fig


def _assert_ray_transformed(
    tr: v3d.Transform,
    shape: v3d.typing.Shape,
    test_values: TransformExpectedValue,
):
  """Test ray transformation."""
  xnp = tr.xnp
  ray = v3d.Ray(
      pos=xnp.array(_RAY_POS),
      dir=xnp.array(_RAY_DIR),
  )
  expected_ray = v3d.Ray(
      pos=xnp.array(test_values.expected_pos),
      dir=xnp.array(test_values.expected_dir),
  )
  ray = ray.broadcast_to(shape)
  expected_ray = expected_ray.broadcast_to(shape)
  assert ray.shape == shape
  _assert_equal(tr @ ray, expected_ray)


def _assert_point_transformed(
    tr: v3d.Transform,
    shape: v3d.typing.Shape,
    test_values: TransformExpectedValue,
):
  """Test point transformation."""
  xnp = tr.xnp

  # Test transform point position
  expected_point_pos = xnp.array(test_values.expected_pos)
  expected_point_pos = xnp.broadcast_to(expected_point_pos, shape + (3,))

  point_pos = xnp.array(_RAY_POS)
  point_pos = xnp.broadcast_to(point_pos, shape + (3,))

  _assert_equal(tr @ point_pos, expected_point_pos)
  _assert_equal(tr.apply_to_pos(point_pos), expected_point_pos)

  # Test transform point direction
  expected_point_dir = xnp.array(test_values.expected_dir)
  expected_point_dir = xnp.broadcast_to(expected_point_dir, shape + (3,))

  point_dir = xnp.array(_RAY_DIR)
  point_dir = xnp.broadcast_to(point_dir, shape + (3,))

  _assert_equal(tr.apply_to_dir(point_dir), expected_point_dir)


def _assert_tr_transformed(
    tr: v3d.Transform,
    test_values: TransformExpectedValue,
):
  """Test transform transformation."""
  xnp = tr.xnp
  other_tr = v3d.Transform(
      R=xnp.array(_TR_R),
      t=xnp.array(_TR_T),
  )
  expected_tr = v3d.Transform(
      R=xnp.array(test_values.expected_r),
      t=xnp.array(test_values.expected_t),
  )
  _assert_equal(tr @ other_tr, expected_tr)
  # Composing transformations or matrix is equivalent
  _assert_equal(
      v3d.Transform.from_matrix(tr.matrix4x4 @ other_tr.matrix4x4),
      expected_tr,
  )


def _assert_equal(x, y):
  """Test that the 2 objects are identical."""
  assert isinstance(x, type(y))
  assert x.shape == y.shape  # pytype: disable=attribute-error
  testing.assert_allclose(x, y)
  if isinstance(x, v3d.DataclassArray):
    assert x.xnp is y.xnp
