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

"""Tests for camera_spec."""

from etils import enp
from jax3d import visu3d as v3d
import numpy as np
import pytest

# Activate the fixture
set_tnp = enp.testing.set_tnp

H, W = 640, 480


def make_camera_spec(
    *,
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
) -> v3d.PinholeCamera:
  spec = v3d.PinholeCamera.from_focal(
      resolution=(H, W),
      focal_in_px=35.,
      xnp=xnp,
  )
  spec = spec.broadcast_to(shape)
  return spec


@enp.testing.parametrize_xnp(with_none=True)
@pytest.mark.parametrize('spec_shape', [(), (3,), (2, 3)])
def test_camera_spec_init(
    xnp: enp.NpModule,
    spec_shape: v3d.typing.Shape,
):
  if spec_shape and xnp is enp.lazy.tnp:
    pytest.skip('Vectorization not supported yet with TF')

  spec = make_camera_spec(xnp=xnp, shape=spec_shape)
  assert spec.resolution == (H, W)
  assert spec.h == H
  assert spec.w == W
  assert spec.shape == spec_shape
  assert spec.K.shape == spec_shape + (3, 3)

  if xnp is None:
    xnp = np
  assert spec.xnp is xnp

  x = _broadcast_to(xnp, [0, 0, 1], (1,) * len(spec_shape) + (3,))
  assert isinstance(spec.cam_to_px(x), xnp.ndarray)

  x = _broadcast_to(xnp, [0, 0], (1,) * len(spec_shape) + (2,))
  assert isinstance(spec.px_to_cam(x), xnp.ndarray)


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('spec_shape', [(), (3,), (2, 3)])
@pytest.mark.parametrize('point_shape', [(), (2, 3)])
def test_camera_spec_central_point(
    xnp: enp.NpModule,
    spec_shape: v3d.typing.Shape,
    point_shape: v3d.typing.Shape,
):
  if spec_shape and xnp is enp.lazy.tnp:
    pytest.skip('Vectorization not supported yet with TF')

  spec = make_camera_spec(xnp=xnp, shape=spec_shape)

  # Projecting the central point (batched)
  central_point_cam = _broadcast_to(
      xnp,
      [0, 0, 1],
      spec_shape + point_shape + (3,),
  )
  central_point_px = spec.cam_to_px(central_point_cam)
  assert isinstance(central_point_px, xnp.ndarray)
  assert central_point_px.shape == spec_shape + point_shape + (2,)
  np.testing.assert_allclose(
      central_point_px,
      np.broadcast_to([H / 2, W / 2], spec_shape + point_shape + (2,)),
  )

  # Round trip conversion
  np.testing.assert_allclose(
      central_point_cam,
      spec.px_to_cam(central_point_px),
      atol=1e-4,
  )


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('spec_shape', [(), (2, 3)])
def test_camera_px_centers(
    xnp: enp.NpModule,
    spec_shape: v3d.typing.Shape,
):
  if spec_shape and xnp is enp.lazy.tnp:
    pytest.skip('Vectorization not supported yet with TF')

  spec = make_camera_spec(xnp=xnp, shape=spec_shape)

  px_centers = spec.px_centers()
  assert isinstance(px_centers, xnp.ndarray)
  assert px_centers.shape == spec_shape + (H, W, 2)

  cam_centers = spec.px_to_cam(px_centers)
  assert isinstance(cam_centers, xnp.ndarray)
  assert cam_centers.shape == spec_shape + (H, W, 3)

  np.testing.assert_allclose(
      cam_centers,
      spec.px_to_cam(),  # Default to cam_centers
  )

  round_trip_px = spec.cam_to_px(cam_centers)
  assert isinstance(round_trip_px, xnp.ndarray)
  assert round_trip_px.shape == spec_shape + (H, W, 2)

  np.testing.assert_allclose(round_trip_px, px_centers, atol=1e-4)
  # Scaling/normalizing don't change the projection
  np.testing.assert_allclose(
      round_trip_px,
      spec.cam_to_px(cam_centers * 3.),
      atol=1e-4,
  )


def _broadcast_to(xnp: enp.NpModule, array, shape):
  return xnp.broadcast_to(xnp.array(array), shape)
