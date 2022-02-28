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

"""Tests for array_dataclass."""

from __future__ import annotations

import dataclasses
from typing import Callable

from etils import enp
from etils.array_types import IntArray, FloatArray  # pylint: disable=g-multiple-import
from jax3d import visu3d as v3d
from jax3d.visu3d.typing import Shape  # pylint: disable=g-multiple-import
import numpy as np
import pytest

# Activate the fixture
set_tnp = enp.testing.set_tnp


@dataclasses.dataclass(frozen=True)
class Point(v3d.DataclassArray):
  x: FloatArray[''] = v3d.array_field(shape=())
  y: FloatArray[''] = v3d.array_field(shape=())


@dataclasses.dataclass(frozen=True)
class Isometrie(v3d.DataclassArray):
  r: FloatArray['3 3'] = v3d.array_field(shape=(3, 3))
  t: IntArray['2'] = v3d.array_field(shape=(2,), dtype=int)


@dataclasses.dataclass(frozen=True)
class Nested(v3d.DataclassArray):
  # pytype: disable=annotation-type-mismatch
  pt: Point = v3d.array_field(shape=(3,), dtype=Point)
  iso: Isometrie = v3d.array_field(shape=(), dtype=Isometrie)
  iso_batched: Isometrie = v3d.array_field(shape=(3, 7), dtype=Isometrie)
  # pytype: enable=annotation-type-mismatch


def _assert_point(p: Point, shape: Shape, xnp: enp.NpModule = None):
  """Validate the point."""
  xnp = xnp or np
  assert isinstance(p, Point)
  _assert_common(p, shape=shape, xnp=xnp)
  assert p.x.shape == shape
  assert p.y.shape == shape
  assert p.x.dtype == np.float32
  assert p.y.dtype == np.float32
  assert isinstance(p.x, xnp.ndarray)
  assert isinstance(p.y, xnp.ndarray)


def _assert_isometrie(p: Isometrie, shape: Shape, xnp: enp.NpModule = None):
  """Validate the point."""
  xnp = xnp or np
  assert isinstance(p, Isometrie)
  _assert_common(p, shape=shape, xnp=xnp)
  assert p.r.shape == shape + (3, 3)
  assert p.t.shape == shape + (2,)
  assert p.r.dtype == np.float32
  assert p.t.dtype == np.int32
  assert isinstance(p.r, xnp.ndarray)
  assert isinstance(p.t, xnp.ndarray)


def _assert_nested(p: Nested, shape: Shape, xnp: enp.NpModule = None):
  """Validate the nested."""
  xnp = xnp or np
  assert isinstance(p, Nested)
  _assert_common(p, shape=shape, xnp=xnp)
  _assert_point(p.pt, shape=shape + (3,), xnp=xnp)
  _assert_isometrie(p.iso, shape=shape, xnp=xnp)
  _assert_isometrie(p.iso_batched, shape=shape + (3, 7), xnp=xnp)


def _assert_common(p: v3d.DataclassArray, shape: Shape, xnp: enp.NpModule):
  """Test the len(p)."""
  assert p  # Object evaluate to True
  assert p.shape == shape
  assert p.xnp is xnp
  if shape:
    assert len(p) == shape[0]
  else:
    with pytest.raises(TypeError, match='of unsized '):
      _ = len(p)


def _make_point(shape: Shape, xnp: enp.NpModule) -> Point:
  """Construct the dataclass array with the given shape."""
  return Point(
      x=xnp.zeros(shape),
      y=xnp.zeros(shape),
  )


def _make_isometrie(shape: Shape, xnp: enp.NpModule) -> Isometrie:
  """Construct the dataclass array with the given shape."""
  return Isometrie(
      r=xnp.zeros(shape + (3, 3)),
      t=xnp.zeros(shape + (2,)),
  )


def _make_nested(shape: Shape, xnp: enp.NpModule) -> Nested:
  """Construct the dataclass array with the given shape."""
  return Nested(
      pt=Point(
          x=xnp.zeros(shape + (3,)),
          y=xnp.zeros(shape + (3,)),
      ),
      iso=Isometrie(
          r=xnp.zeros(shape + (3, 3)),
          t=xnp.zeros(shape + (2,)),
      ),
      iso_batched=Isometrie(
          r=xnp.zeros(shape + (3, 7, 3, 3)),
          t=xnp.zeros(shape + (3, 7, 2)),
      ),
  )


parametrize_dataclass_arrays = pytest.mark.parametrize(
    ['make_dc_array_fn', 'assert_dc_array_fn'],
    [
        (_make_point, _assert_point),
        (_make_isometrie, _assert_isometrie),
        (_make_nested, _assert_nested),
    ],
    ids=[
        'point',
        'isometrie',
        'nested',
    ],
)


@enp.testing.parametrize_xnp(with_none=True)
@pytest.mark.parametrize('x, y, shape', [
    (1, 2, ()),
    ([1, 2], [3, 4], (2,)),
    ([[1], [2]], [[3], [4]], (2, 1)),
])
def test_point_infered_np(
    xnp: enp.NpModule,
    x,
    y,
    shape: Shape,
):
  if xnp is not None:  # Normalize np arrays to test the various backend
    x = xnp.array(x)
    y = xnp.array(y)
  else:
    xnp = np

  p = Point(x=x, y=y)
  _assert_point(p, shape, xnp=xnp)


@enp.testing.parametrize_xnp()
@parametrize_dataclass_arrays
def test_scalar_shape(
    xnp: enp.NpModule,
    make_dc_array_fn: Callable[..., v3d.DataclassArray],
    assert_dc_array_fn: Callable[..., None],
):
  p = make_dc_array_fn(shape=(), xnp=xnp)
  assert_dc_array_fn(p, (), xnp=xnp)
  assert_dc_array_fn(p.reshape((1, 1, 1)), (1, 1, 1), xnp=xnp)
  assert_dc_array_fn(p.flatten(), (1,), xnp=xnp)
  assert_dc_array_fn(p.broadcast_to((7, 4, 3)), (7, 4, 3), xnp=xnp)

  with pytest.raises(TypeError, match='iteration over'):
    _ = list(p)

  with pytest.raises(IndexError, match='too many indices for array'):
    _ = p[0]

  assert_dc_array_fn(p[...], (), xnp=xnp)  # Index on ... is a no-op

  assert_dc_array_fn(v3d.stack([p, p, p]), (3,), xnp=xnp)


@enp.testing.parametrize_xnp()
@parametrize_dataclass_arrays
def test_simple_shape(
    xnp: enp.NpModule,
    make_dc_array_fn: Callable[..., v3d.DataclassArray],
    assert_dc_array_fn: Callable[..., None],
):
  p = make_dc_array_fn(shape=(3, 2), xnp=xnp)
  assert_dc_array_fn(p, (3, 2), xnp=xnp)
  assert_dc_array_fn(p.reshape((2, 1, 3, 1, 1)), (2, 1, 3, 1, 1), xnp=xnp)
  assert_dc_array_fn(p.flatten(), (6,), xnp=xnp)
  assert_dc_array_fn(p.broadcast_to((7, 4, 3, 2)), (7, 4, 3, 2), xnp=xnp)
  assert_dc_array_fn(p[0], (2,), xnp=xnp)
  assert_dc_array_fn(p[1, 1], (), xnp=xnp)
  assert_dc_array_fn(p[:, 0], (3,), xnp=xnp)
  assert_dc_array_fn(p[..., 0], (3,), xnp=xnp)
  assert_dc_array_fn(p[0, ...], (2,), xnp=xnp)
  assert_dc_array_fn(p[...], (3, 2), xnp=xnp)

  p0, p1, p2 = list(p)
  assert_dc_array_fn(p0, (2,), xnp=xnp)
  assert_dc_array_fn(p1, (2,), xnp=xnp)
  assert_dc_array_fn(p2, (2,), xnp=xnp)

  assert_dc_array_fn(v3d.stack([p0, p0, p1, p1]), (4, 2), xnp=xnp)


@enp.testing.parametrize_xnp()
@parametrize_dataclass_arrays
def test_complex_shape(
    xnp: enp.NpModule,
    make_dc_array_fn: Callable[..., v3d.DataclassArray],
    assert_dc_array_fn: Callable[..., None],
):
  p = make_dc_array_fn(shape=(3, 2, 1, 1), xnp=xnp)
  assert_dc_array_fn(p, (3, 2, 1, 1), xnp=xnp)
  assert_dc_array_fn(p.reshape((2, 1, 3, 1, 1)), (2, 1, 3, 1, 1), xnp=xnp)
  assert_dc_array_fn(p.flatten(), (6,), xnp=xnp)
  assert_dc_array_fn(p.broadcast_to((7, 3, 2, 1, 1)), (7, 3, 2, 1, 1), xnp=xnp)
  assert_dc_array_fn(p[0], (2, 1, 1), xnp=xnp)
  assert_dc_array_fn(p[1, 1], (1, 1), xnp=xnp)
  assert_dc_array_fn(p[:, 0], (3, 1, 1), xnp=xnp)
  assert_dc_array_fn(p[:, 0, 0, :], (3, 1), xnp=xnp)
  assert_dc_array_fn(p[..., 0], (3, 2, 1), xnp=xnp)
  assert_dc_array_fn(p[0, ...], (2, 1, 1), xnp=xnp)
  assert_dc_array_fn(p[0, ..., 0], (2, 1), xnp=xnp)

  p0, p1, p2 = list(p)
  assert_dc_array_fn(p0, (2, 1, 1), xnp=xnp)
  assert_dc_array_fn(p1, (2, 1, 1), xnp=xnp)
  assert_dc_array_fn(p2, (2, 1, 1), xnp=xnp)

  assert_dc_array_fn(v3d.stack([p0, p0, p1, p1]), (4, 2, 1, 1), xnp=xnp)


def test_isometrie_wrong_input():
  # Incompatible types
  with pytest.raises(ValueError, match='Conflicting numpy types'):
    _ = Isometrie(
        r=enp.lazy.jnp.zeros((3, 3)),
        t=np.zeros((2,)),
    )

  # Bad inner shape
  with pytest.raises(ValueError, match='last dimensions to be'):
    _ = Isometrie(
        r=np.zeros((3, 2)),
        t=np.zeros((2,)),
    )

  # Bad batch shape
  with pytest.raises(ValueError, match='Conflicting batch shapes'):
    _ = Isometrie(
        r=np.zeros((2, 3, 3)),
        t=np.zeros((3, 2)),
    )

  # Bad reshape
  p = Isometrie(
      r=np.zeros((3, 3, 3)),
      t=np.zeros((3, 2)),
  )
  with pytest.raises(ValueError, match='cannot reshape array'):
    p.reshape((2, 2))


@pytest.mark.parametrize('batch_shape, indices', [
    ((), np.index_exp[...]),
    ((2,), np.index_exp[...]),
    ((3, 2), np.index_exp[...]),
    ((3, 2), np.index_exp[0]),
    ((3, 2), np.index_exp[0, ...]),
    ((3, 2), np.index_exp[..., 0]),
    ((3, 2), np.index_exp[0, 0]),
    ((3, 2), np.index_exp[..., 0, 0]),
    ((3, 2), np.index_exp[0, ..., 0]),
    ((3, 2), np.index_exp[0, 0, ...]),
    ((3, 2), np.index_exp[0, :, ...]),
    ((3, 2), np.index_exp[:, ..., :]),
    ((3, 2), np.index_exp[None,]),
    ((3, 2), np.index_exp[None, :]),
    ((3, 2), np.index_exp[np.newaxis, :]),
    ((2,), np.index_exp[None, ..., None, 0, None, None]),
    ((2,), np.index_exp[None, ..., None, 0, None, None]),
    ((3, 2), np.index_exp[None, ..., None, 0, None, None]),
    ((3, 1, 2), np.index_exp[None, ..., None, 0, None, None]),
])
def test_normalize_indices(batch_shape: Shape, indices):
  # Compare the indexing with and without the extra batch shcape
  x0 = np.ones(batch_shape + (4, 2))
  x1 = np.ones(batch_shape)

  normalized_indices = v3d.array_dataclass._to_absolute_indices(
      indices,
      shape=batch_shape,
  )
  x0 = x0[normalized_indices]
  x1 = x1[indices]
  assert x0.shape == x1.shape + (4, 2)


@enp.testing.parametrize_xnp()
def test_empty(xnp: enp.NpModule):
  p = Point(x=xnp.empty((0, 3)), y=xnp.empty((0, 3)))  # Empty array

  with pytest.raises(ValueError, match='The truth value of'):
    bool(p)


@enp.testing.parametrize_xnp()
@parametrize_dataclass_arrays
def test_convert(
    xnp: enp.NpModule,
    make_dc_array_fn: Callable[..., v3d.DataclassArray],
    assert_dc_array_fn: Callable[..., None],
):
  del assert_dc_array_fn
  p = make_dc_array_fn(xnp=xnp, shape=(2,))
  assert p.xnp is xnp
  assert p.as_np().xnp is enp.lazy.np
  assert p.as_jax().xnp is enp.lazy.jnp
  assert p.as_tf().xnp is enp.lazy.tnp
  assert p.as_xnp(np).xnp is enp.lazy.np
  assert p.as_xnp(enp.lazy.jnp).xnp is enp.lazy.jnp
  assert p.as_xnp(enp.lazy.tnp).xnp is enp.lazy.tnp


@parametrize_dataclass_arrays
def test_jax_tree_map(
    make_dc_array_fn: Callable[..., v3d.DataclassArray],
    assert_dc_array_fn: Callable[..., None],
):
  p = make_dc_array_fn(shape=(3,), xnp=np)
  p = enp.lazy.jax.tree_map(lambda x: x[None, ...], p)
  assert_dc_array_fn(p, (1, 3), xnp=np)


def test_jax_vmap():
  batch_shape = 3

  @enp.lazy.jax.vmap
  def fn(ray: v3d.Ray) -> v3d.Ray:
    assert isinstance(ray, v3d.Ray)
    assert ray.shape == ()  # pylint:disable=g-explicit-bool-comparison
    return ray + 1

  x = v3d.Ray(pos=[0, 0, 0], dir=[1, 1, 1])
  x = x.broadcast_to((batch_shape,))
  x = x.as_jax()
  y = fn(x)
  assert isinstance(y, v3d.Ray)
  assert y.shape == (batch_shape,)
  # pos was updated
  np.testing.assert_allclose(y.pos, np.ones((batch_shape, 3)))
