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

"""Interpolators for grids of latent codes."""

import abc
from typing import Sequence

import chex
from flax import linen as nn
from jax import numpy as jnp
from jax3d.projects.nesf.utils.typing import f32, i32  # pylint: disable=g-multiple-import


GridShape = Sequence[int]


def compute_corner_indices(
    grid_shape: GridShape,
    points: f32['... num_dims']) -> i32['... 2**num_dims num_dims']:
  """Computes the indices of the 2^num_dims corners surrounding points.

  This assumes the point grid has grid_size points in each dimension.
  Furthermore the first point is at 0.0 and the last point at 1.0 in each
  dimension.
  This means a grid_size=[n] would result in n-1 grid cells.

  Example:
    p = jnp.asarray([0.15, 0.25], dtype=jnp.float32)
    grid_shape = [11, 11]
    indices = compute_corner_indices(grid_shape=grid_shape, points=p)
    np.testing.assert_array_equal(indices, np.asarray(
        [[1, 2], [1, 3], [2, 2], [2, 3]]))

  Args:
    grid_shape: List of length num_dims. Number of grid points
      along each dimension. Shape of the point_grid.
    points: [..., num_dims] float32. Positions to get indices for.

  Returns:
    [..., 2^num_dims, num_dims] int32. Point indices of the corners that
    surround each point in points.
  """

  grid_shape = jnp.asarray(grid_shape, dtype=jnp.int32)
  # Convert points to grid coordinates
  points = points * (grid_shape -  1)
  # Get the floor of the points (lower left corner of a 2D voxel)
  points = points.astype(jnp.int32)[..., None, :]
  # For the above example: points.shape[-1] == 2
  # Iteration 1: i = 0, points = [[1, 2]]
  # offset = [1, 0]
  # points = [[1, 2], [2, 2]]
  # Iteration 2: i = 1, points = [[1, 2], [2, 2]]
  # offset = [0, 1]
  # points = [[1, 2], [2, 2], [1, 3], [2, 3]]
  for i in range(points.shape[-1]):
    offset = jnp.asarray([int(i == j) for j in range(points.shape[-1])])
    points = jnp.concatenate([points, points + offset], axis=-2)
  points = jnp.clip(points, 0, grid_shape - 1)
  return points


def compute_corner_weights(
    grid_shape: GridShape,
    points: f32['... num_dims']) -> f32['... 2**num_dims']:
  """Computes the weights of the 8 corners surrounding points.

  This function calculates the weight of each of the 2**num_dims corners, when
  performing tri-linear interpolation.

  Args:
    grid_shape: List of length num_dims. Number of voxels partitions
      along each channel. Shape of the point_grid [x, y, z].
    points: Positions to weights for.

  Returns:
    [..., 2^num_dims] float32. Trilinear weight of each voxel corner,
      in the same order as compute_corner_indices above.
  """
  num_dims = points.shape[-1]
  weights = jnp.ones(points.shape[:-1] + (1,))
  grid_shape = jnp.asarray(grid_shape)
  # The weight depends on how far away the corner is from the point.
  # This is then computed by getting the modulus of the point and the corner
  channel_weight = jnp.fmod(points * (grid_shape - 1), 1)

  # Example:
  # grid_shape = [11, 11]
  # points = jnp.asarray([0.17, 0.25], dtype=jnp.float32)
  # num_dims = 2
  # weights = [1]
  # channel_weight = [0.7, 0.5]
  # Iteration 1: i = 0
  # cw = 0.7
  # weights = [0.3, 0.7]
  # Iteration 2: i = 1
  # cw = 0.5
  # weights = [0.15, 0.35, 0.15, 0.35]

  # The following for-loop isn't as bad as it seems. For all cases we care
  # about, num_dims == 3.
  for i in range(num_dims):
    cw = channel_weight[..., i:i+1]
    # Linear interpolation to calculate weights. Note: this doubles the number
    # of weights for each dimension, so we end up with 2^num_dims weights in
    # total.
    weights = jnp.concatenate([weights * (1 - cw), weights * cw], axis=-1)

  return weights


class InterpolationFn(abc.ABC):
  """Abstract class for all implementations of latent code interpolation."""

  @abc.abstractmethod
  def __call__(
      self,
      grid_size: GridShape,
      points: f32['... num_dims'],
      latents: f32['... 2**num_dims num_features'],
      corner_indices: i32['... 2**num_dims num_dims']
      ) -> f32['... num_features']:
    """Interpolate between corners inside a single voxel.

    Args:
      grid_size: List of length num_dims. Number of voxels partitions along each
        channel. Shape of the point_grid [x, y, z].
      points: Points to get interpolated latent codes for.
      latents: Latent features to interpolate over. Ordered the same way as
        corner_indices.
      corner_indices: Indices for the voxel corners, ordered according to
        compute_corner_indices.

    Returns:
      The interpolated latent.
    """


class TrilinearInterpolation(InterpolationFn):
  """Trilinear interpolation implementation."""

  def __call__(
      self,
      grid_size: GridShape,
      points: f32['... num_dims'],
      latents: f32['... 2**num_dims num_features'],
      corner_indices: i32['... 2**num_dims num_dims']
      ) -> f32['... num_features']:
    """Interpolate between corners inside a single voxel.

    Args:
      grid_size: List of length num_dims. Number of voxels partitions along each
        channel. Shape of the point_grid [x, y, z].
      points: Points to get interpolated latent codes for.
      latents: Latent features to interpolate over. Ordered the same way as
        corner_indices.
      corner_indices: Indices for the voxel corners, ordered according to
        compute_corner_indices.

    Returns:
      The interpolated latent.
    """
    del corner_indices  # Unused.
    weights = compute_corner_weights(grid_size, points)
    return jnp.einsum('...cf,...c->...f', latents, weights)


class GridInterpolator(nn.Module):
  """Interpolation of Grids using a specified function."""
  interpolation: InterpolationFn

  @nn.compact
  def __call__(self,
               voxel_embeddings: f32['num_grids, ..., num_features'],
               grid_indexes: i32['... 1'],
               points: f32['... num_dims']
               ) -> f32['... grid_features']:
    """Tri-Linearly interpolate the latent code in the latent grid.

    For all points outside of the [-1; 1] bounding box, this module will return
    zero latent vectors.

    Args:
      voxel_embeddings: the grid of latent codes to interpolate.
      grid_indexes: the index of the grid to query.
      points: points to get interpolated latent codes for.

    Returns:
      latent codes.
    """
    chex.assert_equal_shape_prefix([grid_indexes, points],
                                   len(points.shape) - 1)
    grid_size = voxel_embeddings.shape[1:-1]
    chex.assert_axis_dimension(points, -1, len(grid_size))
    # rescale points from [-1; 1] to [0; 1]
    points = (points + 1) / 2

    # Shape: [..., 2^num_dims, num_dims] containing the corner indices
    corner_indices = compute_corner_indices(grid_size, points)
    # Shape: [..., 2^num_dims, 1]
    grid_indexes = jnp.broadcast_to(grid_indexes[..., None, :],
                                    (*corner_indices.shape[:-1], 1))
    # Shape: [..., 2^num_dims, num_dims+1]
    grid_corner_indices = jnp.concatenate(
        [grid_indexes, corner_indices], axis=-1)

    # # Move last axis to front.
    # grid_corner_indices = jnp.rollaxis(grid_corner_indices, -1)
    # latents = voxel_embeddings[(*grid_corner_indices,)]
    latents = gather_v1(voxel_embeddings, grid_corner_indices)

    latents = self.interpolation(
        grid_size=grid_size,
        points=points,
        corner_indices=corner_indices,
        latents=latents)

    # Replace all outside value with zeros.
    inside_points = (points >= 0.0) * (points <= 1.0)
    inside_points = jnp.all(inside_points, axis=-1, keepdims=True)
    latents *= inside_points

    return latents


def gather_v1(values, indices):
  """Gather values[indices[..., 0], indices[..., 1], ...] from indices[..., n].

  This implementation flattens batch dimensions in `values` and `indices`.

  Args:
    values: ...
    indices: ...

  Returns:
    ...
  """
  values = jnp.asarray(values)
  indices = jnp.asarray(indices, dtype=jnp.uint32)

  # Flatten leading dimensions of indices.
  *indices_batch_shape, num_spatial_dims = indices.shape
  indices_flat = jnp.reshape(indices, (-1, num_spatial_dims))
  indices_linear = flat_indices(indices_flat, values.shape)

  # Flatten leading dimensions of values.
  *values_batch_shape, num_value_dims = values.shape
  del values_batch_shape
  values_linear = jnp.reshape(values, (-1, num_value_dims))

  # Gather values.
  result_linear = values_linear[indices_linear]

  # Unflatten leading dimensions.
  result = jnp.reshape(result_linear, (*indices_batch_shape, num_value_dims))
  return result


def flat_indices(indices, value_shape):
  """Computes linearized indices for a flattened version of value_shape."""
  assert len(indices.shape) == 2
  assert indices.shape[-1] == len(value_shape)-1
  value_strides = strides(value_shape[:-1])
  value_strides = jnp.asarray(value_strides, dtype=indices.dtype)
  return jnp.sum(indices * value_strides, axis=-1)


def strides(shape):
  """Computes ndarray.strides for an array shape."""
  result = []
  stride = 1
  for _, x in reversed(list(enumerate(shape))):
    result.append(stride)
    stride *= x
  return list(reversed(result))
