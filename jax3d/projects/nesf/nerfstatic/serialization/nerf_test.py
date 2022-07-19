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

# pylint: disable=redefined-outer-name
"""Tests for nerf."""

import flax.core
import jax
import jax.numpy as jnp
import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic.models import models
from jax3d.projects.nesf.nerfstatic.serialization import nerf
from jax3d.projects.nesf.nerfstatic.utils import types
import numpy as np
import pytest


def test_nerf_saver_save_state(nerf_state, nerf_saver):
  # Ensure that save_state() writes files to disk.

  # Write to disk.
  scene_id = 0
  nerf_saver.save_state(scene_id=scene_id, nerf_state=nerf_state)

  # Assert files are on disk.
  assert nerf_saver._filepath(scene_id, nerf.SUFFIX_PARAMS).exists()
  assert nerf_saver._filepath(scene_id, nerf.SUFFIX_VARIABLES).exists()
  assert nerf_saver._filepath(scene_id, nerf.SUFFIX_DENSITY_GRID).exists()


def test_nerf_saver_round_trip(nerf_state, nerf_saver):
  # Ensure that load_state() restores contents from disk.

  # Write to disk.
  scene_id = 0
  nerf_saver.save_state(scene_id=scene_id, nerf_state=nerf_state)

  # Read from disk.
  loaded_nerf_state = nerf_saver.load_state(scene_id=scene_id)

  # Assert files are on disk.
  assert nerf_state.params == loaded_nerf_state.params
  _assert_tree_allclose(nerf_state.variables, loaded_nerf_state.variables)
  np.testing.assert_allclose(nerf_state.density_grid,
                             loaded_nerf_state.density_grid)


def test_nerf_saver_multiple_scenes(nerf_saver):
  # Ensure scene_id is used when storing and loading different scenes.

  nerf_state_a = _nerf_state(jax3d.RandomState(1), _placeholder_batch())
  nerf_state_b = _nerf_state(jax3d.RandomState(2), _placeholder_batch())

  nerf_saver.save_state(scene_id=0, nerf_state=nerf_state_a)
  nerf_saver.save_state(scene_id=1, nerf_state=nerf_state_b)

  loaded_nerf_state_a = nerf_saver.load_state(scene_id=0)
  loaded_nerf_state_b = nerf_saver.load_state(scene_id=1)

  _assert_tree_allclose(loaded_nerf_state_a.variables, nerf_state_a.variables)
  _assert_tree_allclose(loaded_nerf_state_b.variables, nerf_state_b.variables)
  _assert_tree_notclose(loaded_nerf_state_a.variables,
                        loaded_nerf_state_b.variables)


################################################################################
# Fixtures


@pytest.fixture
def placeholder_batch():
  return _placeholder_batch()


@pytest.fixture
def nerf_saver(tmp_path):
  """Constructs a NerfSaver instance in a temporary directory."""
  return nerf.NerfSaver(tmp_path)


@pytest.fixture
def rng():
  """Default random number generator."""
  return jax3d.RandomState(42)


@pytest.fixture
def nerf_state(rng, placeholder_batch):
  return _nerf_state(rng, placeholder_batch)


################################################################################
# Helper functions


def _placeholder_batch():
  """Constructs a placeholder batch for initializing a NeRF model."""
  batch_size = 8
  zeros = lambda n, dtype: jnp.zeros((batch_size, n), dtype=dtype)
  return types.Batch(
      target_view=types.Views(
          rays=types.Rays(
              scene_id=zeros(1, jnp.int32),
              origin=zeros(3, jnp.float32),
              direction=zeros(3, jnp.float32),
              base_radius=None,
          ),
          depth=zeros(1, jnp.float32),
          rgb=zeros(3, jnp.float32),
          semantics=None,
          image_ids=None,
      ))


def _nerf_state(rng, placeholder_batch):
  """Constructs a NerfState instance."""
  n = 6
  params = models.NerfParams(net_depth=1)
  initialized_model = models.construct_nerf(
      rng=rng, num_scenes=1, placeholder_batch=placeholder_batch, args=params)
  density_grid = jnp.zeros((n, n, n, 1))
  return nerf.NerfState(
      params=params,
      variables=flax.core.unfreeze(initialized_model.variables),
      density_grid=density_grid)


def _assert_tree_allclose(tree_a, tree_b):
  """Fails if tree_a != tree_b."""
  jax.tree_map(np.testing.assert_allclose, tree_a, tree_b)


def _assert_tree_notclose(tree_a, tree_b):
  """Fails if tree_a == tree_b."""

  def notclose(x, y):
    return not np.allclose(x, y)

  flat_a, _ = jax.tree_util.tree_flatten(tree_a)
  flat_b, _ = jax.tree_util.tree_flatten(tree_b)
  results = [notclose(x, y) for x, y in zip(flat_a, flat_b)]
  assert any(results)
