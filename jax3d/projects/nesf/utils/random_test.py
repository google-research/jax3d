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

"""Tests for jax3d.utils.random."""

import jax
import jax.numpy as jnp
import jax3d.projects.nesf as jax3d
import pytest


def test_random_sequence():
  jax_rng = jax.random.PRNGKey(0)
  jax_rng, new_key0 = jax.random.split(jax_rng)
  jax_rng, new_key1 = jax.random.split(jax_rng)
  jax_rng, new_key2 = jax.random.split(jax_rng)

  rng = jax3d.RandomState(0)
  assert jnp.allclose(rng.next(), new_key0)
  assert jnp.allclose(rng.next(), new_key1)
  assert jnp.allclose(rng.next(), new_key2)


@pytest.mark.parametrize('n', [1, 3, 117])
def test_random_sequence_n(n):
  jax_rng = jax.random.PRNGKey(0)
  x = jax.random.split(jax_rng, n + 1)
  jax_rng = x[0]
  jax_key = x[1:]

  rng = jax3d.RandomState(0)
  assert jnp.allclose(rng.next(n), jax_key)
  assert jnp.allclose(rng.curr_key, jax_rng)


def test_fork():
  jax_rng0 = jax.random.PRNGKey(0)
  jax_rng0, jax_rng1 = jax.random.split(jax_rng0)
  _, new_key0 = jax.random.split(jax_rng0)
  _, new_key1 = jax.random.split(jax_rng1)

  rng0 = jax3d.RandomState(0)
  rng1 = rng0.fork()
  assert jnp.allclose(rng0.next(), new_key0)
  assert jnp.allclose(rng1.next(), new_key1)


def test_fold():
  jax_rng = jax.random.PRNGKey(17)
  jax_rng = jax.random.fold_in(jax_rng, 19)

  rng = jax3d.RandomState(17)
  rng.fold_in(19)
  assert jnp.allclose(jax_rng, rng.curr_key)


def test_fold_in_stateless():
  jax_rng = jax.random.PRNGKey(17)
  jax_rng = jax.random.fold_in(jax_rng, 19)

  rng = jax3d.RandomState(17)
  assert jnp.allclose(jax_rng, rng.fold_in_stateless(19).curr_key)

  # Applying fold-in twice yield the same results
  rng = jax3d.RandomState(17)
  assert jnp.allclose(jax_rng, rng.fold_in_stateless(19).curr_key)
