# Copyright 2021 The jax3d Authors.
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

"""Small wrapper around `jax.random`."""

from typing import overload, Optional, Union

from etils.array_types import PRNGKey, ui32  # pylint: disable=g-multiple-import
import jax
from jax import numpy as jnp
import numpy as np


class RandomState:
  """Small stateful wrapper around `jax.random.split` to reduce boilerplate.

  Usage:

  ```
  rng = jax3d.RandomState(0)
  jax.random.uniform(rng.next())
  ```

  """

  def __init__(self, seed_or_rng: Union[int, PRNGKey]):
    """Constructor."""
    if isinstance(seed_or_rng, (np.ndarray, jnp.ndarray)):
      self.curr_key = seed_or_rng
    elif isinstance(seed_or_rng, int):
      self.curr_key = jax.random.PRNGKey(seed_or_rng)
    else:
      raise TypeError(f'Invalid seed or key: {seed_or_rng}')

  @overload
  def next(self, n: None = None) -> PRNGKey:
    ...

  @overload
  def next(self, n: int) -> ui32['n 2']:
    ...

  def next(self, n=None):
    """Returns the next rng key."""
    if n is None:
      self.curr_key, key = jax.random.split(self.curr_key)
      return key
    else:
      keys = jax.random.split(self.curr_key, n + 1)
      self.curr_key = keys[0]
      return keys[1:]

  def fork(self) -> 'RandomState':
    """Returns another RandomState initialised with `.next()`."""
    return RandomState(self.next())  # pylint: disable=not-callable

  def fold_in(self, data: int) -> None:
    """Folds in delta into the random state."""
    self.curr_key = jax.random.fold_in(self.curr_key, data)

  def fold_in_stateless(self, data: int) -> 'RandomState':
    """Folds in delta into the random state.

    This version is stateless, so do not modify the random state of the
    instance. Instead, return a new `RandomState` instance with updated state.

    Args:
      data: Delta to fold-in.

    Returns:
      The new `RandomState`
    """
    return RandomState(jax.random.fold_in(self.curr_key, data))

  def bind_to_host_device(self,
                          *,
                          bind_to: str,
                          axis_name: Optional[str] = None):
    """Binds this random state to the host/device we are on."""
    self.curr_key = self.bind_to_host_device_stateless(
        bind_to=bind_to, axis_name=axis_name).curr_key

  def bind_to_host_device_stateless(
      self,
      *,
      bind_to: str,
      axis_name: Optional[str] = None,
  ) -> 'RandomState':
    """Binds the random state to the host/device we are on.

    Must be called from within a pmapped function. Note that when binding to
    "device", we also bind the rng to hosts, as we fold_in the rng with
    axis_index which is unique for devices across all hosts.

    Args:
      bind_to: Must be one of the 'host' or 'device'. None means no binding.
      axis_name: The axis of the devices we are binding rng across.
        Only required when bind_to = 'device'.

    Returns:
      jax.random.PRNGKey specialized to host/device.
    """
    if bind_to == 'host':
      return RandomState(jax.random.fold_in(self.curr_key, jax.process_index()))
    elif bind_to == 'device':
      return RandomState(jax.random.fold_in(self.curr_key,
                                            jax.lax.axis_index(axis_name)))
    else:
      raise ValueError(
          "`bind_to` should be one of the `[None, 'host', 'device']`")

