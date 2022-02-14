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

"""Helper for functions which should only be execute on jax process zero."""

from flax.metrics import tensorboard
from flax.training import checkpoints
import jax


def _make_summary_fn(fn_name: str):
  """Creates a forwarding summary function, which prefixes the given tag."""
  def summary_fn(self, tag, *args, **kwargs):
    # pylint: disable=protected-access
    if self._impl is None:
      return
    getattr(self._impl, fn_name)(tag, *args, **kwargs)
    # pylint: enable=protected-access
  summary_fn.__name__ = fn_name
  return summary_fn


class SummaryWriter(object):
  """Creates a normal SummaryWriter if this is host 0. A dummy otherwise."""

  def __init__(self, log_dir: str):
    self._impl = None
    if jax.process_index() == 0:
      self._impl = tensorboard.SummaryWriter(log_dir)

  text = _make_summary_fn("text")
  image = _make_summary_fn("image")
  scalar = _make_summary_fn("scalar")
  histogram = _make_summary_fn("histogram")


def save_checkpoint(*args, **kwargs):
  """Only save a checkpoint if this is jax process 0. Otherwise pass."""
  if jax.process_index() != 0:
    return
  checkpoints.save_checkpoint(*args, **kwargs)
