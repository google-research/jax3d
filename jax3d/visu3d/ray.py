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

"""Ray utils."""

from __future__ import annotations

import dataclasses

from etils import edc
from etils import enp
from etils.array_types import Array, FloatArray  # pylint: disable=g-multiple-import
from jax3d.visu3d import array_dataclass
from jax3d.visu3d import np_utils
from jax3d.visu3d import plotly
from jax3d.visu3d.lazy_imports import plotly_base
import numpy as np

# TODO(epot): More dynamic sub-sampling controled in `v3d.make_fig`
_MAX_NUM_SAMPLE = 500  # pylint: disable=invalid-name

# TODO(epot):
# * Make the dir optional to allow:
#   ray = Ray(pos=[0, 0, 0]).look_at(target)
# * Add a color argument ?
# * Check broadcasting for `+` and `*`


@edc.dataclass(kw_only=True)
@dataclasses.dataclass(frozen=True)
class Ray(array_dataclass.DataclassArray, plotly.Visualizable):
  """6d vector with position and direction.

  Note: The direction is not normalized by default.

  Attributes:
    pos: Position
    dir: Direction
  """
  pos: FloatArray['*shape 3'] = array_dataclass.array_field(shape=(3,))
  dir: FloatArray['*shape 3'] = array_dataclass.array_field(shape=(3,))

  @property
  def end(self) -> FloatArray['*shape 3']:
    """The extremity of the ray (`ray.pos + ray.dir`)."""
    return self.pos + self.dir

  @classmethod
  def from_look_at(cls, pos: Array['*d 3'], end: Array['*d 3']) -> Ray:
    """Factory to create a look at Ray.

    Alias of `Ray(pos=pos, dir=end-from)`.

    Args:
      pos: Ray position
      end: Ray destination

    Returns:
      ray:
    """
    # Normalize list -> array
    xnp = enp.lazy.get_xnp(pos, strict=False)
    pos = xnp.asarray(pos)
    end = xnp.asarray(end)
    return cls(
        pos=pos,
        dir=end - pos,
    )

  def __add__(self, translation: FloatArray['... 3']) -> Ray:
    """Translate the position."""
    if isinstance(translation, Ray):
      raise TypeError('Cannot add Ray with Ray. '
                      'In `ray + x`: x should be a FloatArray[..., 3].')
    translation = self.xnp.asarray(translation)
    return self.replace(pos=self.pos + translation)

  def scale_dir(self, scale: FloatArray['...']) -> Ray:
    """Scale the dir."""
    if isinstance(scale, Ray):
      raise TypeError('Cannot multiply Ray with Ray. '
                      'In `ray * x`: x should be a scalar factor.')
    scale = self.xnp.asarray(scale)
    return self.replace(dir=self.dir * scale)

  def norm(self, keepdims: bool = False) -> FloatArray['*shape']:
    """Returns the norm of the dir."""
    return enp.compat.norm(self.dir, axis=-1, keepdims=keepdims)

  def normalize(self) -> Ray:
    """Normalize the directions."""
    return self.replace(dir=np_utils.normalize(self.dir))

  def mean(self) -> Ray:
    """Returns the average ray."""
    # Mean reduce across all axis but the last one
    axis = tuple(range(len(self.shape)))
    return self.map_field(lambda t: t.mean(axis=axis))

  def look_at(self, end: Array['*shape 3']) -> Ray:
    """Change the direction to point to the target point."""
    # Could add a `keep_norm=True` ?
    end = self.xnp.asarray(end)
    return self.replace(dir=end - self.pos)

  # Display functions

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    start = self.pos
    end = self.end
    batch_size = np.prod(self.shape)
    if batch_size > _MAX_NUM_SAMPLE:
      rng = np.random.default_rng(0)
      idx = rng.choice(batch_size, size=_MAX_NUM_SAMPLE, replace=False)
      start = start.reshape((batch_size, 3))[idx]
      end = end.reshape((batch_size, 3))[idx]
    return plotly.make_lines_traces(
        start=start,
        end=end,
        end_marker='cone',
        axis=-1,
    )
