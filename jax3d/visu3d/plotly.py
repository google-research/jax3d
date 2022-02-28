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

"""Plotly utils."""

from __future__ import annotations

import abc
from collections.abc import Sequence  # pylint: disable=g-importing-member
from typing import Dict, List, Optional, Union

from etils import enp
from etils.array_types import Array, FloatArray, IntArray  # pylint: disable=g-multiple-import
from jax3d.visu3d import array_dataclass
from jax3d.visu3d.lazy_imports import plotly_base
from jax3d.visu3d.lazy_imports import plotly_go as go
import numpy as np

_Primitive = Union[str, int, bool, float]
_PlotlyKwargs = Dict[str, Union[np.ndarray, _Primitive]]

del abc  # TODO(epot): Why pytype doesn't like abc.ABC ?

# TODO(epot): More dynamic sub-sampling:
# * controled in `v3d.make_fig`
# * globally assigned (collect the global batch shape)
# * Add a tqdm bar ?
_MAX_NUM_SAMPLE = 10_000  # pylint: disable=invalid-name


class Visualizable:  # (abc.ABC):
  """Interface for elements which are visualizable."""

  # @abc.abstractmethod
  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    """Construct the traces of the given object."""
    raise NotImplementedError

  @property
  def fig(self) -> go.Figure:
    """Construct the figure of the given object."""
    return make_fig([self])

  # TODO(epot): Could add a button to switch between repr <> plotly ?
  # Using `_repr_html_` ?


VisualizableItem = Union[Visualizable, Array[...]]
VisualizableArg = Union[VisualizableItem, List[VisualizableItem]]


# TODO(epot): Potential improvement:
# * Accept a `dict[str, data]` (to have trace name)
# * Allow wrapping data in some `v3d.FigData(data, **kwargs)` to allow
#   customize metadata (color, point name,...) ?
# * Allow nested structure to auto-group multiple traces ?
def make_fig(data: VisualizableArg) -> go.Figure:
  """Returns the figure from the given data."""
  traces = make_traces(data)
  fig = go.Figure(data=traces)
  fig.update_scenes(aspectmode='data')  # Keep equal axis
  return fig


def make_traces(data: VisualizableArg) -> list[plotly_base.BaseTraceType]:
  """Returns the traces from the given data."""
  if not isinstance(data, (tuple, list)):
    data = [data]

  # TODO(epot): Should dynamically sub-sample across all traces, instead of
  # subsampling individual traces.
  traces = []
  for val in data:
    if isinstance(val, Visualizable):
      if isinstance(val, array_dataclass.DataclassArray):
        val = val.as_np()
      sub_traces = val.make_traces()
      # Normalizing trace
      if isinstance(sub_traces, plotly_base.BaseTraceType):
        sub_traces = [sub_traces]
      traces.extend(sub_traces)
    elif enp.lazy.is_array(val) or isinstance(val, list):
      val = np.asarray(val)
      traces.extend(make_points(val))
    elif isinstance(val, plotly_base.BaseTraceType):  # Already a trace
      traces.append(val)
    else:
      raise TypeError(f'Unsuported {type(val)}')
  return traces


def make_points(
    array: FloatArray['*d 3'],
    *,
    rgb: IntArray['*d 3'] = None,
) -> list[plotly_base.BaseTraceType]:
  """Uses simple heuristic to display the plot matching the array content."""
  if array.shape[-1] != 3:
    raise ValueError('Only Array[..., 3] supported for now. Got '
                     f'shape={array.shape}')

  # TODO(epot): Subsample array if nb points >500
  array, rgb = subsample(array, rgb, num_samples=_MAX_NUM_SAMPLE)  # pylint: disable=unbalanced-tuple-unpacking

  if rgb is not None:
    assert rgb.shape == array.shape
    rgb = [f'rgb({r}, {g}, {b})' for r, g, b in rgb]

  points_xyz_kwargs = to_xyz_dict(array)
  point_cloud = go.Scatter3d(
      **points_xyz_kwargs,
      mode='markers',
      marker=go.scatter3d.Marker(
          size=2.,
          color=rgb,
      ),
  )
  return [point_cloud]


def make_lines_traces(
    start: FloatArray['... 3'],
    end: FloatArray['... 3'],
    *,
    axis: int = -1,
    end_marker: Optional[str] = None,
) -> list[plotly_base.BaseTraceType]:
  """Trace lines."""
  # TODO(epot): Add `legendgroup` so cones are toogled together
  lines_xyz_kwargs = make_lines_kwargs(
      start=start,
      end=end,
      axis=axis,
  )
  lines_trace = go.Scatter3d(
      **lines_xyz_kwargs,
      mode='lines',
  )
  traces = [lines_trace]
  if end_marker is None:
    pass
  elif end_marker == 'cone':
    cone_kwargs = make_cones_kwargs(
        start=start,
        direction=end - start,
        axis=axis,
    )
    cone_traces = go.Cone(
        **cone_kwargs,
        showlegend=False,
        showscale=False,
        sizemode='absolute',  # Not sure what's the difference with `scaled`
        sizeref=.5,
        # TODO(epot): Add color
        # colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']]
    )
    traces.append(cone_traces)
  else:
    raise ValueError(f'Invalid end_marker={end_marker!r}')
  return traces


def make_lines_kwargs(
    start: FloatArray['... 3'],
    end: FloatArray['... 3'],
    *,
    axis: int = -1,
) -> _PlotlyKwargs:
  """Returns the kwargs to plot lines."""
  assert axis == -1
  # 1) Flatten the arrays
  # Shape is `*d 3`
  assert start.shape == end.shape
  assert start.shape[-1] == 3

  start = start.reshape((-1, 3))
  end = end.reshape((-1, 3))

  # 2) Build the lines
  lines_xyz = [[], [], []]
  for s, e in zip(start, end):
    for i in range(3):
      lines_xyz[i].append(s[i])
      lines_xyz[i].append(e[i])
      lines_xyz[i].append(None)
  return to_xyz_dict(lines_xyz, axis=0)


def make_cones_kwargs(
    start: FloatArray['... 3'],
    direction: FloatArray['... 3'],
    *,
    start_ratio: float = 0.98,
    axis: int = -1,
) -> _PlotlyKwargs:
  """Returns the kwargs to plot cones."""
  assert axis == -1
  # 1) Flatten the arrays
  # Shape is `*d 3`
  assert start.shape == direction.shape
  assert start.shape[-1] == 3

  start = start.reshape((-1, 3))
  direction = direction.reshape((-1, 3))

  # 2) Build the lines
  xyz = start + start_ratio * direction
  uvw = direction
  return {
      **to_xyz_dict(xyz),
      **to_xyz_dict(uvw, names='uvw'),
  }


def to_xyz_dict(
    arr: Array['... 3'],
    *,
    pattern: str = '{}',
    names: Union[str, Sequence[str]] = 'xyz',
    axis: int = -1,
) -> _PlotlyKwargs:
  """Convert np.array to xyz dict.

  Useful to create plotly kwargs from numpy arrays.

  Example:

  ```python
  to_xyz_dict(np.zeros((1, 3))) == {'x': [0], 'y': [0], 'z': [0]}
  to_xyz_dict(
    [0, 1, 2],
    pattern='axis_{}'
    names='uvw',
  ) == {'axis_u': 0, 'axis_v': 1, 'axis_w': 2}
  ```

  Args:
    arr: Array to convert
    pattern: Pattern to use for the axis names
    names: Names of the axis (default to 'x', 'y', 'z')
    axis: Axis containing the x, y, z coordinates to dispatch

  Returns:
    xyz_dict: The dict containing plotly kwargs.
  """
  arr = np.asarray(arr)
  if arr.shape[axis] != len(names):
    raise ValueError(f'Invalid shape: {arr.shape}[{axis}] != {len(names)}')

  # Build the `dict(x=arr[..., 0], y=arr[..., 1], z=arr[..., 2])`
  vals = {
      pattern.format(axis_name): arr_slice.flatten()
      for axis_name, arr_slice in zip(names, np.moveaxis(arr, axis, 0))
  }
  # Normalize scalars (as plotly reject `np.array(1)`)
  vals = {k: v if v.shape else v.item() for k, v in vals.items()}
  return vals


def subsample(
    *arrays: Optional[Array['... d']],
    num_samples: int,
) -> list[Optional[Array['...']]]:
  """Flatten and subsample the arrays (keeping the last dimension)."""
  assert arrays[0] is not None
  shape = arrays[0].shape
  assert len(shape) >= 1
  # TODO(b/198633198): Warning: In TF `bool(shape) == True` for `shape==()`
  if len(shape) == 1:
    batch_size = 1  # Special case because `np.prod([]) == 1.0`
  else:
    batch_size = np.prod(shape[:-1])

  if batch_size > num_samples:
    # All arrays are sub-sampled the same way, so generate ids separately
    rng = np.random.default_rng(0)
    idx = rng.choice(batch_size, size=num_samples, replace=False)

  arrays_out = []
  for arr in arrays:
    if arr is None:
      arrays_out.append(None)
      continue
    if arr.shape != shape:
      raise ValueError('Incompatible shape')
    arr = arr.reshape((batch_size, 3))  # Flatten
    if batch_size > num_samples:
      arr = arr[idx]
    arrays_out.append(arr)

  return arrays_out
