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

"""Wrapper around `xmap`."""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar

import jax
from jax.experimental import maps
from jax3d.projects.nesf.utils import py_utils
from jax3d.projects.nesf.utils.typing import Array, Tree  # pylint: disable=g-multiple-import


_TreeArrayT = TypeVar('_TreeArrayT')
_Args = Tuple[Any, ...]
_Kwargs = Dict[str, Any]


@py_utils.decorator_with_option
def xmap(
    fn: Callable[..., _TreeArrayT],
    in_axes: Sequence[Tree[str]],
    out_axes: Tree[str],
    *,
    axis_resources: Optional[Dict[str, str]] = None,
    backend: Optional[str] = None,
) -> _TreeArrayT:
  """`jax.xmap` with additional features.

  * Can be used as decorator directly (without partial)
  * Less verbose API to declare inputs/outputs
  * All additional arguments not defined in `in_axes`
    are forwarded as static kwargs

  ```python
  @xmap(['b ...'], 'b ...')
  def f(x, is_train=False):
    ...

  y = f(x)
  y = f(x, is_train=True)
  ```

  For the syntax: The shape is defined by a single string of tokens separated
  by space. Tokens can be axis name, `_` or `...`. For example: `b h w c`,
  `batch ...`, `b ... c`, `b _ _ c ...`.

  Args:
    fn: same as `jax.xmap`
    in_axes: same as `jax.xmap`
    out_axes: same as `jax.xmap`
    axis_resources: same as `jax.xmap`
    backend: same as `jax.xmap`

  Returns:
    decorated function or decorator (if fn wasn't provided)
  """
  # TODO(jax3d.projects.nesf):
  # * Add shape validation for fixed dim shape (`'h _ c'` with input length > 3)
  # Possible ameliorations: let's wait concrete use case before implementing.
  # * Optional array args:
  #     @xmap(['b ...', 'b ...'], 'b ...')
  #     def fn(x, y=None):  # Can be called as fn(x) or fn(x, y)
  # * Interleave static and array args (e.g. xmap(['b ...', None, 'b ...']))
  # * Support when function signature is hidden, or when function signature is
  #   args, kwargs.
  # * Performance optimization (e.g. do not inspect signature when not required)
  # Other design suggestion:
  # * Also add name for non vectorized axis, only for documentation/shape
  #   checking but not used by the API. Which syntax ?
  #   `['{b} h w c', '{b} n'], '{b} n c'`
  #   The grid_sample example: `['{b} ... {c}', '{b} num_dim n'] -> '{b} n {c}'`
  #   This would allow to naturally extend `arr.mean('{b} h w c -> h w')`
  #   In this case, remove the `_` token.
  #   (eventually) Have a meta language to allow things like
  #   ['{b} *dims {c}', '{b} len(dims) n'], '{b} n {c}'` but the overhead
  #   complexity is likely not worth the few places where this could be used.
  # * Declare spec as dict (e.g. xmap({'x': 'b ...', 'y': 'b ...'})) in addition
  #   of list for better kwargs support ?

  if not isinstance(in_axes, (list, tuple)):
    raise TypeError(f'in_axes should be a list of args specs. Not {in_axes}')

  in_axes = jax.tree_map(_str_to_dict_shape_spec, tuple(in_axes))
  out_axes = jax.tree_map(_str_to_dict_shape_spec, out_axes)

  @functools.wraps(fn)
  def decorated(*args, **kwargs):

    # Split args, kwargs into array args and static args.
    # The array args are then passed to `xmap` while the static args
    # are passed through closure within the xmap call.
    arr_args, merge_args_fn = _split_static_args(fn, len(in_axes), args, kwargs)

    # TODO(jax3d.projects.nesf):
    # Remove once https://github.com/google/jax/issues/5652 is
    # fixed
    in_axes_normalized = jax.tree_map(
        _normalize_dict_shape, arr_args, in_axes
    )

    def arr_fn(*arr_args):
      merged_args, merged_kwargs = merge_args_fn(arr_args)
      return fn(*merged_args, **merged_kwargs)

    # Apply and call the original xmap
    return maps.xmap(
        arr_fn,
        in_axes=in_axes_normalized,
        out_axes=out_axes,
        axis_resources=axis_resources or {},
        backend=backend,
    )(*arr_args)

  return decorated


def _split_static_args(
    fn: Callable[..., Any],
    num_array_args: int,
    args: Any,
    kwargs: Any,
) -> Tuple[_Args, Callable[[_Args], Tuple[_Args, _Kwargs]]]:
  """Extracts the array and static kwargs.

  Usage:

  ```
  def fn(x, y, is_train=True):
    ...

  arr_args, merge_args_fn = _split_static_args(fn, 2, args, kwargs)
  args, kwargs = merge_args_fn(arr_args)

  # fn(x, y)
  # fn(x)  # TODO
  ```

  Args:
    fn: Function with array and static
    num_array_args: Number of array arguments
    args: Function args
    kwargs: Function kwargs

  Returns:
    arr_args: The array arguments extracted from `args` and `kwargs`
    merge_args_fn: The function to reconstruct `args`/`kwargs` from the updated
      arr_args
  """
  sig = inspect.signature(fn)
  bargs = sig.bind(*args, **kwargs)
  # Extract the first args from the signature
  arr_items = tuple(bargs.arguments.items())[:num_array_args]
  arr_names, arr_args = zip(*arr_items)

  def merge_args_fn(arr_args):
    # Merge the arr_args back
    for arr_name, arr_arg in zip(arr_names, arr_args):
      bargs.arguments[arr_name] = arr_arg
    return bargs.args, bargs.kwargs

  return arr_args, merge_args_fn


def _normalize_dict_shape(
    arr: Array,
    spec_dict: Dict[int, str],
) -> Dict[int, str]:
  """Converts -1 dimensions into absolute positive dimensions."""
  return {
      len(arr.shape) + axis_id if axis_id < 0 else axis_id: axis_name
      for axis_id, axis_name in spec_dict.items()
  }


def _str_to_dict_shape_spec(spec_str: str) -> Dict[int, str]:
  """Converts the `str` .

  Args:
    spec_str: The human readable spec string

  Returns:
    spec_dict: The boilerplate Jax `xmap` format.
  """
  tokens = spec_str.split()
  if tokens.count('...') > 1:
    raise ValueError(f'Invalid format \'{spec_str}\': Only one `...` allowed.')

  axis_id = -1
  spec_dict = {}
  for token in tokens:
    axis_id += 1
    if token == '_':  # Unused single dimension
      pass
    elif token == '...':  # Dynamic dimension
      # Inverse the axis_id to start be the end.
      # '... c' -> -2 == 0 - 2
      # 'b ... c' -> -2 == 1 - 3
      axis_id = axis_id - len(tokens)
    else:
      spec_dict[axis_id] = token
  return spec_dict
