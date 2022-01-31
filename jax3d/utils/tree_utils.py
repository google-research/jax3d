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

"""Tree utils."""

from typing import Callable

from etils.etree import jax as etree
import flax
from flax import traverse_util
from flax.core import freeze
from flax.core import unfreeze
from jax import numpy as jnp

parallel_map = etree.parallel_map
unzip = etree.unzip


def _sorted_items(x):
  """Returns items of a dict ordered by keys."""
  return sorted(x.items(), key=lambda x: x[0])


# TODO(tutmann): Add unit tests.
def filter_tree(variable_dict: flax.core.FrozenDict,
                filter_fn: Callable[[str, jnp.ndarray], bool]
                ) -> flax.core.FrozenDict:
  """Filter a nested dict.

  Args:
    variable_dict: Nested dictionary of variables.
    filter_fn: Function that decides for each variable whether it fn should be
      applied or not.

  Returns:
    Filtered copy of variable_dict containing all keys where filter_fn is True.
  """
  variable_dict = unfreeze(variable_dict)
  flat_dict = traverse_util.flatten_dict(variable_dict, keep_empty_nodes=True)
  new_dict = {}
  for key, value in _sorted_items(flat_dict):
    # empty_node is not an actual leave. It's just a stub for empty nodes
    # in the nested dict.
    if value is not traverse_util.empty_node:
      path = '/' + '/'.join(key)
      if filter_fn(path, value):
        new_dict[key] = value
  new_params = traverse_util.unflatten_dict(new_dict)
  return freeze(new_params)


# TODO(tutmann): Add unit tests.
def merge_trees(destination: flax.core.FrozenDict, source: flax.core.FrozenDict,
                ) -> flax.core.FrozenDict:
  """Merge source into destination.

  Args:
    destination: Destination frozen dict. Defines the default values.
    source: Source frozen dict. Values in this dict override ones in
      destination. Note that this only applies to keys that already exist in
      destination.

  Returns:
    Frozen dict copy of destination with values of source for all keys
    that exist in both destination and source.
  """
  destination = unfreeze(destination)
  source = unfreeze(source)
  flat_destination = traverse_util.flatten_dict(
      destination, keep_empty_nodes=True)
  flat_source = traverse_util.flatten_dict(source, keep_empty_nodes=True)
  new_dict = {}
  for key, value in _sorted_items(flat_destination):
    # empty_node is not an actual leave. It's just a stub for empty nodes
    # in the nested dict.
    if value is not traverse_util.empty_node:
      if key in flat_source:
        new_dict[key] = flat_source[key]
      else:
        new_dict[key] = value
  new_params = traverse_util.unflatten_dict(new_dict)
  return freeze(new_params)
