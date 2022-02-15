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

"""Jax3d public API."""
# pylint: disable=unused-import

from jax3d import math
from jax3d import utils
from jax3d.math import volume_rendering
from jax3d.utils import testing
from jax3d.utils import tree_utils as tree
from jax3d.utils import typing
from jax3d.utils.file_utils import j3d_dir
from jax3d.utils.file_utils import nf_dir
from jax3d.utils.geo_utils import get_coords_grid
from jax3d.utils.random import RandomState
from jax3d.utils.random import uniform_points_on_sphere
from jax3d.utils.random import uniform_polar_points_on_sphere
from jax3d.utils.shape_utils import shape_dtype_like
from jax3d.utils.shape_utils import tensor_spec_like
from jax3d.utils.shape_utils import types_like
from jax3d.utils.shape_utils import zeros_like
from jax3d.utils.shape_validation import assert_typing
from jax3d.utils.xmap import xmap
