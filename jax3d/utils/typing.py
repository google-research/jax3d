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

"""Typing utils."""

from typing import Any, Callable, Type, TypeVar, Union

from etils.array_types import f32
import numpy as np
import tensorflow as tf


# *********** Common typing ***********

_T = TypeVar('_T')

# Recursive type for jax.tree
Tree = Union[_T, Any]

# Could replace by `typing.Protocol`
Dataclass = Any

# *********** Tensor-related typing ***********

Tensor = Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]
# Match both `np.dtype('int32')` and np.int32
DType = Union[np.dtype, Type[np.generic]]

# Any activation function for f32.
ActivationFn = Callable[[f32['...']], f32['...']]
