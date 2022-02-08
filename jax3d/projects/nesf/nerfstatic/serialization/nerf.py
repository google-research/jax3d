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

"""Load and store NeRF state."""

import ast
from typing import Union

import chex
from flax.training import checkpoints
import jax.numpy as jnp
import jax3d.projects.nesf as jax3d
from jax3d.projects.nesf.nerfstatic.models import models
from jax3d.projects.nesf.utils.typing import f32, PathLike, Tree  # pylint: disable=g-multiple-import


# Filepath suffixes for different types of objects written to disk.
SUFFIX_PARAMS = 'params'
SUFFIX_VARIABLES = 'variables'
SUFFIX_DENSITY_GRID = 'density_grid'


@chex.dataclass
class NerfState:
  """All state needed to describe a trained NeRF model."""

  # Model parameters used to instantiate the model.
  params: models.NerfParams

  # Variables describing model's state.
  variables: Tree[jnp.ndarray]

  # Density grid derived from the model.
  density_grid: f32['x y z 1']


class NerfSaver:
  """Utility for saving and restoring NeRF models.

  Models are stored as a triplet of,

    1. The NerfParams used to build the NeRF
    2. The variables of the NeRF model
    3. A grid of density values derived from the NeRF model.

  Each of these is stored in a file or directory under the working directory
  with the following files,

    scene_${scene_id}.params
    scene_${scene_id}.variables/checkpoint_0
    scene_${scene_id}.density_grid

  """

  def __init__(self, working_directory: PathLike):
    """Initializes a NerfSaver object.

    Args:
      working_directory: Base directory to write NeRF models to.

    """
    self._working_directory = jax3d.Path(working_directory)
    self._working_directory.mkdir(parents=True, exist_ok=True)

  def save_state(self, scene_id: Union[int, str], nerf_state: NerfState):
    """Save state of a NeRF model for a single scene.

    Args:
      scene_id: Identifier for this scene.
      nerf_state: State of a NeRF model.
    """
    # TODO(duckworthd): The following is not resilient to preemptions. Write
    # each to a temporary directory, then relocate the directory at the end.
    _write_params(self._filepath(scene_id, SUFFIX_PARAMS), nerf_state.params)
    _write_variables(
        self._filepath(scene_id, SUFFIX_VARIABLES), nerf_state.variables)
    _write_density_grid(
        self._filepath(scene_id, SUFFIX_DENSITY_GRID), nerf_state.density_grid)

  def load_state(self, scene_id: Union[int, str]) -> NerfState:
    """Load state of a NeRF model for a single scene.

    Fails if any element of the NerfState cannot be found on disk. To fully
    reconstruct a usable NeRF model, see `construct_nerf()` in
    `jax3d.nerfstatic.models.models`.

    Args:
      scene_id: Identifier for this scene.

    Returns:
      nerf_state: State of a NeRF model.
    """
    return NerfState(
        params=_read_params(self._filepath(scene_id, SUFFIX_PARAMS)),
        variables=_read_variables(self._filepath(scene_id, SUFFIX_VARIABLES)),
        density_grid=_read_density_grid(
            self._filepath(scene_id, SUFFIX_DENSITY_GRID)))

  def _filepath(self, scene_id: Union[int, str], suffix) -> jax3d.Path:
    if isinstance(scene_id, int):
      scene_id = f'{scene_id:05d}'
    return self._working_directory / f'scene_{scene_id}.{suffix}'


################################################################################
# Helper methods


def _write_params(path: jax3d.Path, params: models.NerfParams):
  text = repr(params.to_dict())
  path.write_text(text)


def _read_params(path: jax3d.Path) -> models.NerfParams:
  text = path.read_text()
  return models.NerfParams.from_dict(ast.literal_eval(text))


def _write_variables(path: jax3d.Path, variables: Tree[jnp.ndarray]):
  checkpoints.save_checkpoint(ckpt_dir=path, target=variables, step=0)


def _read_variables(path: jax3d.Path) -> Tree[jnp.ndarray]:
  return checkpoints.restore_checkpoint(ckpt_dir=path, target=None, step=0)


def _write_density_grid(path, density_grid):
  with jax3d.Path(path).open('wb') as fh:
    jnp.save(fh, density_grid)


def _read_density_grid(path: jax3d.Path) -> jnp.ndarray:
  with jax3d.Path(path).open('rb') as fh:
    result = jnp.load(fh)
  return result
