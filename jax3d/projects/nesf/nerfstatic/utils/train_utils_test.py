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

"""Tests for jax3d.projects.nesf.nerfstatic.utils.train_utils."""

import pathlib

import chex
from flax import optim
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.nerf import utils
from jax3d.projects.nesf.nerfstatic.utils import train_utils
import numpy as np


def test_checkpoint_save_restore(tmp_path: pathlib.Path):

  save_dir = j3d.Path(tmp_path)

  # Mock checkpoint saving.
  ds_state_origin = np.random.PCG64().state['state']

  opt_config = optim.Momentum(learning_rate=0.1, beta=0.1)
  optimizer_ckpt = opt_config.create(target={'var1': 1.0, 'var2': 2.0})
  model_state_ckpt = utils.TrainState(optimizer=optimizer_ckpt)
  step = 0

  train_utils.save_checkpoints_for_process(
      model_state=model_state_ckpt,
      ds_state=ds_state_origin,
      step=step,
      save_dir=save_dir,
  )

  # Restore dataset state
  data_state_restored = train_utils.restore_ds_checkpoint_for_process(
      save_dir=save_dir)
  assert data_state_restored == ds_state_origin

  # Restore model
  optimizer = opt_config.create(target={'var1': 0.0, 'var2': 0.0})
  model_state = utils.TrainState(optimizer=optimizer)
  model_state_restored = train_utils.restore_opt_checkpoint(
      save_dir=save_dir,
      state=model_state,
  )
  chex.assert_tree_all_close(model_state_ckpt.optimizer.target,
                             model_state_restored.optimizer.target)
