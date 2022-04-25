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

"""Training binary."""

from absl import app
from absl import flags

from jax3d.projects.nesf.nerfstatic import train_lib
from jax3d.projects.nesf.nerfstatic import train_semantic_lib
from jax3d.projects.nesf.nerfstatic.utils import config as nerf_config


FLAGS = flags.FLAGS


def main(unused_argv):
  params = nerf_config.root_config_from_flags()

  xm_wid = None

  if params.train.mode == "TRAIN":
    train_lib.train(params=params, xm_wid=xm_wid)
  elif params.train.mode == "SEMANTIC":
    train_semantic_lib.train(params=params, xm_wid=xm_wid)
  else:
    raise ValueError(params.train.mode)


if __name__ == "__main__":
  app.run(main)
