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

"""Generic model training script."""

from absl import app
from absl import flags
import gin
import jax
from jax3d.projects.generative.nerf import configs
import tensorflow as tf

flags.DEFINE_string("experiment_name", "test_experiment", "An experiment name.")
flags.DEFINE_string("base_folder", None, "where to store ckpts and logs")
flags.mark_flag_as_required("base_folder")
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
flags.DEFINE_multi_string("gin_configs", (), "Gin config files.")
flags.DEFINE_multi_string("gin_search_paths", (),
                          "Paths to search for gin config files.")
flags.DEFINE_enum("mode", "train", ["train", "eval"],
                  "Either 'train' or 'eval'")
FLAGS = flags.FLAGS


def main(argv):
  del argv

  tf.config.experimental.set_visible_devices([], "GPU")

  gin.add_config_file_search_path(
      "third_party/py/jax3d/projects/generative/nerf")

  for search_path in FLAGS.gin_search_paths:
    gin.add_config_file_search_path(search_path)

  gin.parse_config_files_and_bindings(
      config_files=FLAGS.gin_configs,
      bindings=FLAGS.gin_bindings,
      skip_unknown=False)

  trainer = configs.ExperimentConfig().trainer(
      experiment_name=FLAGS.experiment_name, working_dir=FLAGS.base_folder)

  if FLAGS.mode == "train":
    trainer.train()
  elif FLAGS.mode == "eval":
    trainer.eval()


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
