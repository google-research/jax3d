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

import dataclasses

from absl.testing import absltest
import gin
import jax
from jax import numpy as jnp
from jax3d.projects.generative.nerf import configs
from jax3d.projects.generative.nerf import trainer
from jax3d.projects.generative.nerf import transforms


@gin.configurable('RandomDataLoader')
@dataclasses.dataclass()
class RandomDataLoader:
  """A class for providing random pixel data."""
  identity_batch_size: int = 2
  view_batch_size: int = 4
  pixel_batch_size: int = 16

  identity_count: int = 5
  max_views_per_identity: int = 10

  def __post_init__(self):
    self.rng_key = jax.random.PRNGKey(1234)

  def __iter__(self):
    return self

  def __next__(self):

    pixel_coordinates = jax.random.uniform(
        self.rng_key, (1, self.identity_batch_size, self.view_batch_size,
                       self.pixel_batch_size, 2),
        dtype=jnp.float32)
    gamma_rgb = jax.random.uniform(
        self.rng_key, (1, self.identity_batch_size, self.view_batch_size,
                       self.pixel_batch_size, 3),
        dtype=jnp.float32)
    weight = jax.random.uniform(
        self.rng_key, (1, self.identity_batch_size, self.view_batch_size,
                       self.pixel_batch_size, 1),
        dtype=jnp.float32)
    identity = jax.random.randint(
        self.rng_key, (1, self.identity_batch_size, self.view_batch_size),
        minval=0,
        maxval=self.identity_count)
    view_subindex = jax.random.randint(
        self.rng_key, (1, self.identity_batch_size, self.view_batch_size),
        minval=0,
        maxval=self.max_views_per_identity)

    camera = {
        'orientation':
            transforms.euler_to_rotation_matrix(
                jnp.zeros(
                    (self.identity_batch_size, 1, self.view_batch_size, 3))),
        'position':
            jnp.tile(
                jnp.array([0., 0., 0.5]),
                [self.identity_batch_size, 1, self.view_batch_size, 1]),
        'focal_length':
            300. * jnp.ones(
                (self.identity_batch_size, 1, self.view_batch_size, 1)),
        'principal_point':
            150. * jnp.ones(
                (self.identity_batch_size, 1, self.view_batch_size, 2)),
        'image_size':
            300 * jnp.ones(
                (self.identity_batch_size, 1, self.view_batch_size, 2),
                jnp.int32),
        'skew':
            jnp.zeros((self.identity_batch_size, 1, self.view_batch_size, 1)),
        'pixel_aspect_ratio':
            jnp.ones((self.identity_batch_size, 1, self.view_batch_size, 1)),
    }

    return {
        'pixel_coordinates': pixel_coordinates,
        'gamma_rgb': gamma_rgb,
        'weight': weight,
        'view_subindex': view_subindex,
        'identity': identity,
        'camera': camera,
    }


@gin.configurable
@dataclasses.dataclass(frozen=True)
class BasicTrainer(trainer.Trainer):

  def init_data_loader(self):
    return RandomDataLoader().__iter__()


class TrainerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    bindings = """
      import jax3d.projects.generative.nerf.trainer
      import jax3d.projects.generative.nerf.configs

      nerf.configs.ExperimentConfig.trainer = @BasicTrainer

      BasicTrainer.max_steps = 10
      BasicTrainer.log_every = 2
      BasicTrainer.save_every = 3
    """
    gin.parse_config(bindings)

  def test_basic_trainer(self):
    """Tests Trainer initialization."""

    basic_trainer = configs.ExperimentConfig().trainer(
        experiment_name='test', working_dir='/tmp/pre-training')
    basic_trainer.train()
    # self.assertEqual()

    # Test loading of the pre-trained checkpoint.
    finetuner = configs.ExperimentConfig().trainer(
        experiment_name='test-finetuning',
        working_dir='/tmp/finetuning',
        pre_trained_checkpoint='/tmp/pre-training/test/checkpoints',
        max_steps=20)

    finetuner.initialize_experiment_directory()
    checkpoint_state = finetuner.load_checkpoint(finetuner.init_state())
    self.assertEqual(checkpoint_state.step, 9)
    finetuner.train()


if __name__ == '__main__':
  absltest.main()
