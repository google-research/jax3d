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

"""Data loaders for 2D autoencoder."""

import dataclasses

import gin
import jax
import tensorflow as tf
import tensorflow_datasets as tfds


@gin.configurable()
@dataclasses.dataclass(frozen=True)
class TFDSImageDatasetReader():
  """Dataset reader wrapping TFDS image datasets."""
  dataset_name: str = gin.REQUIRED
  resolution: int = gin.REQUIRED
  batch_size: int = gin.REQUIRED

  split: str = "train"
  eval_fraction: float = 0.05
  shuffle_buffer_size: int = 1000

  def __iter__(self):
    """Creates an iterator over the dataset."""
    ds = tfds.load(self.dataset_name + f"/{self.resolution}")["train"]

    def filter_for_split(data):
      buckets = int(1.0 / self.eval_fraction)
      index = tf.strings.to_hash_bucket(data["image/filename"], buckets)
      if self.split == "train":
        return tf.not_equal(index, 0)
      else:
        return tf.equal(index, 0)

    ds = ds.filter(filter_for_split)

    def extract_image(data):
      return {
          "image_data": {
              "image": tf.cast(data["image"][None], tf.float32) / 255.0
          }
      }

    ds = ds.map(extract_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.repeat().shuffle(self.shuffle_buffer_size)
    ds = ds.batch(self.batch_size // jax.device_count(), drop_remainder=True)
    ds = ds.batch(jax.local_device_count())

    return ds.as_numpy_iterator()

  def load_summary_data(self, num_identities: int):
    """Load the data needed to render summary images."""
    ds = tfds.load(self.dataset_name + f"/{self.resolution}")["train"]

    def extract_image(data):
      return {
          "image_data": {
              "image": tf.cast(data["image"][None], tf.float32) / 255.0
          }
      }

    ds = ds.map(extract_image)
    data = next(ds.batch(num_identities).take(1).as_numpy_iterator())
    return data


@gin.configurable()
@dataclasses.dataclass(frozen=True)
class TiledMNISTDatasetReader():
  """Dataset reader wrapping TFDS image datasets."""
  resolution: int = gin.REQUIRED
  batch_size: int = gin.REQUIRED
  tile_factor: int = gin.REQUIRED

  split: str = "train"
  eval_fraction: float = 0.05
  shuffle_buffer_size: int = 1000
  digit_res: int = 16

  def __iter__(self):
    """Creates an iterator over the dataset."""
    ds = tfds.load("mnist")["train"]

    def extract_and_rescale(data):
      image = tf.cast(tf.image.resize(
          data["image"], (self.digit_res, self.digit_res),
          method=tf.image.ResizeMethod.AREA), tf.float32) / 255.0
      return tf.stack([image] * 3, axis=-1)

    ds = ds.map(
        extract_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    grid_width = self.resolution // self.digit_res
    inner_grid_width = int(grid_width // 2**self.tile_factor)
    ds = ds.batch(inner_grid_width**2, drop_remainder=True)

    def make_grid(images):
      grid = tf.reshape(images, (inner_grid_width, 1, inner_grid_width, 1,
                                 self.digit_res, self.digit_res, 3))
      grid = tf.tile(grid,
                     (1, 2**self.tile_factor, 1, 2**self.tile_factor, 1, 1, 1))
      grid = tf.reshape(
          grid, (grid_width, grid_width, self.digit_res, self.digit_res, 3))
      grid = tf.transpose(grid, (0, 2, 1, 3, 4))
      grid = tf.reshape(
          grid,
          (1, grid_width * self.digit_res, grid_width * self.digit_res, 3))
      return grid

    ds = ds.map(make_grid)

    def wrap(image):
      return {"image_data": {"image": image}}

    ds = ds.map(wrap, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.repeat().shuffle(self.shuffle_buffer_size)
    ds = ds.batch(self.batch_size // jax.device_count())
    ds = ds.batch(jax.local_device_count())

    return ds.as_numpy_iterator()

  def load_summary_data(self, num_identities: int):
    """Load the data needed to render summary images."""
    ds = tfds.load("mnist")["train"]

    def extract_and_rescale(data):
      image = tf.cast(tf.image.resize(
          data["image"], (self.digit_res, self.digit_res),
          method=tf.image.ResizeMethod.AREA), tf.float32) / 255.0
      return tf.stack([image] * 3, axis=-1)

    ds = ds.map(
        extract_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    grid_width = self.resolution // self.digit_res
    inner_grid_width = int(grid_width // 2**self.tile_factor)
    ds = ds.batch(inner_grid_width**2, drop_remainder=True)

    def make_grid(images):
      grid = tf.reshape(images, (inner_grid_width, 1, inner_grid_width, 1,
                                 self.digit_res, self.digit_res, 3))
      grid = tf.tile(grid,
                     (1, 2**self.tile_factor, 1, 2**self.tile_factor, 1, 1, 1))
      grid = tf.reshape(
          grid, (grid_width, grid_width, self.digit_res, self.digit_res, 3))
      grid = tf.transpose(grid, (0, 2, 1, 3, 4))
      grid = tf.reshape(
          grid,
          (1, grid_width * self.digit_res, grid_width * self.digit_res, 3))
      return grid

    ds = ds.map(make_grid)

    def wrap(image):
      return {"image_data": {"image": image}}

    ds = ds.map(wrap, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = next(ds.batch(num_identities).take(1).as_numpy_iterator())
    return data

