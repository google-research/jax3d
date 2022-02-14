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

"""Provides PredictionImageLog.

This library provides PredictionImageLog, a library for managing model
predictions and ground truth images. One can use this library to store images
and serialize them to disk as a directory of PNGs.

"""

import json
import re
from typing import Any, Dict, List, Optional

from absl import logging
import jax
from jax import numpy as jnp
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.utils.typing import f32, i32  # pylint: disable=g-multiple-import
import kubric.file_io as kb_file_io
import numpy as np


DEFAULT_IMAGE_KEY = {
    "scene_name": "UNKNOWN_SCENE",
    "image_name": "UNKNOWN_IMAGE",
}


class PredictionImageLog:
  """Temporary storage for images for TensorBoard and disk.

  For each component we store a list of images, which are stacked right before
  logging to tensorboard.

  """

  def __init__(self):
    # Identifiers associated with each image.
    self._image_keys: List[Optional[str]] = []

    # Predicted RGB images.
    self._rgb: List[f32["h w 3"]] = []

    # Ground truth RGB images.
    self._rgb_ground_truth: List[f32["h w 3"]] = []

    # Color coded predicted semantic images.
    self._semantic: List[i32["h w 1"]] = []

    # Color coded ground truth semantic images.
    self._semantic_ground_truth: List[i32["h w 1"]] = []

  def append_images(
      self,
      image_key: Optional[str],
      rgb: f32["h w 3"],
      rgb_ground_truth: f32["h w 3"],
      semantic_logits: Optional[f32["h w c"]],
      semantic_ground_truth: Optional[i32["h w 1"]],
  ):
    """Append a set of images to the log.

    Args:
      image_key: String identifier for this frame. Format is,
        ${SCENE_NAME}_rgba_${IMAGE_NAME}. Use None if this information is
        missing.
      rgb: RGB prediction image. Values must be in [0, 1].
      rgb_ground_truth: RGB ground truth image. Values must be in [0, 1].
      semantic_logits: Optional semantic predictions. Contains per-class
        logits.
      semantic_ground_truth: Optional semantic prediction ground truth.
        Contains integer ID of the ground truth semantic class. Values must be
        in {0...255}.
    """
    self._image_keys.append(image_key)
    self._rgb.append(jax.device_get(jnp.clip(rgb, 0, 1)))
    self._rgb_ground_truth.append(
        jax.device_get(jnp.clip(rgb_ground_truth, 0, 1)))

    if semantic_logits is not None and semantic_logits.shape[-1]:
      # Semantic prediction.
      semantic = (jnp.argmax(semantic_logits, axis=-1)
                  .reshape(semantic_logits.shape[0:-1] + (1,)))
      self._semantic.append(jax.device_get(semantic))

      # Semantic ground truth.
      self._semantic_ground_truth.append(jax.device_get(semantic_ground_truth))

  @property
  def rgb(self) -> f32["n h w 3"]:
    return np.stack(self._rgb, axis=0)

  @property
  def rgb_ground_truth(self) -> f32["n h w 3"]:
    return np.stack(self._rgb_ground_truth, axis=0)

  @property
  def semantic(self) -> i32["n h w 1"]:
    return np.stack(self._semantic, axis=0)

  @property
  def semantic_ground_truth(self) -> i32["n h w 1"]:
    return np.stack(self._semantic_ground_truth, axis=0)

  @property
  def image_metadatas(self) -> List[Dict[str, Any]]:
    """Constructs image metadatas, one per image."""
    result = []
    for image_key in self._image_keys:
      # image_key isn't defined.
      if image_key is None:
        result.append(DEFAULT_IMAGE_KEY)
        continue

      # image_key is defined. See the following line of code describing how
      # image_key was built,
      # http://jax3d.projects.nesf/nerfstatic/datasets/klevr.py;l=235;rcl=408850085
      match = re.search(r"^(\d+)_rgba_(\d+)$", image_key)
      if not match:
        logging.warn("Failed to parse image key: %s", image_key)
        result.append(DEFAULT_IMAGE_KEY)
        continue

      scene_name, image_name = match.groups()
      result.append({
          "scene_name": scene_name,
          "image_name": image_name,
      })
    return result

  def write_images_to_disk(self, output_dir: j3d.Path):
    """Writes all images to an output directory.

    Args:
      output_dir: Directory to write all PNGs to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    kb_file_io.write_rgb_batch(
        self.rgb,
        output_dir,
        file_template="rgb_{:05d}.png")
    kb_file_io.write_rgb_batch(
        self.rgb_ground_truth,
        output_dir,
        file_template="rgb_ground_truth_{:05d}.png")
    kb_file_io.write_segmentation_batch(
        self.semantic.astype(np.uint32),
        output_dir,
        file_template="segmentation_{:05d}.png")
    kb_file_io.write_segmentation_batch(
        self.semantic_ground_truth.astype(np.uint32),
        output_dir,
        file_template="segmentation_ground_truth_{:05d}.png")
    (output_dir / "metadata.json").write_text(
        json.dumps(self.image_metadatas))
