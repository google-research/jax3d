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

"""Data loader for SunDs datasets."""

# TODO(noharadwan): Decide if we wanna keep or remove before code release.

import dataclasses
from typing import List, Optional, Tuple

from absl import logging
import jax
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.datasets import dataset_utils
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import Array
import numpy as np
import sunds
import tensorflow as tf
import tensorflow_datasets as tfds

_TensorTree = tfds.typing.TreeDict[Array[...]]

_TARGET_CAMERA_NAME = 'target'
_SCENES_METADATA_FEATURE = 'scenes'


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class SundsParams:
  """Sunds params.

  Attributes:
    name: The dataset name selector passed to `sunds.load` (e.g.
      `nerf_synthetic/lego`, `streetview_3d`, `my_dataset:2.0.0`)
    normalize_rays: Whether or not to normalize the rays.
    center_example: Whether or not to center each example. Centering will be
      performed based on the axis aligned bounding box of the target view
      frustrum and the origins of the input views.
    far_plane_for_centering: far plane to use when calculating frustrums.
    centering_jitter:  translational jitter to apply after centering.
  """
  name: str
  normalize_rays: bool = True
  center_example: bool = False
  far_plane_for_centering: float = 30.0  # meters for non-normalized Streetview
  centering_jitter: float = 3.0


def make_examples(
    data_dir: Optional[j3d.Path],
    *,
    split: str,
    example_type: dataset_utils.ExampleType,
    params: SundsParams,
) -> Tuple[tf.data.Dataset, List[dataset_utils.DatasetMetadata]]:
  """Load examples from the sub-directory.

  Works for datasets with and without "category_image" features. If
  "category_image" is found in any camera spec, it will be returned in the
  resulting batches. If it is not found, dummy values of the appropriate shape
  and dtype will be return instead.

  Args:
    data_dir: Directory containing the examples
    split: Split name to load.
    example_type: IMAGE or RAY mode.
    params: Pipeline params

  Returns:
    batch: Batch containing the full dataset
    all_metadata: Sequence of DatasetMetadata objects, one per scene.
  """
  # We explicitly disable conditional datasets
  is_conditional = False
  is_image = (example_type == dataset_utils.ExampleType.IMAGE)

  builder = sunds.builder(params.name, data_dir=data_dir)

  camera_specs = next(
      iter(builder.frame_builder.info.features['cameras'].values()))
  has_category = 'category_image' in camera_specs

  if params.normalize_rays and params.center_example:
    raise ValueError('normalize_rays and center_example are mutually exclusive')
  if params.center_example:
    normalize_rays = sunds.tasks.CenterNormalizeParams(
        far_plane=params.far_plane_for_centering,
        jitter=params.centering_jitter
    )
  else:
    normalize_rays = params.normalize_rays

  if is_conditional:
    # In conditional mode, keep images dims (flattened later)
    yield_mode = sunds.tasks.YieldMode.DICT
  elif is_image:
    yield_mode = sunds.tasks.YieldMode.IMAGE
  else:
    yield_mode = sunds.tasks.YieldMode.RAY

  # Load the frames
  task = sunds.tasks.Nerf(
      yield_mode=yield_mode,
      normalize_rays=normalize_rays,
      additional_camera_specs={
          # Include category if present in the dataset
          'category_image': has_category,
      },
      add_name=True,
  )
  ds = builder.as_dataset(
      split=split,
      task=task,
      shuffle_files=not dataset_utils.ExampleType.IMAGE,
  )

  # Reshape dict -> Batch
  ds_info: tfds.core.DatasetInfo = builder.frame_builder.info
  scene_names = _get_scene_names(ds_info)
  ds = ds.map(
      _dict_to_batch(  # pylint: disable=no-value-for-parameter
          scene_id_mapping=_SceneIdMapping(scene_names),
      ))

  # Construct string labels for each semantic category.
  category_labels: List[str] = []
  if has_category:
    category_labels = camera_specs['category_image'].names
    logging.info('Dataset contains semantic maps with %d semantic categories.',
                 len(category_labels))

  # Construct one DatasetMetadata object per scene.
  base_metadata = dataset_utils.DatasetMetadata(
      labels=category_labels, cameras=None, scene_name=None)
  if scene_names:
    all_ds_metadata = [
        dataclasses.replace(base_metadata, scene_name=scene_name)
        for scene_name in scene_names
    ]
  else:
    all_ds_metadata = [base_metadata]
  logging.info('Constructed DatasetMetadata objects for %d scene(s).',
               len(all_ds_metadata))

  return ds, all_ds_metadata


def load_all_examples(ds: tf.data.Dataset) -> types.Batch:
  """Load all examples in-memory."""
  # Might take a while. Could use
  all_exs = list(tfds.as_numpy(ds))
  return jax.tree_map(lambda *arrs: np.stack(arrs), *all_exs)


@sunds.utils.map_fn
def _dict_to_batch(
    ex: _TensorTree,
    *,
    scene_id_mapping: '_SceneIdMapping',
) -> types.Batch:
  """Converts `dict` -> `Batch`."""
  image_id = _make_image_id(ex=ex)
  scene_id = scene_id_mapping.lookup(ex['scene_name'])

  target_view = _make_view(ex, image_id=image_id, scene_id=scene_id)

  return types.Batch(
      target_view=target_view,
  )


def _make_image_id(
    *,
    ex: _TensorTree,
) -> Optional[tf.Tensor]:
  """Returns the sunds image id.

  Args:
    ex: Features for this camera.

  Returns:
    tf.Tensor(shape=[], dtype=tf.string). Unique identifier for this image.
  """
  if 'frame_name' not in ex:
    return None
  scene_name = ex['scene_name']
  frame_name = ex['frame_name']
  camera_name = ex['camera_name']
  return tf.strings.join([scene_name, frame_name, camera_name], separator='-')


def _make_view(
    ex_cam: _TensorTree,
    image_id: Optional[tf.Tensor] = None,
    scene_id: Optional[tf.Tensor] = None,
) -> types.Views:
  """Process example.

  Args:
    ex_cam: Features for this camera.
    image_id: shape=[] dtype=tf.string. Unique identifier for this camera.
    scene_id: shape=[] dtype=tf.int32. Integer identifier for this scene.
      Ascending from zero.

  Returns:
    types.Views instance representing the same content as `ex_cam`.
  """
  *batch_shape, _ = ex_cam['color_image'].shape  # pytype: disable=attribute-error

  # Create dummy arrays for values in the Batch that aren't defined. Downstream
  # code is not capable of handling "None" values.
  def _dummy_array(dtype, shape=(*batch_shape, 1)):
    return tf.zeros(shape=shape, dtype=dtype)

  # Each ray has the same scene_id.
  if scene_id is None:
    scene_id = _dummy_array(tf.int32)
  else:
    scene_id = tf.fill((*batch_shape, 1), scene_id)

  return types.Views(
      rays=types.Rays(
          scene_id=scene_id,
          origin=ex_cam['ray_origins'],
          direction=ex_cam['ray_directions'],
      ),
      depth=_dummy_array(tf.float32),
      rgb=tf.cast(ex_cam['color_image'], tf.float32) / 255.,
      semantics=ex_cam.get('category_image', _dummy_array(tf.int32)),
      image_ids=image_id,
  )


def _stack_cameras(ex_cams: _TensorTree) -> _TensorTree:
  """Stack all given camera together.

  Args:
    ex_cams: A `dict` camera_name -> camera fields.

  Returns:
    ex_cams: camera fields of all camera stacked together.
  """
  # Stack each fields togethers (color_image from camera0, 1, ...)
  # This assume the shape of all cameras is statically known and identical,
  # otherwise this code will fail.
  return {
      field_name: tf.stack(field_values, axis=0)
      for field_name, field_values in tfds.core.utils.zip_dict(
          *ex_cams.values())
  }


def _get_scene_names(ds_info: tfds.core.DatasetInfo) -> Optional[List[str]]:
  """Get list of scene names."""
  if ds_info.metadata:
    return ds_info.metadata.get(_SCENES_METADATA_FEATURE)
  return None


class _SceneIdMapping:
  """Scene name to scene ID mapping."""

  def __init__(self, scene_names: Optional[List[str]]):
    """Initializes _SceneIdMapping.

    Args:
      scene_names: List of scene names, in order.
    """
    if not scene_names:
      logging.info('Unable to construct scene_name -> scene_id mapping.')
      self._table = None
      return

    scene_ids = range(len(scene_names))
    self._table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(scene_names, scene_ids),
        default_value=-1)
    logging.info(
        'Constructed scene_name -> scene_id mapping with %d entries: %s...',
        len(scene_names), scene_names[0:3])

  def lookup(self, scene_name: tf.Tensor) -> Optional[tf.Tensor]:
    """Constructs an integer scene_id for this camera."""
    if self._table:
      return self._table.lookup(scene_name)
    return None


dataset_utils.register_dataset(
    dataset_utils.RegisteredDataset(
        name='sunds',
        make_examples_fn=make_examples,
        # TODO(epot): We should also support in-memory mode for small datasets
        in_memory=False,
        config_cls=SundsParams,
    ))
