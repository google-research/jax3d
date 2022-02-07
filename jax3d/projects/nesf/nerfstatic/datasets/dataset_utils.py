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

"""Shared utils among the dataset implementation."""

import collections
import contextlib
import dataclasses
import enum
import itertools
import re
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Type, TypeVar, Union
from unittest import mock

import chex
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.utils import camera_utils
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils import tree_utils
from jax3d.projects.nesf.utils.typing import f32, i32  # pylint: disable=g-multiple-import
import mediapy
import numpy as np
import skimage.transform
import tensorflow as tf

_T = TypeVar('_T')

# Load functions loading examples from one or more scenes.
_InMemoryLoadExamplesFn = Callable[..., Tuple[types.Batch, 'DatasetMetadata']]
_StreamingLoadExamplesFn = Callable[..., Tuple[tf.data.Dataset,
                                               List['DatasetMetadata']]]

_DATASET_REGISTER: Dict[str, 'RegisteredDataset'] = {}


@dataclasses.dataclass
class RegisteredDataset:
  """Structure containing dataset metadata.

  Attributes:
    name: dataset name
    make_examples_fn: Function which load and returns the examples. If
      `in_memory=True`, should return a `Batch` and a `DatasetMetadata`
      representing a single scene. If `in_memory=False`, should return
      `tf.data.Dataset` and a list of `DatasetMetadata` representing all
      scenes.
    in_memory: Whether the dataset is loaded in-memory or should be streamed.
    config_cls: The eventual class containing dataset specific params
  """
  name: str
  make_examples_fn: Union[_InMemoryLoadExamplesFn, _StreamingLoadExamplesFn]
  in_memory: bool = True
  config_cls: Optional[Type[gin_utils.ConfigurableDataclass]] = None


def register_dataset(dataset: RegisteredDataset) -> None:
  """Register the dataset."""
  assert dataset.name not in _DATASET_REGISTER
  _DATASET_REGISTER[dataset.name] = dataset


def find_registered_dataset(name: str) -> RegisteredDataset:
  """Returns the registered dataset."""
  registered_dataset = _DATASET_REGISTER.get(name.lower())
  if registered_dataset is None:
    raise ValueError(
        f'Unknown dataset {name}. Should be one of {_DATASET_REGISTER.keys()}')
  return registered_dataset


@dataclasses.dataclass
class DatasetMetadata:
  """Metadata for a single scene in a dataset."""
  # Mapping from ints to human-interpretable names for semantic category images.
  labels: List[str] = dataclasses.field(default_factory=list)

  # Camera parameters.
  cameras: Optional[camera_utils.Camera] = None

  # Human-interpretable identifier for this scene.
  scene_name: Optional[str] = None


class ExampleType(enum.Enum):
  """Output format of the dataset.

  Attributes:
    RAY: Each batch shape is (local_device_count, local_batch_size /
      local_device_count,...)
    IMAGE: Each batch is (h, w, ...)
  """
  RAY = enum.auto()
  IMAGE = enum.auto()


def prefetch(iterator: Iterable[_T], *, buffer_size) -> Iterator[_T]:
  """Pre-fetch the iterator (synchronously)."""
  iterator = iter(iterator)

  queue = collections.deque()
  # Prefetch buffer size to the queue
  for x in itertools.islice(iterator, buffer_size):
    queue.append(x)

  while queue:
    yield queue.popleft()

    # Eventually push the next element to the queue
    try:
      queue.append(next(iterator))
    except StopIteration:
      pass


def _make_view(batch_shape) -> types.Views:
  # Ids are not broadcasted, so is shape `()` when `(800, 800, 3)`
  id_shape = np.array(batch_shape[:-2])
  image_ids = ['img0.png'] * id_shape.prod()
  image_ids = np.array(image_ids).reshape(id_shape)

  return types.Views(
      rays=types.Rays(
          scene_id=np.zeros((*batch_shape, 1), dtype=np.int32),
          origin=np.random.ranf((*batch_shape, 3)).astype(np.float32),
          direction=np.random.ranf((*batch_shape, 3)).astype(np.float32),
      ),
      rgb=np.random.ranf((*batch_shape, 3)).astype(np.float32),
      depth=np.random.ranf((*batch_shape, 1)).astype(np.float32),
      semantics=np.zeros((*batch_shape, 1), dtype=np.int32),
      image_ids=image_ids,
  )


def make_examples(
    target_batch_shape,
) -> types.Batch:
  """Creates placeholder examples of a batch."""
  target_view = _make_view(target_batch_shape)
  return types.Batch(target_view=target_view)


@contextlib.contextmanager
def mock_sunds(examples: types.Batch):
  """Mock sunds to returns the dummy examples."""
  ds = tf.data.Dataset.from_tensor_slices(examples)

  def mock_streaming_make_examples_fn(**kwargs):
    del kwargs
    return ds, [DatasetMetadata()]

  with mock.patch.object(
      _DATASET_REGISTER['sunds'],
      'make_examples_fn',
      mock_streaming_make_examples_fn,
  ):
    yield


@chex.dataclass
class ImageSet:
  """Class for Images predicted by DeepLab."""
  # string identifier for the scene this image comes from.
  scene_name: str

  # string identifier for which image within a scene this image comes from.
  image_name: str

  # A glob filepattern one can use to find the files held by this ImageSet.
  glob_pattern: str

  # RGB
  rgb: f32['h w 3']
  rgb_ground_truth: f32['h w 3']
  semantic: i32['h w']
  semantic_ground_truth: i32['h w']


class DeepLabImageLoader:
  """For loading colorized semantic maps.

  Includes ground truth labels and RGB images.

  """

  def __init__(self, xm_work_unit_dir: j3d.Path):
    self.xm_work_unit_dir = xm_work_unit_dir
    self._image_dir = xm_work_unit_dir / 'vis' / 'segmentation_results'

  def step(self) -> int:
    return np.nan

  def load_all(self) -> List[ImageSet]:
    return tree_utils.parallel_map(lambda x: self[x], list(range(len(self))))

  def __getitem__(self, key: int) -> ImageSet:
    return ImageSet(
        scene_name='UNKNOWN_SCENE',
        image_name='UNKNOWN_IMAGE',
        glob_pattern=str(self._image_dir / f'{key:06d}_*.png'),
        rgb=np.full((256, 256, 3), np.nan),
        rgb_ground_truth=self._load_png(self._image_dir /
                                        f'{key:06d}_image.png'),
        semantic=self._load_png(self._image_dir / f'{key:06d}_prediction.png'),
        semantic_ground_truth=self._load_png(self._image_dir /
                                             f'{key:06d}_label.png'),
    )

  def __contains__(self, key: int) -> bool:
    return 0 <= key < len(self)

  def __iter__(self) -> Iterable[ImageSet]:
    for idx in range(len(self)):
      yield self[idx]

  def __len__(self) -> int:
    items = list(self._image_dir.glob('*.png'))
    assert len(items) % 4 == 0, len(items)
    return len(items) // 4

  def _load_png(self, path: j3d.Path) -> np.ndarray:
    image = mediapy.read_image(path)
    image = image / 255.0
    image = skimage.transform.resize(
        image, (256, 256), order=1, preserve_range=True)
    return image


class DeepLabSemanticMapLoader:
  """For loading raw semantic maps.

  Does not include ground truth semantic labels or RGB images.

  """

  def __init__(self, xm_work_unit_dir: j3d.Path):
    self.xm_work_unit_dir = xm_work_unit_dir
    self._image_dir = xm_work_unit_dir / 'vis' / 'raw_segmentation_results'

  def step(self) -> int:
    return np.nan

  def load_all(self) -> List[ImageSet]:
    return tree_utils.parallel_map(lambda x: self[x], list(range(len(self))))

  def __getitem__(self, key: int) -> ImageSet:
    filepath = self._filepaths[key]

    # Parse scene, image names
    match = re.search("b'(\d+)_rgba_(\d+)'.png$", filepath.name)  # pylint: disable=anomalous-backslash-in-string
    if not match:
      raise ValueError(filepath)
    scene_name, image_name = match.groups()

    return ImageSet(
        scene_name=scene_name,
        image_name=image_name,
        glob_pattern=str(filepath),
        rgb=np.full((256, 256, 3), np.nan),
        rgb_ground_truth=np.full((256, 256, 3), np.nan),
        semantic=self._load_png(filepath),
        semantic_ground_truth=np.full((256, 256, 3), np.nan),
    )

  def __contains__(self, key: int) -> bool:
    return 0 <= key < len(self)

  def __iter__(self) -> Iterable[ImageSet]:
    for idx in range(len(self)):
      yield self[idx]

  def __len__(self) -> int:
    return len(self._filepaths)

  def _load_png(self, path: j3d.Path) -> np.ndarray:
    """Load image from path."""
    image = mediapy.read_image(path)
    assert image.shape == (1024, 1024), image.shape

    # Take the top-left pixel of every 4x4 square. This isn't the smartest
    # thing to do, but it's a lot faster the code below.
    image = image[::4, ::4]

    # # We want to resize the image back to (256, 256). We do so by grouping
    # # 2x2 squares of pixels together and taking the most common semantic
    # # category from each.
    # image = np.reshape(image, (1024, 1024, 1))
    # image = block_to_depth(image, block_size=4)
    #
    #  # MultiShapeNet-13 has 14 semantic categories.
    # image = most_common_value_along_axis(image, max_value=14)

    return image

  @property
  def _filepaths(self) -> List[j3d.Path]:
    return list(self._image_dir.glob('*.png'))


def filter_images_per_scene(all_images: List[ImageSet],
                            scene_name: str) -> Dict[int, i32['h w']]:
  result = {}
  for image in all_images:
    if image.scene_name == scene_name:
      result[int(image.image_name)] = image.semantic
  return result
