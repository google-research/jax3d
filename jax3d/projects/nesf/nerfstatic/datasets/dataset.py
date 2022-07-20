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

"""Dataset main class."""


import dataclasses
import os
import re
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

import einops
from etils import etqdm as tqdm
import jax
import jax.numpy as jnp
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.datasets import dataset_utils
from jax3d.projects.nesf.nerfstatic.datasets import klevr  # pylint: disable=unused-import
from jax3d.projects.nesf.nerfstatic.datasets import scene_understanding  # pylint: disable=unused-import
from jax3d.projects.nesf.nerfstatic.datasets.dataset_utils import ExampleType
from jax3d.projects.nesf.nerfstatic.models import model_utils
from jax3d.projects.nesf.nerfstatic.utils import gin_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import Tree
import numpy as np
import sunds
import tensorflow as tf
import tensorflow_datasets as tfds

# pylint: disable=unused-import,g-bad-import-order
# pylint: enable=unused-import,g-bad-import-order

# Import datasets for registration
# State of the dataset pipeline (in in-memory mode).
# Internally, this correspond to the numpy random state used in to generate
# batches. Currently is 2 128-bit unsigned numbers of `np.random.PCG64()`.
DsState = Tree[Any]

# Scene selector can be:
# * int: Select scene `[0, n-1]`
# * str: np-like indexing (e.g. ':10', '5:', '3:10')
# * seq: List or tuple of ints `[1, 2, 5]`
SceneSelection = Union[int, str, Sequence[int]]


@dataclasses.dataclass(eq=True, frozen=True)
class BatchSize:
  """Batch size util.

  Attributes:
    total: The total batch size across all host/devices
    per_process: The batch size for a single host
    per_device: The batch size for a single device in a single host
  """
  total: int
  per_process: int
  per_device: int

  def __init__(self, global_batch_size: int):
    # Use setattr to bypass the frozen=True
    super().__setattr__('total', global_batch_size)
    super().__setattr__('per_process', global_batch_size // jax.process_count())
    super().__setattr__('per_device',
                        self.per_process // jax.local_device_count())

    if self.total and self.total % jax.device_count() != 0:
      raise ValueError(
          f'Batch size ({self.total}) must be divisible by the number '
          f'of devices ({jax.device_count()}).')


class BatchSizeField(j3d.utils.DataclassField[int, BatchSize]):
  """Field which normalize batch size."""

  def __init__(self, total: int):
    """Constructor.

    Args:
      total: The total batch size across all host/devices. This is the default
        value used if `DatasetParams.batch_size` is not provided.
    """
    self._total = total
    super().__init__()

  def _default(self) -> int:  # pytype: disable=signature-mismatch
    # Lazy construct the field to avoid `jax` calls at import time, before
    # `absl.main` is called.
    return self._total

  def _validate(self, value: int) -> BatchSize:
    return BatchSize(value)


@gin_utils.dataclass_configurable
@dataclasses.dataclass
class DatasetParams:
  """Dataset Params.

  Attributes:
    data_dir: Input data directory.
    dataset: The type of dataset to feed nerf.
    batch_size: The number of rays in a mini-batch (for training).
    specific_params: Params specific to the selected dataset (e.g. KlevrParams)
    factor: The downsample factor of images, 0 for no downsample.
    ray_shuffle_buffer_size: size of ray shuffle buffer
    num_input_ray_samples: If set, sample this many rays from input view for
      RAY mode. Will be ignored in IMAGE mode.
    crop_views: if specified crop the input or target images by this
      amount. Format is (top, bottom, left, right).
      Value in pixels to be cropped, e.g. (0, 0, 0, 0) is no cropping.
    train_scenes: Number of scenes to use for training. If `None`, data_dir
      should point to the specific scene otherwise, data_dir should contains
      `0/`, `1/`,... sub-dirs. If `int`, scenes `[0, train_scenes[` will be used
      for training
      If `str`, should be np-like indexing (e.g. ':10', '3:7')
    eval_scenes: Number of scenes to use for eval. If `None`, the same value
      than train_scenes will be used If int, scenes `[0, eval_scenes[` will be
      used for eval.
      If `str`, should be np-like indexing (e.g. ':10', '5:', '3:10')
    novel_scenes: Set of scene used in the semantic generalization mode.
      Like `train_scenes` can be `int` of `str`. Scene selected here will be
      used for both split='train' and 'eval' when
      `get_dataset(is_novel_scenes=True)`.
    ignore_scenes: Set of scenes to ignore in train, eval, and novel scenes.
      Use this if some scenes should be ignored due to upstream errors.
    num_scenes_per_batch: When set, a set of scenes are sampled at random, then
      rays from those scenes are sampled.
    max_num_train_images_per_scene: When set, limits the maximum number of
      images to load from each scene for model training and "eval_train".
    max_num_test_images_per_scene: When set, limits the maximum number of
      test images to load for "eval_test".
    eval_novel_train_image_idxs: Which image indices to use for
      "eval_novel_train". Index is with respect to
      metadata["split_ids"]["train"].
    eval_novel_test_image_idxs: Which image indices to use for
      "eval_novel_test". Index is with respect to
      metadata["split_ids"]["test"].
    enable_sqrt2_buffer: If set, the scene's bounding box will be increased by
      a factor of sqrt(2) along the x-axis and y-axis. For use with random
      scene rotations.
    enable_mipnerf: Enables Mip-NeRF mode. Currently, this will only populate
      the rays.base_radius fields of the data points.
    pseudo_semantic_labels_path: If set, loads the semantic labels for training
      from the given path.
  """
  data_dir: j3d.Path = j3d.utils.PathField()  # pytype: disable=annotation-type-mismatch
  dataset: str = 'klevr'
  batch_size: BatchSize = BatchSizeField(4096)  # pytype: disable=annotation-type-mismatch

  specific_params: Optional[gin_utils.ConfigurableDataclass] = None

  # TODO(epot): Should migrate dataset specific fields into their subclass.
  factor: int = 4
  ray_shuffle_buffer_size: int = 684 * 456 * 400
  num_input_ray_samples: Optional[int] = None
  crop_views: Optional[Tuple[int, int, int, int]] = None

  # Args to control the scenes to use in _examples_loader:
  train_scenes: Optional[SceneSelection] = None
  eval_scenes: Optional[SceneSelection] = None
  novel_scenes: Optional[SceneSelection] = None
  ignore_scenes: Optional[SceneSelection] = None

  num_scenes_per_batch: Optional[int] = None
  max_num_train_images_per_scene: Optional[int] = None
  max_num_test_images_per_scene: Optional[int] = None

  # Which image indices to use for each scene. This index is with respect to
  # the "split_ids" field in metadata.json.
  eval_novel_train_image_idxs: Optional[Sequence[int]] = None
  eval_novel_test_image_idxs: Optional[Sequence[int]] = None

  enable_sqrt2_buffer: bool = False
  enable_mipnerf: bool = False  # Whether to compute ray radii for MipNeRF.

  # If set, uses pseudo labels for semantic training.
  pseudo_semantic_labels_path: Optional[str] = None

  def __post_init__(self):
    # Maybe a cleaner way would be to have each dataset to be
    # a subclass of DatasetParams and use `ConfigModel.dataset = 'SundsParams'`
    # inside gin (like for the model class)
    # An advantage of this solution is that we are guarantee that
    # `DatasetParams` are never used in dataset-specific functions.

    # Initialize the dataset specific params
    registered_dataset = dataset_utils.find_registered_dataset(self.dataset)
    if self.specific_params is None and registered_dataset.config_cls:
      self.specific_params = registered_dataset.config_cls()


def get_dataset(
    split: str,
    args: DatasetParams,
    model_args: model_utils.ModelParams,
    example_type: ExampleType,
    ds_state: Optional[DsState] = None,
    *,  # TODO(epot): Make other args kwargs
    is_novel_scenes: bool = False,
) -> 'DatasetIterable':
  """Returns the dataset.

  Args:
    split: Split to load (train or test)
    args: Dataset arguments
    model_args: Model arguments
    example_type: ExampleType.RAY or ExampleType.IMAGE
    ds_state: State used to set e.g. np.random.RandomState instance state (When
      set to None, a default state value is used.)
    is_novel_scenes: Whether or not in novel scene evaluation mode.

  Returns:
    DatasetIterable object, initialized according to provided arguments.

  """
  registered_dataset = dataset_utils.find_registered_dataset(args.dataset)

  # TODO(epot): When dataset won't fit in memory, we could batch multiple
  # images, flatten batch dimention, then apply ds.shuffle on the flat_map
  # dataset ? Need to be carefull on determinism on multi-tpu workers

  # Select train/test examples from the different scenes
  if registered_dataset.in_memory:
    all_exs, all_metadata = _in_memory_examples_loader(
        registered_dataset=registered_dataset,
        split=split,
        args=args,
        # Blender-specific kwargs
        background=model_args.background,
        factor=args.factor,
        is_novel_scenes=is_novel_scenes,
    )
    ds = _make_in_memory_dataset(  # Create the tf.data.Dataset object
        all_exs,
        batch_size=args.batch_size,
        example_type=example_type,
        ds_state=ds_state,
        args=args,
    )
  else:
    assert not is_novel_scenes
    assert not args.max_num_train_images_per_scene
    assert not args.max_num_test_images_per_scene
    ds, all_metadata = _streaming_example_loader(
        registered_dataset=registered_dataset,
        split=split,
        example_type=example_type,
        args=args,
    )

  # Create the dataset iterable
  return DatasetIterable(
      ds=ds,
      all_metadata=all_metadata,
      example_type=example_type,
  )


def _in_memory_examples_loader(
    *,
    registered_dataset: dataset_utils.RegisteredDataset,
    split: str,
    args: DatasetParams,
    is_novel_scenes: bool,
    **load_scene_kwargs,
) -> Tuple[List[types.Batch], List[dataset_utils.DatasetMetadata]]:
  """Load examples from the requested scenes.

  Args:
    registered_dataset: The dataset to load
    split: Split to load (train or test)
    args: Dataset arguments
    is_novel_scenes: Whether or not in novel scene evaluation mode.
    **load_scene_kwargs: Kwargs forwarded to the `_DS_TO_LOADER` function.

  Returns:
    examples: List of all examples as a single Batch per scene, all in-order.
    dataset_metadata: List of all dataset metadata (e.g. semantic labels), one
      per scene.
  """
  if split not in ('train', 'test'):
    raise ValueError(f'Unknown split: {split}')

  # Get the number of scenes
  scene_range = _get_scene_range(
      eval_scenes=args.eval_scenes,
      train_scenes=args.train_scenes,
      novel_scenes=args.novel_scenes,
      ignore_scenes=args.ignore_scenes,
      split=split,
      is_novel_scenes=is_novel_scenes,
  )

  # Load all scenes
  all_exs: List[types.Batch] = []
  all_metadata: List[dataset_utils.DatasetMetadata] = []

  if scene_range is None:  # Single scene case
    all_data_dirs = [args.data_dir]
  else:
    all_data_dirs = [args.data_dir / str(i) for i in scene_range]

  # Decide on which image indices to load.
  load_split, image_idxs = {
      ('train', False): ('train', args.max_num_train_images_per_scene),
      ('test', False): ('test', args.max_num_test_images_per_scene),
      ('train', True): ('train', args.eval_novel_train_image_idxs),
      ('test', True): ('test', args.eval_novel_test_image_idxs),
  }[(split, is_novel_scenes)]

  if args.pseudo_semantic_labels_path is not None:
    deeplab_semantic_image_loader = dataset_utils.DeepLabSemanticMapLoader(
        j3d.Path(args.pseudo_semantic_labels_path))
    semantic_images = deeplab_semantic_image_loader.load_all()
  else:
    semantic_images = None

  # Warning: It's important that scene ids start at 0. So the selected
  # scene ids and file path folder name may not match
  for scene_id, data_dir in enumerate(tqdm.tqdm(all_data_dirs)):
    if semantic_images is not None:
      scene_semantic_images = dataset_utils.filter_images_per_scene(
          semantic_images, os.path.basename(data_dir))
      load_scene_kwargs['scene_semantic_images'] = scene_semantic_images
    scene_exs, scene_metadata = registered_dataset.make_examples_fn(
        data_dir=data_dir,
        split=load_split,
        params=args.specific_params,
        image_idxs=image_idxs,
        enable_sqrt2_buffer=args.enable_sqrt2_buffer,
        **load_scene_kwargs,
    )

    # In-memory datasets load one scene at a time.
    if not isinstance(scene_metadata, dataset_utils.DatasetMetadata):
      raise ValueError(
          f'Unexpected type for scene_metadata={scene_metadata}. '
          f'make_examples_fn() should have returned an instance of '
          f'DatasetMetadata. Is this a bug?')

    # Add the scene ids to the target view rays
    scene_rays = scene_exs.target_view.rays
    scene_rays = scene_rays.replace(
        scene_id=np.full(
            (*scene_rays.batch_shape, 1),
            scene_id,
            dtype=np.int32,
        ))
    scene_exs.target_view = scene_exs.target_view.replace(rays=scene_rays)

    # For MipNeRF, add ray base radii.
    if args.enable_mipnerf:
      scene_exs = _add_base_radii(scene_exs)

    all_exs.append(scene_exs)
    all_metadata.append(scene_metadata)
  return all_exs, all_metadata


def _streaming_example_loader(
    *,
    registered_dataset: dataset_utils.RegisteredDataset,
    split: str,
    args: DatasetParams,
    example_type: ExampleType,
) -> Tuple[tf.data.Dataset, List[dataset_utils.DatasetMetadata]]:
  """Constructs the streaming data pipeline.

  Args:
    registered_dataset: The dataset to load
    split: Split to load (train or test)
    args: Dataset params
    example_type: RAY or IMAGE mode

  Returns:
    ds: The `tf.data.Dataset` which yield the examples
    metadata: Additional dataset metadata (e.g. semantic labels) for each scene.
  """
  # Streaming datasets automatically contains all scenes. There is no option to
  # filter.
  (ds, all_metadata) = registered_dataset.make_examples_fn(
      data_dir=args.data_dir,
      split=split,
      example_type=example_type,
      params=args.specific_params)

  # all_metadata should be a list of DatasetMetadata objects.
  if (not isinstance(all_metadata, list) or not all([
      isinstance(metadata, dataset_utils.DatasetMetadata)
      for metadata in all_metadata
  ])):
    raise ValueError(f'all_metadata should be a List[DatasetMetadata]. Found: '
                     f'{all_metadata}. Is this a bug?')

  # TODO(duckworthd): Verify that DatasetParams.(train|eval)_scenes matches the
  # scenes described by `all_metadata`.

  # We expect `make_examples_fn` to return the following batch_shape:
  # * RAY: target_view=()
  # * IMAGE: target_view=(h, w)

  if args.crop_views:
    ds = ds.map(_crop_views(crop=args.crop_views))   # pylint: disable=no-value-for-parameter

  # For MipNeRF, add ray base radii.
  if args.enable_mipnerf:
    ds = ds.map(_add_base_radii)

  # Shuffle and batch dataset
  if example_type == ExampleType.RAY:
    # Image ids are not used in RAY mode, so remove them
    ds = ds.map(_pop_image_ids)

    # Shuffle rays
    ds = ds.shuffle(args.ray_shuffle_buffer_size)

    # Repeat dataset forever.
    ds = ds.repeat()

    # Batch as (num_local_devices, batch_size_per_device, ...). Set
    # drop_remainder=True to keep shapes fully-defined.
    ds = ds.batch(args.batch_size.per_device, drop_remainder=True)
    ds = ds.batch(jax.local_device_count(), drop_remainder=True)

  # After the transformations, batch_shape is:
  # * RAY: target_view=(device_count, bs_per_device)
  # * IMAGE: target_view=(h, w)

  # Add the dummy ds_state
  ds = ds.map(lambda x: (None, x))
  return ds, all_metadata


def _get_scene_range(
    *,
    train_scenes: Optional[SceneSelection],
    eval_scenes: Optional[SceneSelection],
    novel_scenes: Optional[SceneSelection],
    ignore_scenes: Optional[SceneSelection],
    split: str,
    is_novel_scenes: bool,
) -> Optional[List[int]]:
  """Returns the scenes to use (and perform validations)."""
  # Set eval_scene == train_scene if not defined
  if eval_scenes is None:
    eval_scenes = train_scenes
  if is_novel_scenes and novel_scenes is None:
    raise ValueError('novel_scenes should be set when `is_novel_scenes=True`')

  # Get the number of scenes
  if (train_scenes is None) != (eval_scenes is None):
    raise ValueError(
        'Both or none of train_scenes and eval_scenes should be set. Got: '
        f'eval_scenes={eval_scenes}, train_scenes={train_scenes}')

  # TODO(epot): Forward number of scene to allow `5:`
  train_range = make_scene_range(train_scenes)
  eval_range = make_scene_range(eval_scenes)
  novel_range = make_scene_range(novel_scenes)
  ignore_range = make_scene_range(ignore_scenes)

  # Remove ignore scenes
  if ignore_range:
    train_range = _remove_ignore_scenes(train_range, ignore_range)
    eval_range = _remove_ignore_scenes(eval_range, ignore_range)
    novel_range = _remove_ignore_scenes(novel_range, ignore_range)

  # Perform additional validation
  if novel_scenes is not None:
    if train_scenes is None or novel_range is None:
      raise ValueError(
          'When is_novel_scenes=True, both train and novel should be set. Got: '
          f'{train_scenes!r} vs novel_scenes={novel_scenes!r}')
    if eval_range != train_range:
      raise ValueError(
          'When is_novel_scenes=True, train and eval scene should be identical. '
          f'Got: train_scenes={train_scenes!r} vs eval_scenes={eval_scenes!r}')
    if set(train_range) & set(novel_range):
      raise ValueError(
          'Train and novel scenes should not overlap. Got: train_scenes='
          f'{train_scenes!r} vs novel_scenes={novel_scenes!r}')

  if is_novel_scenes:  # novel scenes has the same train and test split
    return novel_range
  elif split == 'train':
    return train_range
  else:
    return eval_range


def make_scene_range(
    scene_selection: Optional[SceneSelection],
) -> Optional[List[int]]:
  """Returns the range of scene matching the scene selection."""
  if scene_selection is None:
    return None
  elif isinstance(scene_selection, int):
    return list(range(scene_selection))
  elif isinstance(scene_selection, str):
    match = re.fullmatch(r'(\d*):(\d+)', scene_selection)
    if not match:
      raise ValueError(
          f'Invalid scene selection pattern: {scene_selection}. Should be `:n` '
          'or `n:k`')
    start = int(match.group(1) or 0)
    end = int(match.group(2))
    if end <= start:
      raise ValueError(f'Invalid scene selection pattern: {scene_selection}. '
                       f'{end} <= {start}')
    return list(range(start, end))
  elif isinstance(scene_selection, (tuple, list)):
    return list(scene_selection)
  else:
    raise TypeError(f'Invalid scene selector type: {scene_selection!r}')


def _remove_ignore_scenes(
    scenes: Optional[List[int]],
    ignore: Optional[List[int]],
) -> Optional[List[int]]:
  """Remove entries from 'scenes' that also appear in 'ignore'."""
  if scenes is None or ignore is None:
    return scenes
  ignore = set(ignore)
  return [s for s in scenes if s not in ignore]


def num_scenes(scene_selection: Optional[SceneSelection]) -> int:
  """Returns the number of scenes matching the selection.

  Example: `num_scenes('2:6') == 4`).

  Args:
    scene_selection: Same format as DatasetParams.train_scenes

  Returns:
    num_scenes
  """
  scenes_ids = make_scene_range(scene_selection)
  if scenes_ids is None:
    return 1
  else:
    return len(scenes_ids)


def _shuffle_deterministically(ex: np.ndarray) -> np.ndarray:
  # Use same seed for all fields
  rng = np.random.RandomState(178)
  return rng.permutation(ex)


class DatasetIterable:
  """Dataset class which contains the input pipeline.

  DatasetIterable has 2 modes:

  * `ExampleType.RAY`: Infinite iterable yielding random rays from all images.
    Determinism depends on the dataset type (streaming or in-memory).
    In this mode, each host/process yield different examples.
  * `ExampleType.IMAGE`: Finite iterator yielding all example images once.
    Iterating twice on the iterable will yield images **in the same order**.

  Additionally, the `ds` can be of 2 types:

  * In memory: The full examples are pre-loaded in memory. This allow to have
    deterministic and pre-emptable pipeline.
  * Streaming: The examples are read from TFRecord proto. There is no
    determinism in this mode.

  Warning: The state yield by the dataset will correspond to the next example
  yield. Thus restoring the state saved at the step 20 will yield the example
  21.

  """

  def __init__(
      self,
      ds: tf.data.Dataset,
      *,
      all_metadata: Sequence[dataset_utils.DatasetMetadata] = (),
      example_type: ExampleType,
  ):
    """Initialize the dataset iterator."""
    self._all_metadata = list(all_metadata)
    self._example_type = example_type

    # Create the input pipeline
    ds = ds.prefetch(tf.data.AUTOTUNE)
    self._ds = ds

  def __iter__(self) -> Iterator[Tuple[Optional[DsState], types.Batch]]:
    """Returns the dataset converted back to dataclasses."""
    ds = tfds.as_numpy(self._ds)

    # TODO(epot): Some values should not be put on device (but kept on CPU
    # as numpy array):
    # * tf.string tensors (e.g. image id) (as not supported by Jax)
    # * np.random state (as only used on CPU, so create unecessery TPU<>host
    #   overhead)
    # Logic could be injected here to only convert the required values

    # Prefetch to device
    # * RAY examples have shape `(num_local_devices, batch_size_per_device, c)`
    # * IMAGE examples have shape `(h, w, c)`
    # In RAY mode, each device receive a slice of data (data parallelism). In
    # eval, the sharding (for the pmap) is done inside the `render_fn`.
    if self._example_type == ExampleType.RAY:
      # Convert np to `ShardedDeviceArray`
      # Only send `Batch` to TPU, but not the `ds_state`
      ds = map(lambda ex: (ex[0], _to_devices_sharded(ex[1])), ds)
    else:
      # Should use `jax.device_put` or `jax.device_put_replicated` instead ?
      # Convert np.array into `DeviceArray`
      ds = map(lambda ex: (ex[0], _to_device(ex[1])), ds)

    # Pre-load 2 batch on device (when the element x is yielded, the element
    # x+1 is already been pre-loaded). Jax already lazy-executes everything,
    # so this should only affect blocking operations such as TF-summaries and
    # print statements
    ds = dataset_utils.prefetch(ds, buffer_size=2)
    return ds

  def peek(self) -> Tuple[None, types.Batch]:
    """Returns a dummy batch."""
    # TODO(epot): Update callers to only use types.Batch
    _, ex = self._ds.element_spec
    # Remove the image id (as not supported by Jax)
    # Use `types_like` as hack to convert NoneTensorSpec() to `None`, otherwise
    # `pop_image_id_stateless()` crash as `input_views` is `NoneTensorSpec()`
    ex = j3d.types_like(ex)
    ex, _ = ex.pop_image_id_stateless()
    return None, j3d.zeros_like(ex)

  @j3d.utils.cached_property
  def semantic_labels(self) -> List[str]:
    """Returns the semantic labels."""
    # No metadata provided
    if not self._all_metadata:
      return []

    # Validate that labels are consistent across scenes
    first_metadata, *all_rest_metadata = self._all_metadata
    for rest_metadata in all_rest_metadata:
      if first_metadata.labels != rest_metadata.labels:
        raise ValueError(
            f'Inconsistent labels across scenes: {first_metadata.labels} '
            f'vs {rest_metadata.labels}.')

    return first_metadata.labels

  @property
  def all_metadata(self) -> Sequence[dataset_utils.DatasetMetadata]:
    """Metadata for each scene_id."""
    return self._all_metadata


def _concat_examples(examples: List[types.Batch]) -> types.Batch:
  """Merges list of examples by concatenation of all features along axis 0."""
  return jax.tree_map(lambda *arrs: np.concatenate(arrs), *examples)


def _stack_examples(examples: List[types.Batch]) -> types.Batch:
  """Stacks list of examples on axis 0 for all features."""
  return jax.tree_map(lambda *arrs: np.stack(arrs), *examples)


def _make_in_memory_dataset(
    examples: List[types.Batch],
    *,
    batch_size: Optional[BatchSize],
    example_type: ExampleType,
    ds_state: Optional[DsState],
    args: DatasetParams,
) -> tf.data.Dataset:
  """Create a `tf.data.Dataset` pipeline from in-memory examples.

  In-memory datasets have the following advantages:

   * Are fully deterministic
   * Are fully random (no shuffle buffer size limitation)

  Args:
    examples: List of pre-loaded examples from each scene, as a single batch.
    batch_size: The batch size
    example_type: Example mode
    ds_state: Dataset state to save/restore the data pipeline.
    args: Dataset parameters.

  Returns:
    ds: The dataset object
  """
  if example_type == ExampleType.RAY:
    if batch_size is None:
      raise ValueError('batch_size should be defined when RAY mode')
    # Image ids are only used in IMAGE mode, so remove them
    # from the examples in RAY mode.
    # Rational: `str` tensors are not supported by Jax, and broadcasting
    # is wasteful.
    examples = [ex.pop_image_id_stateless()[0] for ex in examples]

  gen_batch_kwargs = {
      'examples': examples,
      'args': args,
      'ds_state': ds_state,
      'example_type': example_type,
      'batch_size': batch_size,
  }
  gen_batch = _gen_batch_unconditional(**gen_batch_kwargs)

  ds = tf.data.Dataset.from_generator(
      gen_batch,
      output_signature=j3d.tensor_spec_like(next(gen_batch())),
  )
  if example_type == ExampleType.RAY:
    ds = ds.map(lambda ds_state, ex: (ds_state, _shard(ex)))
  return ds


_GenBatchFn = Callable[[], Iterator[Tuple[Optional[DsState], types.Batch]]]


def _get_random_generator(ds_state: Optional[DsState]) -> np.random.Generator:
  """Returns the random generator, initialized with the state."""
  # Use SeedSequence to get randomly independent seed for each host
  seed_seq = np.random.SeedSequence(7263, spawn_key=(jax.process_index(),))
  bit_generator = np.random.PCG64(seed_seq)
  if ds_state is not None:  # Eventually restore the random state
    state = bit_generator.state
    state['state'] = to_ds_state_int(ds_state)
    bit_generator.state = state

  rng = np.random.Generator(bit_generator)
  return rng


def _gen_batch_unconditional(
    examples: List[types.Batch],
    args: DatasetParams,
    ds_state: Optional[DsState],
    example_type: ExampleType,
    batch_size: BatchSize,
) -> _GenBatchFn:
  """Returns gen_batch generator for datasets without target_views.

  Args:
    examples: List of per-scene datasets. Each batch's per-pixel Tensors has
      shape [num_images, height, width, ...].
    args: Dataset hyperparameters.
    ds_state: Optional dataset state for deterministic training.
    example_type: IMAGE or RAY.
    batch_size: Controls batch size.

  Returns:
    A function for generating batches from the dataset. Includes dataset state
      and the batch itself. Batch size varies as follows,

      +------------------------------+------------------------+
      |       Condition              |        batch shape     |
      +------------------------------+------------------------+
      | example_type == IMAGE        | [height, width]        |
      +------------------------------+------------------------+
      | example_type == RAY and      |                        |
      | args.num_scenes_per_batch is | [num_rays_per_process] |
      | NOT set.                     |                        |
      +------------------------------+------------------------+
      | example_type == RAY and      |                        |
      | args.num_scenes_per_batch is | [num_scenes_per_batch, |
      | set.                         |  num_rays_per_process] |
      +------------------------------+------------------------+
  """

  if example_type == ExampleType.IMAGE:
    # Concatenate all scenes together
    examples = _concat_examples(examples)
    # Shuffle images across scenes
    examples = jax.tree_map(_shuffle_deterministically, examples)

  elif example_type == ExampleType.RAY:
    # Merge (num_images, h, w) dimensions together.
    examples = jax.tree_map(
        lambda t: einops.rearrange(t, 'b h w ... -> (b h w) ...'),
        examples
    )
    if args.num_scenes_per_batch:
      # Select rays from a random subset of scenes.
      next_batch_fn = _next_batch_rays_from_scenes(examples, args, batch_size)
    else:  # args.num_scenes_per_batch is False
      next_batch_fn = _next_batch_rays_all_scenes(examples, args, batch_size)
  else:
    raise ValueError(f'Unrecognized ExampleType: {example_type}')

  def _gen_batch() -> Iterator[Tuple[Optional[DsState], types.Batch]]:
    """Yields batch."""
    rng = _get_random_generator(ds_state)
    if example_type == ExampleType.RAY:
      # TODO(epot): Should permute examples once per epoch, then emit batches.
      while True:
        batch_ex = next_batch_fn(rng)
        yield (
            to_ds_state_bytes(rng.bit_generator.state['state']),
            batch_ex,
        )
    elif example_type == ExampleType.IMAGE:
      # Yields individual images
      for ex in j3d.tree.unzip(examples):
        yield None, ex

  return _gen_batch


def _next_batch_rays_from_scenes(examples: List[types.Batch],
                                 args: DatasetParams,
                                 batch_size: BatchSize):
  """Constructs next_batch() function s.t. rays are chosen from K scenes."""
  # Outputs are of shape [num_scenes_per_batch, num_rays_per_scene, ...]

  # Shuffle rays within each scene.
  examples = [jax.tree_map(_shuffle_deterministically, ex) for ex in examples]

  num_scenes_in_dataset = len(examples)
  num_rays_per_batch = batch_size.per_process
  num_rays_per_scene, remainder = divmod(
      num_rays_per_batch, args.num_scenes_per_batch)
  num_scenes_per_process = args.num_scenes_per_batch * jax.local_device_count()

  if remainder != 0:
    raise ValueError(
        f'Unable to evenly split per-process batch of size '
        f'{batch_size.per_process} rays across {num_scenes_in_dataset} '
        f'scenes. Please choose batch_size and num_scenes_per_batch '
        f'accordingly.')

  # For lambda-capture within the for-loop below.
  def _select_idxs(idxs):
    return lambda t: t[idxs]

  def next_batch_fn(rng: np.random.Generator) -> types.Batch:
    """Returns a batch of rays for a single host.

    Result has leading shape [num_scenes_per_batch * local_device count,
    num_rays_per_scene, ...]

    Args:
      rng: Random number generator.

    Returns:
      Batch for a single device.

    """
    scene_idxs = rng.integers(num_scenes_in_dataset,
                              size=(num_scenes_per_process))
    result = []
    for scene_idx in scene_idxs:
      all_rays_in_scene = examples[scene_idx]
      num_rays_in_scene = all_rays_in_scene.batch_shape[0]
      ray_idxs_in_scene = rng.integers(num_rays_in_scene,
                                       size=(num_rays_per_scene,))
      select_rays_in_scene = jax.tree_map(_select_idxs(ray_idxs_in_scene),
                                          all_rays_in_scene)
      result.append(select_rays_in_scene)

    # List[Batch] --> Batch
    result = jax.tree_map(lambda *arrs: np.stack(arrs), *result)

    return result

  return next_batch_fn


def _next_batch_rays_all_scenes(examples: List[types.Batch],
                                args: DatasetParams,
                                batch_size: BatchSize):
  """Constructs next_batch() function s.t. rays are sampled at random."""
  del args

  # Concatenate all scenes together.
  examples = _concat_examples(examples)

  # Shuffle rays across scenes.
  examples = jax.tree_map(_shuffle_deterministically, examples)

  def next_batch_fn(rng):
    idx = rng.integers(examples.batch_shape[0],
                       size=(batch_size.per_process,))  # pytype: disable=attribute-error
    return jax.tree_map(lambda t: t[idx], examples)

  return next_batch_fn


def _to_devices_sharded(ex: types.Batch) -> types.Batch:
  """Prefetch the values into the device.

  Args:
    ex: The numpy example to prefetch with shape (num_local_devices, ...).

  Returns:
    ex: The `jax.ShardedDeviceArray` (same shape).
  """
  devices = jax.local_devices()
  return jax.tree_map(lambda x: jax.device_put_sharded(list(x), devices), ex)


def _shard(ex: types.Batch) -> types.Batch:
  """Reshape batch dimension to (num_local_device, batch_size_per_device)."""
  return jax.tree_map(_shard_single, ex)


def _shard_single(ex: tf.Tensor) -> tf.Tensor:
  device_count = jax.local_device_count()
  new_shape = [device_count, ex.shape[0] // device_count, *ex.shape[1:]]
  return tf.reshape(ex, new_shape)


def to_ds_state_bytes(state: DsState) -> DsState:
  """Converts state into tf/numpy compatible values."""

  def _to_bytes(x) -> bytes:
    if isinstance(x, bytes):
      return x
    assert isinstance(x, int), type(x)
    return str(x).encode()

  # tf/np does not support 128-bits integers, so convert to tf.string instead
  return jax.tree_map(_to_bytes, state)


def to_ds_state_int(state: DsState) -> DsState:
  """Restore the state to expected `np.random` value."""

  def _to_int(x) -> int:
    if isinstance(x, int):
      return x
    elif isinstance(x, str):
      # Flax saving/restoring mechanism do not preserve the types. Saving
      # bytes restore `str`
      return int(x)
    assert isinstance(x, bytes), type(x)
    return int(x.decode())

  return jax.tree_map(_to_int, state)


def _valid_ray_dir(ray_dir: tf.Tensor) -> tf.Tensor:
  """Return a boolean mask which encodes whether the ray_dir is valid."""
  return tf.reduce_mean(abs(ray_dir), axis=-1) > 0


def _compute_base_radii(rays: types.Rays) -> tf.Tensor:
  """Computes base radii for rays used in Mip-NeRF.

  Note that the base radii are computed based on the x-axis only.

  Args:
    rays: Rays to compute base_radii for.

  Returns:
    Base radii for the given rays.
  """
  # _    valid values
  # x    invalid values

  # x x _ _ x _ _ x x     ray_d
  ray_d = rays.direction
  del rays

  # Distance from each unit-norm direction vector to its x-axis neighbor.
  # x x _ _ x _ _ x       ray_d[..., :-1, :]
  # x _ _ x _ _ x x       ray_d[..., 1:, :]
  # x x _ x x _ x x       dx
  dx = (ray_d[..., :-1, :] - ray_d[..., 1:, :])**2
  dx = tf.sqrt(tf.reduce_sum(dx, -1))

  # Replace dx values with left neighbour if the original ray_d was invalid.
  # x _ _ x _ _ x x     left_shifted_mask
  left_shifted_mask = _valid_ray_dir(ray_d)[..., 1:]

  # x x x _ x x _ x     right_shifted_dx
  right_shifted_dx = tf.concat([dx[..., :1], dx[..., :-1]], -1)

  # x x _ _ x _ _ x     dx
  dx = tf.where(left_shifted_mask, dx, right_shifted_dx)

  # Note how the set of valid values in dx here is equal to the ones in ray_d.
  # x x _ _ x _ _ x x   dx
  dx = tf.concat([dx, dx[..., -1:]], -1)

  # Cut the distance in half, multiply it to match the variance of a uniform
  # distribution the size of a pixel (1/12, see paper).
  base_radii = dx[..., None] * 2 / np.sqrt(12.0)

  assert base_radii.shape[:-1] == ray_d.shape[:-1], (
      base_radii.shape, ray_d.shape)
  assert base_radii.shape[-1] == 1, base_radii.shape
  return base_radii


@sunds.utils.map_fn
def _add_base_radii(ex: types.Batch) -> types.Batch:
  """Populates the base_radius field of all target rays in a batch."""
  ex.target_view.rays.base_radius = _compute_base_radii(ex.target_view.rays)
  return ex


@sunds.utils.map_fn
def _crop_views(ex: types.Batch,
                crop: Tuple[int, int, int, int]) -> types.Batch:
  """Crop input and target views by specified amount.

  Args:
    ex: Input with batch shape `input_views=(n, h, w) target_view=(h, w)`
    crop: Amount of pixels to crop (top, bottom, left, right)

  Returns:
    ex: Output with cropped input and target views.
  """
  def crop_image(x: tf.Tensor) -> tf.Tensor:
    if x.dtype == tf.string:
      # If this is the image_id return as is.
      return x
    h = x.shape[-3]
    w = x.shape[-2]
    return x[...,
             crop[0]: h - crop[1],
             crop[2]: w - crop[3], :]

  ex.target_view = jax.tree_map(crop_image, ex.target_view)
  return ex


def _pop_image_ids(ex: types.Batch) -> types.Batch:
  """Remove the image_ids from the batch."""
  target_view = ex.target_view.replace(image_ids=None)
  return ex.replace(
      target_view=target_view,
  )


@np.vectorize
def _to_str_array(x):
  """Decodes bytes -> str array."""
  # tf.string tensors are returned as bytes, so need to convert them back to str
  return x.decode('utf8') if isinstance(x, bytes) else x


def _to_jnp_array(x: np.ndarray):
  """Convert array to jax (ignoring string arrays)."""
  # Forward str array as-is (as not supported by jax)
  return _to_str_array(x) if j3d.utils.is_array_str(x) else jnp.array(x)


def _to_device(xs):
  """Transfer data to devices (GPU/TPU)."""
  return jax.tree_map(_to_jnp_array, xs)
