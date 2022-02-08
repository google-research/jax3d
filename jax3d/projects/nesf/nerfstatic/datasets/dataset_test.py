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

"""Tests for jax3d.projects.nesf.nerfstatic.datasets.dataset."""

import collections
import itertools
from typing import List, Optional, Union
from unittest import mock

import chex
import jax
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.datasets import dataset
from jax3d.projects.nesf.nerfstatic.datasets import dataset_utils
from jax3d.projects.nesf.nerfstatic.utils import types
import numpy as np
import pytest


def _get_devices(num_devices: int) -> List[jax.xla.Device]:
  return num_devices * [jax.local_devices()[0]]


def _make_metadata(num_scenes: int = 1) -> List[dataset_utils.DatasetMetadata]:
  return [
      dataset_utils.DatasetMetadata(labels=['class0', 'class1'],)
      for i in range(num_scenes)
  ]


def _ds_iterable_from_examples(
    examples: Union[types.Batch, List[types.Batch]],
    *,
    batch_size: Optional[dataset.BatchSize],
    example_type: dataset.ExampleType,
    ds_state: Optional[dataset.DsState] = None,
    num_scenes_per_batch: Optional[int] = None,
    **kwargs,
):
  dset_param_kwargs = {}
  args = dataset.DatasetParams(data_dir=j3d.Path(''),
                               num_scenes_per_batch=num_scenes_per_batch,
                               **dset_param_kwargs)

  # Argument may be a single scene's examples OR examples from multiple
  # scenes. If a single scene's, convert it into a (length-1) list of scenes.
  if isinstance(examples, types.Batch):
    examples = [examples]

  ds = dataset._make_in_memory_dataset(
      examples=examples,
      batch_size=batch_size,
      example_type=example_type,
      ds_state=ds_state,
      args=args,
  )
  return dataset.DatasetIterable(ds, example_type=example_type, **kwargs)


def test_ds_iterator_imgs():
  b, h, w = 4, 28, 28
  examples = dataset_utils.make_examples(target_batch_shape=(b, h, w))
  ds = _ds_iterable_from_examples(
      examples=examples,
      all_metadata=_make_metadata(),
      batch_size=None,
      example_type=dataset.ExampleType.IMAGE,
  )
  assert ds.semantic_labels == ['class0', 'class1']
  all_exs = [ex for _, ex in ds]
  shuffled_examples = jax.tree_map(
      dataset._shuffle_deterministically,
      examples,
  )
  j3d.testing.assert_trees_all_close(
      all_exs, list(j3d.tree.unzip(shuffled_examples)))


def test_ds_iterator_imgs_num_scenes_per_batch():
  # Ensure example_type == IMAGE is unaffected by num_scenes_per_batch.
  b, h, w = 4, 28, 28
  examples = dataset_utils.make_examples(target_batch_shape=(b, h, w))
  ds = _ds_iterable_from_examples(
      examples=examples,
      all_metadata=_make_metadata(),
      batch_size=None,
      example_type=dataset.ExampleType.IMAGE,
      num_scenes_per_batch=2,
  )
  assert ds.semantic_labels == ['class0', 'class1']
  shuffled_examples = jax.tree_map(
      dataset._shuffle_deterministically,
      examples,
  )

  # Assert that all scene_ids are equal as all rays come from the same image.
  single_scene_id = shuffled_examples.target_view.rays.scene_id[0, 0, 0, 0]
  np.testing.assert_array_equal(single_scene_id,
                                shuffled_examples.target_view.rays.scene_id)


def test_ds_iterator_ray():
  b, h, w = 4, 28, 28
  examples = dataset_utils.make_examples(target_batch_shape=(b, h, w))
  ds = _ds_iterable_from_examples(
      examples=examples,
      all_metadata=_make_metadata(),
      batch_size=dataset.BatchSize(b),
      example_type=dataset.ExampleType.RAY,
  )
  assert ds.semantic_labels == ['class0', 'class1']
  _, ex = next(iter(ds))

  # target_batch_shape=(1, 4): Single host, single device
  assert j3d.types_like(ex) == types.Batch.as_types(target_batch_shape=(1, b))
  assert ds.peek()[1].target_view.rgb.shape == ex.target_view.rgb.shape


def test_ds_iterator_ray_num_scenes_per_batch():
  # Ensure that num_scenes_per_batch arguments has intended effect.
  k, b, h, w = 2, 8, 28, 28

  examples1 = dataset_utils.make_examples(target_batch_shape=(b, h, w))
  examples1.target_view.rays.scene_id = (
      np.zeros_like(examples1.target_view.rays.scene_id))

  examples2 = dataset_utils.make_examples(target_batch_shape=(2*b, h, w))
  examples2.target_view.rays.scene_id = (
      np.ones_like(examples2.target_view.rays.scene_id))

  ds = _ds_iterable_from_examples(
      examples=[examples1, examples2],
      all_metadata=_make_metadata(),
      batch_size=dataset.BatchSize(b),
      example_type=dataset.ExampleType.RAY,
      num_scenes_per_batch=k,
  )
  assert ds.semantic_labels == ['class0', 'class1']
  _, ex = next(iter(ds))

  # Assert that shape is [num_devices, num_scenes, num_rays_per_scene, ...]
  chex.assert_shape(ex.target_view.rays.origin, (1, k, b//k, 3))

  # Assert that Tensors are sharded by scene.
  assert np.all(np.equal(ex.target_view.rays.scene_id[0, 0, 0, 0],
                         ex.target_view.rays.scene_id[0, 0]))
  assert np.all(np.equal(ex.target_view.rays.scene_id[0, 1, 0, 0],
                         ex.target_view.rays.scene_id[0, 1]))

  # target_batch_shape=(1, 2, 2): Single host, single device
  assert (j3d.types_like(ex) ==
          types.Batch.as_types(target_batch_shape=(1, k, b//k)))
  assert ds.peek()[1].target_view.rgb.shape == ex.target_view.rgb.shape


def test_ds_labels():
  b, h, w = 4, 28, 28
  examples = dataset_utils.make_examples(target_batch_shape=(b, h, w))
  ds = _ds_iterable_from_examples(
      examples=examples,
      batch_size=None,
      example_type=dataset.ExampleType.IMAGE,
  )
  assert ds.semantic_labels == []  # pylint: disable=g-explicit-bool-comparison

  ds = _ds_iterable_from_examples(
      examples=examples,
      all_metadata=[
          dataset_utils.DatasetMetadata(labels=['class0', 'class1']),
          dataset_utils.DatasetMetadata(labels=['class1', 'class0']),
      ],
      batch_size=None,
      example_type=dataset.ExampleType.IMAGE,
  )
  with pytest.raises(ValueError, match='Inconsistent labels'):
    assert ds.semantic_labels


def test_ds_iterator_deterministic():
  b, h, w = 100, 28, 28
  examples = dataset_utils.make_examples(target_batch_shape=(b, h, w))

  # Examples 1: Dataset initialized from default state at step 0.
  ds1 = _ds_iterable_from_examples(
      examples=examples,
      all_metadata=_make_metadata(),
      batch_size=dataset.BatchSize(2),
      example_type=dataset.ExampleType.RAY,
  )

  sequence_exs1 = list(itertools.islice(ds1, 40))

  # Examples 2: Dataset restored from the 20th example (state 19 will yield
  # example 20)
  ds_state = sequence_exs1[19][0]
  ds2 = _ds_iterable_from_examples(
      examples=examples,
      all_metadata=_make_metadata(),
      batch_size=dataset.BatchSize(2),
      example_type=dataset.ExampleType.RAY,
      ds_state=ds_state,
  )

  sequence_exs2 = list(itertools.islice(ds2, 20))

  states1, exs1 = zip(*sequence_exs1[20:])
  states2, exs2 = zip(*sequence_exs2)
  assert states1 == states2
  assert states1[0] != states1[1]
  chex.assert_tree_all_close(exs1, exs2, ignore_nones=True)

  # to_ds_state_int is a no-op if value are already int
  assert (dataset.to_ds_state_int(ds_state) == dataset.to_ds_state_int(
      dataset.to_ds_state_int(ds_state)))
  assert (dataset.to_ds_state_bytes(ds_state) == dataset.to_ds_state_bytes(
      dataset.to_ds_state_bytes(ds_state)))


@mock.patch('jax.process_count', return_value=2)
@mock.patch('jax.device_count', return_value=8)
@mock.patch('jax.local_device_count', return_value=4)
@mock.patch('jax.local_devices', return_value=_get_devices(4))
def test_ds_iterator_ray_multi_host(*mock_args):
  del mock_args

  b, h, w = 10, 28, 28
  examples = dataset_utils.make_examples(target_batch_shape=(b, h, w))

  with pytest.raises(ValueError, match='Batch size .* must be divisible by'):
    dataset.BatchSize(4)

  batch_size = dataset.BatchSize(16)
  assert batch_size.total == 16
  assert batch_size.per_process == 8  # batch_size // process_count
  assert batch_size.per_device == 2  # per_process // local_device_count

  ds = _ds_iterable_from_examples(
      examples=examples,
      batch_size=batch_size,
      example_type=dataset.ExampleType.RAY,
  )
  _, ex = next(iter(ds))

  # Each host get (local_device_count, batch_size.per_device)
  assert j3d.types_like(ex) == types.Batch.as_types(target_batch_shape=(4, 2))
  assert ds.peek()[1].target_view.rgb.shape == ex.target_view.rgb.shape


def test_example_loader():
  b, h, w = 5, 28, 28
  examples = dataset_utils.make_examples(target_batch_shape=(b, h, w))

  def _in_memory_make_test_examples(
      *,
      data_dir: j3d.Path,
      split: str,
      **kwargs,
  ):
    del kwargs
    assert isinstance(data_dir, j3d.Path)
    ids = np.array([2, 1, 4, 0] if split == 'train' else [3])
    metadata = dataset_utils.DatasetMetadata(labels=['class0', 'class1'],)
    return jax.tree_map(lambda t: t[ids], examples), metadata

  ds_type = 'test_dataset'
  registered_ds = dataset_utils.RegisteredDataset(
      name=ds_type,
      make_examples_fn=_in_memory_make_test_examples,
  )
  dataset_utils.register_dataset(registered_ds)

  # Single scene loaded
  args = dataset.DatasetParams(
      data_dir=j3d.Path(''),
      dataset=ds_type,
  )
  exs, _ = dataset._in_memory_examples_loader(
      registered_dataset=registered_ds,
      split='train',
      args=args,
      is_novel_scenes=False,
  )
  exs = dataset._concat_examples(exs)
  assert j3d.types_like(exs) == types.Batch.as_types(
      target_batch_shape=(4, 28, 28), image_id=True)
  assert set(_get_scene_ids(exs)) == {0}  # Single scene has id==0
  exs, _ = dataset._in_memory_examples_loader(
      registered_dataset=registered_ds,
      split='test',
      args=args,
      is_novel_scenes=False,
  )
  exs = dataset._concat_examples(exs)
  assert j3d.types_like(exs) == types.Batch.as_types(
      target_batch_shape=(1, 28, 28), image_id=True)
  assert set(_get_scene_ids(exs)) == {0}  # Single scene has id==0

  # Multi-scene loading
  args = dataset.DatasetParams(
      data_dir=j3d.Path(''),
      dataset=ds_type,
      train_scenes=5,
      eval_scenes=2,
  )
  exs, _ = dataset._in_memory_examples_loader(
      registered_dataset=registered_ds,
      split='test',
      args=args,
      is_novel_scenes=False,
  )
  exs = dataset._concat_examples(exs)
  # 2 scenes of 1 examples
  assert exs.target_view.rgb.shape == (1 * 2, 28, 28, 3)
  exs, _ = dataset._in_memory_examples_loader(
      registered_dataset=registered_ds,
      split='train',
      args=args,
      is_novel_scenes=False,
  )
  exs = dataset._concat_examples(exs)
  exs = jax.tree_map(dataset._shuffle_deterministically, exs)
  # 5 scenes of 4 examples
  assert exs.target_view.rgb.shape == (4 * 5, 28, 28, 3)

  all_scene_ids = _get_scene_ids(exs)
  # Scene ids are randomly but deteministically shuffled
  assert all_scene_ids == [
      1, 4, 4, 2, 1, 0, 1, 2, 3, 4, 0, 1, 0, 2, 3, 4, 2, 3, 3, 0
  ]

  # Check that the scene_ids have been propertly set:
  scene_ids_count = collections.Counter(all_scene_ids)
  assert scene_ids_count == {0: 4, 1: 4, 2: 4, 3: 4, 4: 4}

  with pytest.raises(ValueError, match='Unknown split'):
    dataset._in_memory_examples_loader(
        registered_dataset=registered_ds,
        split='eval',
        args=args,
        is_novel_scenes=False,
    )

  args = dataset.DatasetParams(
      data_dir=j3d.Path(''),
      dataset=ds_type,
      eval_scenes=1,
  )
  with pytest.raises(ValueError, match='Both or none of train_scenes'):
    dataset._in_memory_examples_loader(
        registered_dataset=registered_ds,
        split='train',
        args=args,
        is_novel_scenes=False,
    )

  # Test is_novel_scenes=True/False
  args = dataset.DatasetParams(
      data_dir=j3d.Path(''),
      dataset=ds_type,
      train_scenes=5,
      novel_scenes='5:7',
  )
  exs, _ = dataset._in_memory_examples_loader(
      registered_dataset=registered_ds,
      split='test',
      args=args,
      is_novel_scenes=False,
  )
  exs = dataset._concat_examples(exs)
  all_scene_ids = _get_scene_ids(exs)
  assert set(all_scene_ids) == {0, 1, 2, 3, 4}  # ids start at 0

  # When is_novel_scenes=True, get a different set of scenes
  exs, _ = dataset._in_memory_examples_loader(
      registered_dataset=registered_ds,
      split='test',
      args=args,
      is_novel_scenes=True,
  )
  exs = dataset._concat_examples(exs)
  all_scene_ids = _get_scene_ids(exs)
  # Scene ids are 0-1
  assert set(all_scene_ids) == {0, 1}


def _get_scene_ids(exs: types.Batch) -> List[int]:
  return [ex.target_view.rays.scene_id[0, 0, 0] for ex in j3d.tree.unzip(exs)]


@pytest.mark.parametrize('in_, expected', [
    (None, None),
    (1, [0]),
    (4, [0, 1, 2, 3]),
    (':1', [0]),
    (':5', [0, 1, 2, 3, 4]),
    ('2:5', [2, 3, 4]),
    ('20:100', list(range(20, 100))),
    (':100', list(range(100))),
    ((1, 3, 5), [1, 3, 5]),
    ([1, 2], [1, 2]),
])
def test_make_scene_range(in_, expected):
  assert dataset.make_scene_range(in_) == expected


@pytest.mark.parametrize('pattern',
                         [':', ':0', '2:', ':-1', '2:3.', '2:-3', '4:3', '4:4'])
def test_make_scene_range_invalid(pattern):
  with pytest.raises(ValueError, match='Invalid scene selection pattern'):
    dataset.make_scene_range(pattern)


@pytest.mark.parametrize('split', ['train', 'test'])
def test_get_scene_range(split: str):
  # Single dataset
  assert dataset._get_scene_range(
      train_scenes=None,
      eval_scenes=None,
      novel_scenes=None,
      ignore_scenes=None,
      split=split,
      is_novel_scenes=False,
  ) is None
  # Multi-scene (train == test)
  assert dataset._get_scene_range(
      train_scenes='3:5',
      eval_scenes=None,
      novel_scenes=None,
      ignore_scenes=None,
      split=split,
      is_novel_scenes=False,
  ) == list(range(3, 5))
  # Multi-scene (train == test)
  assert dataset._get_scene_range(  # pylint: disable=g-long-ternary
      train_scenes=5,
      eval_scenes='5:7',
      novel_scenes=None,
      ignore_scenes=None,
      split=split,
      is_novel_scenes=False,
  ) == list(range(5)) if split == 'train' else list(range(5, 7))
  # novel_scenes == True
  assert dataset._get_scene_range(
      train_scenes='3:5',
      eval_scenes='3:5',
      novel_scenes='5:10',
      ignore_scenes=None,
      split=split,
      is_novel_scenes=True,
  ) == list(range(5, 10))
  # novel_scenes == False
  assert dataset._get_scene_range(
      train_scenes='3:5',
      eval_scenes=None,
      novel_scenes='6:10',
      ignore_scenes=None,
      split=split,
      is_novel_scenes=False,
  ) == list(range(3, 5))
  # ignore_scenes, train
  assert dataset._get_scene_range(
      train_scenes='1:5',
      eval_scenes=None,
      novel_scenes='6:10',
      ignore_scenes=[1, 4, 8],
      split=split,
      is_novel_scenes=False,
  ) == [2, 3]
  # ignore_scenes, novel
  assert dataset._get_scene_range(
      train_scenes='1:5',
      eval_scenes=None,
      novel_scenes='6:10',
      ignore_scenes=[1, 4, 8],
      split=split,
      is_novel_scenes=True,
  ) == [6, 7, 9]


def test_num_scenes():
  assert dataset.num_scenes(None) == 1
  assert dataset.num_scenes(4) == 4
  assert dataset.num_scenes('2:6') == 4


def test_get_scene_range_invalid():

  with pytest.raises(ValueError, match='novel_scenes should be set'):
    dataset._get_scene_range(
        train_scenes='3:5',
        eval_scenes='3:5',
        novel_scenes=None,  # Missing novel_scenes
        ignore_scenes=None,
        split='train',
        is_novel_scenes=True,
    )

  with pytest.raises(ValueError, match='Both .* train_scenes and eval_scenes'):
    dataset._get_scene_range(
        train_scenes=None,  # Eval scene set, but not train
        eval_scenes='3:5',
        novel_scenes=None,
        ignore_scenes=None,
        split='train',
        is_novel_scenes=False,
    )

  with pytest.raises(ValueError, match='both train and novel should be set'):
    dataset._get_scene_range(
        train_scenes=None,
        eval_scenes=None,
        novel_scenes='3:4',  # novel set, but not train
        ignore_scenes=None,
        split='test',
        is_novel_scenes=False,
    )

  with pytest.raises(
      ValueError,
      match='is_novel_scenes=True, train and eval scene should be identical'):
    dataset._get_scene_range(
        train_scenes='3:4',
        eval_scenes='3:7',  # eval_scenes != train_scenes
        novel_scenes='6:10',
        ignore_scenes=None,
        split='test',
        is_novel_scenes=False,
    )

  with pytest.raises(ValueError, match='novel scenes should not overlap'):
    dataset._get_scene_range(
        train_scenes=':4',
        eval_scenes=None,
        novel_scenes='3:10',  # novel setoverlapp with train
        ignore_scenes=None,
        split='test',
        is_novel_scenes=False,
    )


def test_compute_base_radii():
  """Simple test for base radii to confirm that masking works."""
  ray_d = np.array([[0.0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]])
  ray_d = ray_d[None]  # Add an extra dimension to catch more errors.
  mask = np.array([0, 1, 1, 1, 0]).astype(bool)[None]
  v = np.sqrt(2/3)
  expected_base_radii = np.array([0, v, v, v, 0])[None, :, None]

  rays = types.Rays(scene_id=None, origin=0.0, direction=ray_d)
  actual_base_radii = np.array(dataset._compute_base_radii(rays))
  chex.assert_equal_shape([actual_base_radii, expected_base_radii])

  # Only assert base_radii for valid ray_d's.
  actual_base_radii = actual_base_radii[mask]
  expected_base_radii = expected_base_radii[mask]
  assert actual_base_radii.size == expected_base_radii.size == mask.sum()
  j3d.testing.assert_trees_all_close(actual_base_radii, expected_base_radii)
