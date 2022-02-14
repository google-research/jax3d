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

"""Klevr dataset."""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import chex
import jax
import jax3d.projects.nesf as j3d
from jax3d.projects.nesf.nerfstatic.datasets import dataset_utils
from jax3d.projects.nesf.nerfstatic.utils import camera_utils
from jax3d.projects.nesf.nerfstatic.utils import img_utils
from jax3d.projects.nesf.nerfstatic.utils import types
from jax3d.projects.nesf.utils.typing import f32, i32  # pylint: disable=g-multiple-import
import numpy as np


def make_examples(
    data_dir: j3d.Path,
    *,
    split: str,
    image_idxs: Union[None, int, Sequence[int]],
    enable_sqrt2_buffer: bool,
    **kwargs,
) -> Tuple[types.Batch, dataset_utils.DatasetMetadata]:
  return _make_examples_impl(data_dir=data_dir,
                             split=split,
                             metadata_cls=MetadataHandler,
                             image_idxs=image_idxs,
                             enable_sqrt2_buffer=enable_sqrt2_buffer,
                             **kwargs)


def make_unreal_examples(
    data_dir: j3d.Path,
    *,
    split: str,
    **kwargs,
) -> Tuple[types.Batch, dataset_utils.DatasetMetadata]:
  return _make_examples_impl(data_dir=data_dir,
                             split=split,
                             metadata_cls=UnrealMetadataHandler,
                             image_idxs=None,
                             enable_sqrt2_buffer=False,
                             **kwargs)


class MetadataHandler(object):
  """Metadata that matches code generated in kubric.

  For details see:
  third_party/py/kubric/google/scene_centric_klevr_worker_borg.py.
  """

  def __init__(self, data_dir: j3d.Path):
    self._data_dir = data_dir
    self._metadata = json.loads((data_dir / 'metadata.json').read_text())
    self._scene_id = os.path.basename(data_dir)

  @property
  def split_ids(self):
    return self._metadata['split_ids']

  def rgba_filename(self, index: int):
    return self._data_dir / f'rgba_{index:05d}.png'

  def semantic_filename(self, index: int):
    return self._data_dir / f'segmentation_{index:05d}.png'

  def depth_filename(self, index: int):
    return self._data_dir / f'depth_{index:05d}.tiff'

  def make_cameras(self, ids: List[int], width: int, height: int):
    return _make_cameras(self._metadata['camera'], ids=ids,
                         width=width, height=height)

  @property
  def scene_id(self):
    return self._scene_id

  @property
  def scene_boundaries(self):
    return types.BoundingBox3d(
        min_corner=np.array(self._metadata['scene_boundaries']['min']),
        max_corner=np.array(self._metadata['scene_boundaries']['max']),
    )

  @property
  def segmentation_labels(self):
    return self._metadata['segmentation_labels']


class UnrealMetadataHandler(MetadataHandler):
  """Metadata handler for Unreal generated scenes."""

  @property
  def split_ids(self):
    total_frames = len(self._metadata['camera']['positions'])
    return _generate_splits(total_frames=total_frames,
                            test_frames=int(total_frames * 0.1))

  def depth_filename(self, index: int):
    return self._data_dir / f'depth_{index:05d}.png'

  def make_cameras(self, ids: List[int], width: int, height: int):
    return _make_cameras(self._metadata['camera'], ids=ids,
                         width=width, height=height, use_unreal_axes=True)

  @property
  def scene_boundaries(self):
    inv_x = np.array([-1, 1, 1])
    return types.BoundingBox3d(
        min_corner=np.array(self._metadata['scene_boundaries']['min']) * inv_x,
        max_corner=np.array(self._metadata['scene_boundaries']['max']) * inv_x,
    )


def _make_cameras(
    scene_cameras: Dict[str, Any],
    ids: i32['n'],
    width: int,
    height: int,
    use_unreal_axes: bool = False,
) -> camera_utils.Camera:
  """Returns the Camera object containing camera info for all frames."""
  return camera_utils.Camera.from_position_and_quaternion(
      positions=np.array(scene_cameras['positions'])[ids],
      quaternions=np.array(scene_cameras['quaternions'])[ids],
      resolution=(height, width),
      # Assume square pixels: width / sensor_width == height / sensor_height
      focal_px_length=(
          scene_cameras['focal_length'] * width / scene_cameras['sensor_width']
      ),
      use_unreal_axes=use_unreal_axes,
  )


# TODO(tutmann): Hardcode splits during data generation.
def _generate_splits(total_frames: int,
                     test_frames: int) -> Dict[str, List[int]]:
  # Use same seed for all fields
  rng = np.random.RandomState(178)
  all_frames = list(range(total_frames))
  rng.shuffle(all_frames)  # Shuffle in-place
  return {'train': all_frames[test_frames:], 'test': all_frames[:test_frames]}


def _choose_image_ids(selected: Union[None, int, Sequence[int]],
                      available: List[int]) -> List[int]:
  """Choose which image ids to load from disk."""

  # Load all.
  if selected is None:
    return available

  # Load a random selection.
  if isinstance(selected, int):
    if selected > len(available):
      raise ValueError(f'Unable to sample {selected} random indices from '
                       f'{len(available)} available.')
    rng = np.random.RandomState(42)
    ids = rng.choice(available, size=selected, replace=False)
    return ids.tolist()

  # Load a specific set of indices.
  if isinstance(selected, (list, tuple)):
    if any([i >= len(available) for i in selected]):
      raise ValueError(f'Unable to select indices {selected} from '
                       f'{len(available)} available.')
    return [available[i] for i in selected]

  raise NotImplementedError(
      f'Unable to choose image_ids using selected of type {type(selected)}: '
      f'{selected}')


def _rescale_scene_boundaries(original: types.BoundingBox3d,
                              scale: f32['3']) -> types.BoundingBox3d:
  """Rescale scene boundaries by multiplier 'scale'."""
  chex.assert_shape(scale, (3,))
  lower = original.min_corner
  upper = original.max_corner
  center = (lower + upper) / 2
  rescale = lambda x: (x - center) * scale + center
  result = types.BoundingBox3d(min_corner=rescale(lower),
                               max_corner=rescale(upper))
  return result


def _make_examples_impl(
    data_dir: j3d.Path,
    *,
    split: str,
    metadata_cls: Type[MetadataHandler],
    image_idxs: Union[None, int, Sequence[int]],
    enable_sqrt2_buffer: bool,
    scene_semantic_images: Optional[Dict[int, i32['h w']]] = None,
    **kwargs,
) -> Tuple[types.Batch, dataset_utils.DatasetMetadata]:
  """Load examples from the sub-directory.

  Args:
    data_dir: Directory containing the examples
    split: Split name to load.
    metadata_cls: a MetadataHandler class type
    image_idxs: Which images to load. Index is with respect to position in
      metadata["split_ids"][split].
    enable_sqrt2_buffer: If set, the scene's bounding box will be increased by
      a factor of sqrt(2) along the x-axis and y-axis. For use with random
      scene rotations.
    scene_semantic_images: If set, the semantic labels are loaded from
      the given set.
    **kwargs: Unused

  Returns:
    batch: Batch containing the full dataset
  """
  del kwargs

  metadata = metadata_cls(data_dir)

  # Only load a subset of the ids
  ids = metadata.split_ids[split]

  # Choose the subset of valid image indices
  ids = _choose_image_ids(selected=image_idxs, available=ids)

  # Add all paths to load
  images = []
  semantics = []
  depths = []
  image_ids = []
  for id_ in ids:
    images.append(metadata.rgba_filename(id_))
    semantics.append(metadata.semantic_filename(id_))
    depths.append(metadata.depth_filename(id_))
    image_ids.append(metadata.scene_id + '_' + metadata.rgba_filename(id_).stem)

  # Load all images in parallel
  images, semantics, depths = j3d.tree.parallel_map(
      img_utils.imread,
      (images, semantics, depths),
      report_progress=True,
  )
  semantics, depths = jax.tree_map(
      lambda i: i[..., :1],
      (semantics, depths),
  )
  image_ids = np.array(image_ids)

  # Scene camera contains the position/quaternion
  cameras = metadata.make_cameras(ids=ids,
                                  width=images[0].shape[1],
                                  height=images[0].shape[0])

  # Convert rgba -> rgb, normalize to [0, 1]
  rgb = np.stack(images, axis=0)[..., :3].astype(np.float32) / 255.
  semantics = np.stack(semantics, axis=0).astype(np.int32)
  depths = np.stack(depths, axis=0)

  if scene_semantic_images:
    pseudo_semantics = []
    mask = []
    for ii, id_ in enumerate(ids):
      if id_ in scene_semantic_images:
        pseudo_semantics.append(scene_semantic_images[id_][..., None])
        mask.append(np.ones_like(pseudo_semantics[-1]))
      else:
        pseudo_semantics.append(semantics[ii])
        mask.append(np.zeros_like(pseudo_semantics[-1]))
    semantics = np.stack(pseudo_semantics, axis=0).astype(np.int32)
    semantic_mask = np.stack(mask, axis=0)
  else:
    semantic_mask = None

  assert rgb.shape == (len(ids), *cameras.resolution, 3)
  assert semantics.shape == (*rgb.shape[:-1], 1)
  assert depths.shape == (*rgb.shape[:-1], 1)
  assert image_ids.shape == (len(ids),)
  if semantic_mask is not None:
    assert semantic_mask.shape == (*rgb.shape[:-1], 1)

  scene_boundaries = metadata.scene_boundaries
  if enable_sqrt2_buffer:
    scene_boundaries = _rescale_scene_boundaries(
        scene_boundaries, scale=np.array([np.sqrt(2), np.sqrt(2), 1]))

  unscaled_rays = cameras.pixel_centers2rays(scene_boundaries=None)

  # Rescale depth according to scene_boundaries.
  oriented_depths = unscaled_rays.direction * depths
  oriented_depths /= (scene_boundaries.size / 2)
  depths = np.linalg.norm(oriented_depths, axis=-1, keepdims=True)

  rays = cameras.pixel_centers2rays(scene_boundaries=scene_boundaries)

  batch = types.Batch(
      target_view=types.Views(
          rays=rays,
          rgb=rgb,
          depth=depths,
          semantics=semantics,
          image_ids=image_ids,
          semantic_mask=semantic_mask
          ),
      )
  # TODO(epot): Also returns metadata to help debugging (camera, length,
  # scene_boundaries...) ?
  metadata = dataset_utils.DatasetMetadata(
      labels=['background'] + metadata.segmentation_labels,
      cameras=cameras,
      scene_name=data_dir.parts[-1])

  return batch, metadata


dataset_utils.register_dataset(
    dataset_utils.RegisteredDataset(
        name='klevr',
        make_examples_fn=make_examples,
    ))


dataset_utils.register_dataset(
    dataset_utils.RegisteredDataset(
        name='unreal',
        make_examples_fn=make_unreal_examples,
    ))
