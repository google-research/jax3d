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

"""Functions for visualizing instance segmentation."""
import functools
from typing import Optional, Union

from etils.array_types import f32
from etils.array_types import i32
from etils.array_types import ui8
import matplotlib as mpl
import numpy as np
import scipy.ndimage
import skimage.color

# Oklab colorspace conversion matrices
# see https://bottosson.github.io/posts/oklab/
# and here for an interactive demo:
# https://raphlinus.github.io/color/2021/01/18/oklab-critique.html

_M1 = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
])

_M1_inv = np.linalg.inv(_M1)

_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_M2_inv = np.linalg.inv(_M2)


def oklab_from_rgb(rgb: f32['... 3']) -> f32['... 3']:
  """Convert colors from RGB (sRGB) colorspace to Oklab colorspace.

  See https://bottosson.github.io/posts/oklab/

  Args:
    rgb: An array of RGB color values as floats in the range [0, 1]

  Returns:
    an array of Oklab color values as floats
  """
  xyz = skimage.color.rgb2xyz(rgb)
  lms = np.cbrt(np.einsum('...ij,...j->...i', _M1, xyz))
  lab = np.einsum('...ij,...j->...i', _M2, lms)
  return lab


def rgb_from_oklab(lab: f32['... 3']) -> f32['... 3']:
  """Convert colors from Oklab colorspace to RGB colorspace.

  See https://bottosson.github.io/posts/oklab/

  Args:
    lab: An array of RGB color values as floats

  Returns:
    an array of RGB color values as floats
  """
  lms = np.einsum('...ij,...j->...i', _M2_inv, lab)
  lms_cubed = np.power(lms, 3)
  xyz = np.einsum('...ij,...j->...i', _M1_inv, lms_cubed)
  return skimage.color.xyz2rgb(xyz)


def lch_from_oklab(lab: f32['... 3']) -> f32['... 3']:
  """Convert Oklab colors to LCh colors (the polar form of Oklab).

  This form is useful for generating color palettes.
  (Oklab is to LCh) as (RGB is to HLS)
  The meaning of the three components is as follows:

  L: Lightness
  C: Chroma (similar to saturation)
  h: hue

  Args:
    lab: An array of Oklab color values as floats

  Returns:
    an array of LCh color values as floats
  """
  l = lab[..., 0]
  a = lab[..., 1]
  b = lab[..., 2]
  c = np.sqrt(np.square(a) + np.square(b))
  h = np.arctan2(b, a)
  return np.stack([l, c, h], axis=-1)


def oklab_from_lch(lch: f32['... 3']) -> f32['... 3']:
  """Convert LCh colors (the polar form of Oklab) to Oklab colors.

  This form is useful for generating color palettes.
  (Oklab is to LCh) as (RGB is to HSV)
  The meaning of the three components is as follows:

  L: Lightness
  C: Chroma (similar to saturation)
  h: hue

  Args:
    lch: An array of LCh color values as floats

  Returns:
    an array of Oklab color values as floats
  """
  l = lch[..., 0]
  c = lch[..., 1]
  h = lch[..., 2]
  a = c * np.cos(h)
  b = c * np.sin(h)
  return np.stack([l, a, b], axis=-1)


def lch_from_rgb(rgb: f32['... 3']) -> f32['... 3']:
  lab = oklab_from_rgb(rgb)
  return lch_from_oklab(lab)


def rgb_from_lch(lch: f32['... 3']) -> f32['... 3']:
  lab = oklab_from_lch(lch)
  return rgb_from_oklab(lab)


COLOR_SPACES = {
    'rgb': (lambda x: x, lambda x: x),
    'xyz': (skimage.color.rgb2xyz, skimage.color.xyz2rgb),
    'lab': (skimage.color.rgb2lab, skimage.color.lab2rgb),
    'oklab': (oklab_from_rgb, rgb_from_oklab),
    'hsv': (skimage.color.rgb2hsv, skimage.color.hsv2rgb),
    'lch': (lch_from_rgb, rgb_from_lch),
}


def maybe_convert_to_wide_form(
    segmentation: Union[f32['... K'], i32['... 1']],
    nr_segments: Optional[int] = None,
) -> f32['... K']:
  """Convert segmentation into the form with a dimension per segment."""
  k = segmentation.shape[-1]
  int_types = {np.uint8, np.uint16, np.int16, np.int32, np.int64}
  if k == 1:  # convert segmentation to one-hot
    if segmentation.dtype.type not in int_types:
      raise ValueError('Segmentation has final dimension 1 so it should be'
                       f'int-typed, but is {segmentation.dtype})')
    max_seg = np.max(segmentation) + 1
    nr_segments = nr_segments if nr_segments is not None else max_seg
    segmentation = np.eye(nr_segments, dtype=np.float32)[segmentation[..., 0]]
  return segmentation


def convert_to_one_hot(
    segmentation: Union[f32['... K'], i32['... 1']],
    nr_segments: Optional[int] = None,
) -> f32['... K']:
  """Convert segmentation into one-hot form."""
  if segmentation.shape[-1] == 1:  # convert to one-hot
    return maybe_convert_to_wide_form(segmentation, nr_segments)
  else:  # first argmax then convert to one-hot
    argmax_seg = segmentation.argmax(axis=-1)[..., None]
    return maybe_convert_to_wide_form(argmax_seg, nr_segments)


def palette_lch(n: int, chroma=0.3, lightness=0.8) -> f32['n 3']:
  starting_hue = np.pi * 1.5  # blue
  hues = np.linspace(starting_hue, 2*np.pi + starting_hue, n + 1) % (2*np.pi)
  chroma = np.ones_like(hues) * chroma
  lightness = np.ones_like(hues) * lightness
  pal = rgb_from_lch(np.stack([lightness, chroma, hues], axis=-1))
  # skip last color since it is the same as first
  return pal[:-1]


def palette_hsv(n: int) -> f32['n 3']:
  starting_hue = 2/3  # blue
  r = np.linspace(starting_hue, starting_hue + 1.0, n + 1) % 1.0
  cmap = mpl.cm.get_cmap('hsv')
  pal = cmap(r)
  # skip last color since it is the same as first
  # only use RGB values (drop alpha value)
  return pal[:-1, :3]


def palette_mpl(
    name: str,
    n: int,
    minv: float = 0.1,
    maxv: float = 0.9,
) -> f32['n 3']:
  """Create a discrete palette from a matplotlib colormap.

  Also discard the alpha channel, and optionally limit the range
  because the very bright / dark values are not well suited for visualization.

  Args:
    name: name of the colormap.
      See https://matplotlib.org/stable/gallery/color/colormap_reference.html
    n: number of colors
    minv: minimum colormap value to use (defaults to 0.1)
    maxv: maximum colormap value to use (defaults to 0.9)
  Returns:
    a discrete RGB colormap with n color in the shape f32['n 3']
  """
  r = np.linspace(minv, maxv, n)
  cmap = mpl.cm.get_cmap(name)
  return cmap(r)[:, :3]


# commonly used palettes
PALETTES = {
    'twilight': functools.partial(palette_mpl, 'twilight_shifted'),
    'plasma': functools.partial(palette_mpl, 'plasma'),
    'viridis': functools.partial(palette_mpl, 'viridis'),
    'cividis': functools.partial(palette_mpl, 'cividis'),
    'gray': functools.partial(palette_mpl, 'gray'),
    'turbo': functools.partial(palette_mpl, 'turbo', minv=0, maxv=1),
    'lch': palette_lch,
    'hsv': palette_hsv,
}


def _safe_log2(x: f32['...'], eps: float = 1e-6):
  """Avoid NaNs from log(0) by clipping the input to the range [eps, inf)."""
  safe_x = np.clip(x, a_min=eps, a_max=np.inf)
  return np.log2(safe_x)


def norm_entropy(x: f32['... k'], axis=-1) -> f32['... 1']:
  """Compute the normalized entropy [0, 1] along the final axis."""
  entropy = -np.sum(x * _safe_log2(x), axis=axis, keepdims=True)
  if x.shape[-1] == 1:
    return np.ones_like(x)
  else:
    return np.clip(entropy / np.log2(x.shape[-1]), 0, 1)


def blur_image(img: f32['... H W C'], sigma: float = 1.0) -> f32['... H W C']:
  batch_axis = np.arange(img.ndim - 3)  # all axis except the last three
  sigma = (0,) * len(batch_axis) + (sigma, sigma, 0)
  return scipy.ndimage.gaussian_filter(img, sigma=sigma)


def optimize_palette(
    palette: f32['k 3'],
    segmentation: Union[f32['...H W K'], i32['...H W 1']],
    hard: bool = False,
    color_space: str = 'oklab',
    blur: int = 5,
    iterations: int = 10000,
    seed: int = 42,
    threshold: float = 0.75,
    perm_mask: Optional[ui8['k']] = None,
) -> f32['k 3']:
  """Shuffle a palette to reduce similar colors being next to each other."""

  to_colorspace, _ = COLOR_SPACES[color_space]
  palette_cs = to_colorspace(palette)
  color_diff = np.linalg.norm(palette_cs[None] - palette_cs[:, None], axis=-1)
  color_sim = np.sqrt(3) - color_diff

  assert 0.0 <= threshold <= 1.0, threshold
  threshold = color_sim.min() + (color_sim.max() - color_sim.min()) * threshold
  color_sim[color_sim < threshold] = 0.   # only penalize very similar colors

  if hard:
    seg = convert_to_one_hot(segmentation, palette.shape[0])
  else:
    seg = maybe_convert_to_wide_form(segmentation, palette.shape[0])

  blurred_seg = blur_image(seg, sigma=blur)
  seg_sim = np.einsum('...hwk, ...hwl -> ...kl', blurred_seg, blurred_seg)
  seg_sim = seg_sim.sum(axis=tuple(range(seg.ndim - 3)))

  # TODO(klausg): use scipy.optimize.quadratic_assignment instead
  rng = np.random.RandomState(seed)
  perm = np.arange(palette.shape[0])
  if perm_mask is None:
    perm_mask = np.zeros(palette.shape[0])
  modifiable_perm = perm[perm_mask == 0]

  best = (np.inf, perm.copy())
  for _ in range(iterations):
    # compute np.trace(seg_similarities @ color_similarities) as einsum
    c = np.einsum('ij,ji->', seg_sim, color_sim[perm, :][:, perm])
    if c < best[0]:
      best = c, perm.copy()
    rng.shuffle(modifiable_perm)
    perm[perm_mask == 0] = modifiable_perm

  return palette[best[1]]


def align_segmentation_to_reference(
    segmentation: Union[f32['...H W K'], i32['...H W 1']],
    reference: Union[f32['...H W K'], i32['...H W 1']],
    hard: bool = False,
) -> f32['...H W K']:
  """Permute segmentation indices to best match a reference segmentation."""
  k = max(segmentation.shape[-1], reference.shape[-1])
  nr_segments = k if k != 1 else None

  if hard:
    seg = convert_to_one_hot(segmentation, nr_segments)
    ref = convert_to_one_hot(reference, nr_segments)
  else:
    seg = maybe_convert_to_wide_form(segmentation, nr_segments)
    ref = maybe_convert_to_wide_form(reference, nr_segments)

  # compute cost matrix as the negative co-occurance matrix between seg and ref
  flat_seg = seg.reshape((-1, seg.shape[-1]))
  flat_ref = ref.reshape((-1, ref.shape[-1]))
  cost = - flat_ref.T @ flat_seg
  _, idx = scipy.optimize.linear_sum_assignment(cost)
  return seg[..., idx]


def plot_segmentation(
    segmentation: Union[f32['... K'], i32['... 1']],
    palette: Optional[f32['K 3']] = None,
    color_space: str = 'oklab',
    img: Union[f32['... c'], ui8['... c']] = None,
    hard: bool = False,
    entropy: bool = False,
    edges: bool = False,
    edge_lightness: float = 1.0,
    image_contrast: float = 0.6,
) -> f32['... 3']:
  """Function for various kinds of instance segmentation visualization.

  Args:
    segmentation: the segmentation to visualize either as per-pixel
      probabilities f32[... nr_segments] or as per-pixel indices i32[..., 1]
    palette: Color palette to use as a float array of RGB colors f32[K, 3]
      Optional, defaults to a palette of uniform hues in OKLAB space
    color_space: The color-space used for interpolating soft segmentations.
      Can be one of ['rgb', 'xyz', 'lab', 'oklab', 'hsv', 'lch'].
      Defaults to 'oklab'.
    img: The original image either as f32[..., c] or ui8[..., c].
      Optional. If given then the segmentation colors are multiplied with the
      lightness values of this image.
    hard: (bool) Whether to argmax the segmentations before plotting.
      Defaults to False. Only has an effect on soft segmentations.
    entropy: (bool) Whether to scale the lightness of the plot in proportion
      to the (normalized) per-pixel entropy of the soft-segmentation.
    edges: (bool) Whether to highlight the edges of the segments.
      Defaults to False. Only makes sense for hard=True.
    edge_lightness: (float) The lightness of the edges between 0.0 and 1.0.
      Defaults to 1.0.
    image_contrast: Re-scale the contrast of img before superimposing.
      Between 1.0 (full contrast) and 0.0 (all gray). Defaults to 0.6.
  Returns:
    The visualization as an RGB image.
  """

  nr_colors = palette.shape[0] if palette is not None else None
  soft_seg = maybe_convert_to_wide_form(segmentation, nr_colors)
  hard_seg = convert_to_one_hot(segmentation, nr_colors)
  k = soft_seg.shape[-1]

  if palette is None:
    palette = palette_lch(k)
  assert k <= palette.shape[0]

  if hard:
    rgb_segmentation = hard_seg @ palette[:k]
  else:
    to_colorspace, from_colorspace = COLOR_SPACES[color_space]
    palette_cs = to_colorspace(palette)
    rgb_segmentation = from_colorspace(soft_seg @ palette_cs[:k])

  light = np.ones_like(rgb_segmentation[..., 0:1])

  if img is not None:
    assert img.shape[-1] in {1, 3}
    assert img.shape[:-1] == segmentation.shape[:-1]
    if img.dtype.type == np.uint8:
      img = img / 255.

    img_gray = img.mean(-1, keepdims=True)

    light *= img_gray * image_contrast + (1-image_contrast) / 2

  if entropy:
    light *= 1.0 - norm_entropy(soft_seg)

  if edges:
    grad = np.gradient(hard_seg.argmax(-1)[..., None], axis=(-3, -2))
    edge = (np.abs(grad[0]) + np.abs(grad[1]) != 0).astype(np.float32)
    edge_blur = blur_image(edge, sigma=1.0)
    edge = np.maximum(edge, edge_blur)
    light = light * (1 - edge) + edge * edge_lightness

  return rgb_segmentation * light
