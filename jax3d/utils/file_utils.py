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

"""File-related utils."""

import contextlib
import functools
import io
import typing
from typing import Iterator, Optional

from etils import epath
from etils.epath import Path, PathLike  # pylint: disable=g-multiple-import
from jax3d.utils import dataclass_utils


class PathField(
    dataclass_utils.DataclassField[Optional[PathLike], Optional[Path]]
):
  """Descriptor which converts `str` to pathlib-like Path.

  Meant to be used in dataclasses (like `dataclasses.field`) to accept `str` as
  input, and convert them to pathlib-like objects.

  Example:

  ```python
  @dataclasses.dataclass
  class MyData:
    root_dir: epath.Path = j3d.utils.PathField()


  my_data = MyData(root_dir='/path/to/file')   # `str` as input
  # `str` is automatically converted to pathlib-like abstraction:
  my_data.root_dir.joinpath('file.txt').read_text()
  ```

  """

  def _validate(self, value: Optional[PathLike]) -> Optional[Path]:
    return None if value is None else Path(value)


@functools.lru_cache()
def j3d_dir() -> Path:
  """Root directory of `jax3d/`."""
  path = epath.resource_path('jax3d')
  return typing.cast(Path, path)


@functools.lru_cache()
def nf_dir() -> Path:
  """Root directory for `jax3d/nerfstatic/`."""
  return j3d_dir() / 'nerfstatic'


@contextlib.contextmanager
def open_seekable(path: PathLike, mode: str) -> Iterator[io.BytesIO]:
  """Same as `path.open('rb')`, but write to intermediate buffer.

  Rather than reading/writing directly to the file, file operations are applied
  on an in-memory buffer. This require the full file to be loaded in-memory.

  This is useful when file operation requires `f.seek()` which is not supported
  on some file systems.

  Args:
    path: Path on which save the value
    mode: `rb` or `wb`

  Yields:
    The file-like object on which write.
  """
  path = Path(path)
  if mode == 'rb':
    buffer = io.BytesIO(path.read_bytes())
  elif mode == 'wb':
    buffer = io.BytesIO()
  else:
    raise ValueError(f'Unsuported mode: {mode}')
  yield buffer
  if mode == 'wb':
    path.write_bytes(buffer.getvalue())
