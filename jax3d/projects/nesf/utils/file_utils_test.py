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

"""Tests for jax3d.projects.nesf.utils.file_utils."""

import dataclasses
import os
import pathlib
from typing import Any

import jax3d.projects.nesf as j3d
import pytest


@pytest.mark.parametrize('frozen', [True, False])
@pytest.mark.parametrize('default', [
    None, '/some/path', dataclasses.MISSING, pathlib.Path('/some/pathlib'),
])
def test_path_field(frozen: bool, default: Any):

  # With or without default
  if default is dataclasses.MISSING:
    default_args = ()
  else:
    default_args = (default,)

  @dataclasses.dataclass(frozen=frozen)
  class A:
    path: Any = j3d.utils.PathField(*default_args)

  # Check that the default value works correctly.
  if default is dataclasses.MISSING:
    with pytest.raises(TypeError, match='missing .* argument'):
      A()

    with pytest.raises(
        AttributeError, match="type object '.*' has no attribute 'path'"
    ):
      _ = A.path
  elif default is None:
    a = A()
    assert a.path is None
  else:
    a = A()
    assert isinstance(a.path, j3d.Path)
    assert os.fspath(a.path) == os.fspath(default)

  # Explicitly setting the path in constructor should work.
  a = A('/my_path')
  assert isinstance(a.path, j3d.Path)
  assert os.fspath(a.path) == '/my_path'

  a = A(path=None)
  assert a.path is None

  # Updating the path should only work for non-frozen instances.
  if frozen:
    with pytest.raises(dataclasses.FrozenInstanceError):
      a.path = '/some/other/path'
  else:
    # Can overwrite the path.
    a.path = '/some/other/path'
    assert isinstance(a.path, j3d.Path)
    assert os.fspath(a.path) == '/some/other/path'

    a.path = None
    assert a.path is None

    a.path = pathlib.Path('/some/pathlib/path')
    assert isinstance(a.path, j3d.Path)
    assert os.fspath(a.path) == '/some/pathlib/path'


def test_nf_path():
  """Test the proper suffix only, since the prefix may vary."""
  assert j3d.j3d_dir().name == 'nesf'
  assert j3d.nf_dir().parts[-2:] == ('nesf', 'nerfstatic')


def test_open_seakable(tmp_path: pathlib.Path):
  """Test open_seakable."""
  tmp_path = tmp_path / 'file.bin'

  with j3d.utils.open_seekable(tmp_path, 'wb') as f:
    f.write(b'some content')

  with j3d.utils.open_seekable(tmp_path, 'rb') as f:
    assert f.read() == b'some content'
