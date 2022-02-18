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

"""Camera util."""

from __future__ import annotations

import dataclasses

from jax3d.visu3d import camera_spec
from jax3d.visu3d import ray as ray_lib

# TODO(epot): Complete implementation


@dataclasses.dataclass
class Camera:  # TODO(epot): Make this DataclassArray
  """Camera."""
  spec: camera_spec.CameraSpec
  ray: ray_lib.Ray

  @classmethod
  def from_ray(
      cls,
      *,
      ray: ray_lib.Ray,
      spec: camera_spec.CameraSpec,
  ) -> Camera:
    """Factory which create a camera from a ray.

    This assume the camera is pointing.

    Args:
      ray: Center ray of the camera.
      spec: Camera specifications.

    Returns:
      cam: Camera pointing to the ray.
    """
    return cls(spec=spec, ray=ray)
