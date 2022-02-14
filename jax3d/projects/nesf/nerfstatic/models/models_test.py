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

"""Tests for models."""

import json

from jax3d.projects.nesf.nerfstatic.models import models


def test_nerf_params_to_dict():
  # Ensure fields are converted to serializable types.
  params = models.NerfParams().to_dict()
  assert params['background'] == 'WHITE'
  assert params['unet_activation_fn'] == 'relu'
  assert params['skip_layer'] == 4


def test_nerf_params_to_dict_is_serializable():
  # Ensure dict is serializable as JSON.
  params = models.NerfParams()
  params_dict = params.to_dict()
  json.dumps(params_dict)


def test_nerf_params_round_trip():
  # Ensure params can be restored from JSON.
  params_before = models.NerfParams()
  params_after = models.NerfParams.from_dict(params_before.to_dict())
  assert params_before == params_after
