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

"""GPU utilization monitor for train/eval jobs.

Invokes nvidia-smi regularly from a background process and averages samples to
track utilization over time.
"""

import multiprocessing as mp
import subprocess
import time

import numpy as np


def runner(nvidia_smi_path, interval, pipe):
  """Run nvidia-smi in a loop in the background."""
  while True:
    result = subprocess.run(nvidia_smi_path,
                            check=True,
                            stdout=subprocess.PIPE,
                            encoding="utf-8").stdout
    # Very hacky parsing
    lines = result.splitlines()
    lines = list(l[58:68] for l in lines[9::4])
    lines = lines[:list(l[0] for l in lines).index("-")]
    lines = list(l.strip()[:-1] for l in lines)
    utilizations = list(int(l) for l in lines)

    pipe.send(utilizations)
    time.sleep(interval)


class GPUMonitor:
  """Monitor object that runs nvidia-smi in the background."""

  def __init__(self, interval=2, nvidia_smi_path="/usr/lib/libcuda/nvidia-smi"):
    with open(nvidia_smi_path, "rb"):
      pass  # Will error if nvidia-smi not available

    self.receiver, self.sender = mp.Pipe()
    self.child = mp.Process(
        target=runner, args=(nvidia_smi_path, interval, self.sender))
    self.child.start()

  def get_utilization(self):
    """Get average utilization since the last call."""
    utilizations = []
    while self.receiver.poll():
      utilizations.append(self.receiver.recv())

    if utilizations:
      return np.mean(np.array(utilizations), axis=0)

    else:
      return None

  def stop(self):
    self.sender.close()
    self.receiver.close()
    self.child.kill()
