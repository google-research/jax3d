# Copyright 2024 The jax3d Authors.
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

"""A class to track accumulated training/eval/... time."""
import collections
import copy
import threading
import time

from typing import Dict, Iterable, Optional, Union

try:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  JAX_AVAILABLE = True
except ImportError:
  JAX_AVAILABLE = False

try:
  import flax.struct  # pylint: disable=g-import-not-at-top
  FLAX_AVAILABLE = True
except ImportError:
  FLAX_AVAILABLE = False

TIME_UNITS = {
    "nanoseconds": 1,
    "ns": 1,
    "microseconds": 1e3,
    "milliseconds": 1e6,
    "ms": 1e6,
    "seconds": 1e9,
    "sec": 1e9,
    "minutes": 60 * 1e9,
    "min": 60 * 1e9,
    "hours": 60 * 60 * 1e9,
    "hrs": 60 * 60 * 1e9,
    "days": 24 * 60 * 60 * 1e9,
    "weeks": 7 * 24 * 60 * 60 * 1e9,
    "months": 30.437 * 24 * 60 * 60 * 1e9,
    "years": 365.2425 * 24 * 60 * 60 * 1e9,
    "yrs": 365.2425 * 24 * 60 * 60 * 1e9,
}

if FLAX_AVAILABLE:
  # if flax is available define a ChronoState that can be stored in checkpoints.

  @flax.struct.dataclass
  class ChronoState:
    """State of a Chrono object as a flax dataclass for checkpointing."""
    accumulated_times: Dict[str, jnp.ndarray] = flax.struct.field(
        pytree_node=True, default_factory=dict)

    @classmethod
    def from_nanosec_dict(cls,
                          accumulated_times: Dict[str, int],
                          num_devices: Optional[int] = None) -> "ChronoState":
      """Create a ChronoState from a label->nanoseconds mapping."""
      num_devices = num_devices or jax.device_count()
      # convert int times to jnp.arrays for replication / dereplication
      # also store as two int32 because jnp silently converts int64 to int32
      accumulated_times_jnp = {
          k: to_int32_array(v, num_devices)
          for k, v in accumulated_times.items()
      }
      return cls(accumulated_times=accumulated_times_jnp)

    def to_nanosec_dict(self) -> Dict[str, int]:
      # convert jnp.arrays back to int
      return {k: from_int32_array(v) for k, v in self.accumulated_times.items()}

  def to_int32_array(ns: int, num_devices: int) -> jnp.ndarray:
    # save as [whole seconds, residual as ns]
    return jnp.array([[ns // 1e9, ns % 1e9]] * num_devices, dtype=jnp.int32)

  def from_int32_array(arr: jnp.ndarray) -> int:
    if arr.ndim == 2:
      arr = arr[0]  # remove device dimension if present
    return int(arr[0] * 1e9 + arr[1])


class Chrono:
  """Accumulate time in several buckets, for reporting in training loops.

  Usage:
  This example of using Chrono in a typical training loop, will log accumulated
  time in four buckets: "init", "train", "checkpoint", and "total".
  It will also discard the first two "train" ticks (warmup), to account for the
  fact that they often are much slower due to compilation etc.
  The exported summary will contain entries for (depending on device type):
    - hours_init, hours_train, hours_checkpoint, hours_total
    - hours_total_per_step, hours_train_per_step
    - hours_total_per_TPU_v3, hours_train_per_TPU_v3
    - steps_per_second
    - hours_uptime

  ```python
  # at the very begining of the training code
  chrono = Chrono(warmup={"train": 2})  # ignore the first two "train" ticks

  # restore state from checkpoint
  ...
  chrono.restore_state(state.chrono)
  ...

  # after init is done, just before the training loop
  chrono.tick("init")
  for step in range(max_steps):
    ...
    # at the end of each training step
    chrono.tick("train")

    # checkpoint saving
    if step % save_chkpt_every == 0:
      # add chrono state to the checkpoint state
      state.chrono = chrono.get_state()
      ...

      # after saving the checkpoint is done
      chrono.tick("checkpoint")

    # export the summary fo measurements
    for k, v in chrono.summary(step=step).items():
      summary_writer.scalar(f"stats/{k}", v, step)
  ```
  """

  def __init__(
      self,
      warmup: Optional[Dict[str, int]] = None,
      total_label: str = "total",
      train_label: str = "train",
  ) -> None:
    """Create a new Chrono object.

    Args:
      warmup: Number of initial ticks to skip in per label, e.g. {"train": 2}.
        Useful to ignore the first few training steps which often are
        unrepresentatively slow due to compilation, caching, etc.
      total_label: The name of the bucket for tracking the total elapsed time.
        Unless set to None, this label is added to each call of tick(). Defaults
        to "total".
      train_label: The name of the bucket for tracking the training time. This
        is only used in the summary which by default only returns
        "{}_steps_per_second" for total_label and train_label. Defaults to
        "train".
    """
    self._lock = threading.Lock()
    with self._lock:
      self.total_label = total_label
      self.train_label = train_label
      self._accumulated_times = collections.defaultdict(int)
      self.warmup = self._validate_warmup(warmup)
      self._ticks_to_skip = copy.copy(self.warmup)

      self.program_start_time = self.prev_time = time.monotonic_ns()
      self._uptime = 0

  def tick(self, labels: Union[str, Iterable[str]] = ()) -> None:
    """A chronometer tick.

    Call repeatedly to accumulate elapsed time, and to (optionally) label the
    time elapsed since the last call to tick().
    For example calling tick("train") will add the time passed since the
    last tick() both to the "total" and the "train" time bucket.
    Intended usecase is to call tick("train") after each training step,
    tick("eval") after each evaluation step, etc...

    Args:
     labels: (str or Iterable[str]) A single label, or a list of labels to be
       applied to the passed time since the last call to tick().
    """
    now = time.monotonic_ns()
    dt = now - self.prev_time

    labels = {labels} if isinstance(labels, str) else set(labels)
    if self.total_label is not None:
      labels |= {self.total_label}  # always log the total time

    with self._lock:
      self._uptime = now - self.program_start_time

      for l in labels:
        # check if we want to skip tick due to warmup
        if self._ticks_to_skip.get(l, 0) > 0:
          self._ticks_to_skip[l] -= 1
          continue

        self._accumulated_times[l] += dt
      self.prev_time = now

  @property
  def uptime(self) -> float:
    """The current uptime (time since `__init__`) in nanoseconds.

    Note: this time is not persisted through checkpointing, and thus resets to
    0 each time the job is rescheduled.
    """
    return self._uptime

  def accumulated_times(self,
                        time_unit="hours",
                        subtract_warmup: bool = False,
                        per_labels: Optional[Iterable[str]] = None,
                        **denominators) -> Dict[str, float]:
    """Return dict of accumulated times in hours (by default)."""

    normalize_by = get_normalizer_from_time_unit_str(time_unit)

    # add entries for each label
    # e.g. "hours_total", "hours_training", and "hours_eval"
    times = {
        f"{time_unit}_{k}": v / normalize_by
        for k, v in self._accumulated_times.items()
    }

    if per_labels is None:
      per_labels = self._accumulated_times.keys()
    # add entries for each passed kwarg
    # e.g. for step=... we would add "hours_total_per_step"
    for dk, dv in denominators.items():
      for k in per_labels:
        denom_value = dv if not subtract_warmup else dv - self.warmup.get(k, 0)
        t = self._accumulated_times[k] / normalize_by
        if denom_value > 0:
          times[f"{time_unit}_{k}_per_{dk}"] = t / denom_value
    return times

  def steps_per(self,
                step: int,
                label: str,
                time_unit: str = "seconds",
                subtract_warmup: bool = True) -> Dict[str, float]:
    """Compute steps per sec for a given label, taking warmup into account."""
    step = step if not subtract_warmup else step - self.warmup.get(label, 0)
    normalize_by = normalize_by = get_normalizer_from_time_unit_str(time_unit)
    total_time = self._accumulated_times[label] / normalize_by
    if total_time == 0 or step < 0:
      return {}
    return {f"{label}_steps_per_{time_unit}": step / total_time}

  def summary(
      self,
      step: Optional[int] = None,
      prefix: str = "",
      time_unit: str = "hours",
      per_labels: Iterable[str] = (
          "total",
          "train",
      ),
      steps_per_time_unit: str = "sec",
      subtract_warmup_from_steps=True,
      **kwargs,
  ) -> Dict[str, float]:
    """Compile a summary dict with various information.

    Args:
      step: The current step number (optional). If passed, this is used to
        compute "{LABEL}_hours_per_step" and "{LABEL}_steps_per_sec" for each
        LABEL in the per_label argument (default "total" and "train").
      prefix: Optional prefix to prepend to every key. Defaults to "".
      time_unit: Unit of time for {UNIT}_{LABEL} and {UNIT}_{LABEL}_per_step.
        E.g. "seconds", "minutes", "days". Defaults to "hours".
      per_labels: List of labels for which to compute "{LABEL}_hours_per_step"
        (if step is given) and "{LABEL}_hours_per_{DEVICE}" (if jax device is
        available).
      steps_per_time_unit: Time unit used for "{LABEL}_steps_per_{UNIT}".
        Defaults to "sec".
      subtract_warmup_from_steps: If true, then the warmup for each label is
        subtracted from the number of steps for computing the corresponding
        "{LABEL}_hours_per_step" and "{LABEL}_steps_per_sec".
      **kwargs: additional denominators for adding entries of the form
        "{LABEL}_hours_per_{DENOM}".
    Returns:
      A dictionary with lots of entries.
      This includes (depending on arguments and used labels):
      - hours_total, hours_train, ...
      - hours_total_per_step, hours_train_per_step
      - hours_total_per_TPU_v3, hours_train_per_TPU_v3
      - steps_per_second
      - hours_uptime
    """

    times = self.accumulated_times(time_unit=time_unit, **kwargs)

    if step is not None:
      times.update(
          self.accumulated_times(
              time_unit=time_unit,
              per_labels=per_labels,
              subtract_warmup=subtract_warmup_from_steps,
              step=step))
      for label in per_labels:
        times.update(self.steps_per(step, label, steps_per_time_unit))

    if JAX_AVAILABLE:
      num_cores = jax.device_count()  # Global device count
      device_kind = str(jax.devices()[0].device_kind).replace(" ", "_")
      times.update(
          self.accumulated_times(
              time_unit=time_unit,
              per_labels=per_labels,
              **{device_kind: num_cores}))

    normalize_by = get_normalizer_from_time_unit_str(time_unit)
    times[f"{time_unit}_uptime"] = (self.uptime / normalize_by)
    return {f"{prefix}{k}": v for k, v in times.items()}

  @staticmethod
  def _validate_warmup(warmup: Optional[Dict[str, int]]) -> Dict[str, int]:
    """Check that warmup values are non-negative int."""
    if warmup is None:
      return {}

    for label, num_ticks_to_ignore in warmup.items():
      if not (isinstance(num_ticks_to_ignore, int) and
              num_ticks_to_ignore >= 0):
        raise ValueError("warmup values must be non-negative int, but got "
                         f"{num_ticks_to_ignore} of "
                         f"type {type(num_ticks_to_ignore)} for '{label}'")
    return warmup

  if FLAX_AVAILABLE:

    def restore_state(self, state: ChronoState) -> None:
      """Restore from a ChronoState as exported by get_state()."""
      # pylint: disable=protected-access
      with self._lock:
        dict_state = state.to_nanosec_dict()
        self._accumulated_times = collections.defaultdict(int, **dict_state)
        self._ticks_to_skip = copy.copy(self.warmup)

    def get_state(self, num_devices: Optional[int] = None) -> ChronoState:
      """Serialize state as a ChronoState for storing in a checkpoint."""
      return ChronoState.from_nanosec_dict(
          self._accumulated_times, num_devices=num_devices)


def get_normalizer_from_time_unit_str(time_unit: str) -> float:
  time_unit = time_unit.lower()  # ignore case
  if time_unit not in TIME_UNITS:
    raise ValueError(f"Unknown time unit: {time_unit}. "
                     f"Must be one of {TIME_UNITS.keys()}")
  return float(TIME_UNITS[time_unit])
