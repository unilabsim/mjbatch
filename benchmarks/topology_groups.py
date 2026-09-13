# SPDX-License-Identifier: Apache-2.0

"""Reproducible CPU benchmark for explicit topology-affine routing.

uv run python benchmarks/topology_groups.py --num-sims 512 --threads 8
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mjbatch import Batch, ModelAffineBatch

ONE_JOINT_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="one" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="arm" type="capsule" fromto="0 0 0 .5 0 0" size=".02" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge"/></actuator>
</mujoco>
"""

TWO_JOINT_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="proximal" pos="0 0 1">
      <joint name="hip" type="hinge" axis="0 1 0"/>
      <geom name="upper" type="capsule" fromto="0 0 0 .35 0 0" size=".02" mass=".6"/>
      <body name="distal" pos=".35 0 0">
        <joint name="knee" type="hinge" axis="0 1 0"/>
        <geom name="lower" type="capsule" fromto="0 0 0 .35 0 0" size=".02" mass=".4"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="hip"/></actuator>
</mujoco>
"""


def rss_kib() -> int:
  with Path("/proc/self/status").open() as status:
    for line in status:
      if line.startswith("VmRSS:"):
        return int(line.split()[1])
  return 0


def build(models: list[mujoco.MjModel], count: int, threads: int) -> ModelAffineBatch:
  per_group = count // 2
  batches = [
    Batch(models[0], per_group, num_threads=threads),
    Batch(models[1], count - per_group, num_threads=threads),
  ]
  return ModelAffineBatch(batches, names=["one_joint", "two_joint"])


def timed_cold_start(
  models: list[mujoco.MjModel], count: int, threads: int, repeats: int
) -> tuple[float, int]:
  elapsed: list[float] = []
  delta: list[int] = []
  for _ in range(repeats):
    gc.collect()
    before = rss_kib()
    start = time.perf_counter_ns()
    sharded = build(models, count, threads)
    elapsed.append((time.perf_counter_ns() - start) / 1e6)
    delta.append(rss_kib() - before)
    del sharded
  return statistics.median(elapsed), max(delta)


def time_step(sharded: ModelAffineBatch, calls: int) -> float:
  for group in sharded.groups:
    control = np.zeros((group.num_sims, group.batch.model.nu))
    group.bind("ctrl")[:] = control
  for _ in range(20):
    sharded.step()
  rates = []
  for _ in range(5):
    start = time.perf_counter_ns()
    for _ in range(calls):
      sharded.step()
    seconds = (time.perf_counter_ns() - start) / 1e9
    rates.append(sharded.num_sims * calls / seconds)
  return statistics.median(rates)


def time_field_update(sharded: ModelAffineBatch, rng: np.random.Generator, calls: int) -> tuple[float, float]:
  values = {
    group.name: rng.uniform(0.4, 1.2, (group.num_sims, group.batch.model.ngeom, 3))
    for group in sharded.groups
  }
  total_values = sum(array.size for array in values.values())
  for _ in range(3):
    with sharded.model_update("geom_friction"):
      for group in sharded.groups:
        group.expand("geom_friction")[:] = values[group.name]
  samples = []
  for _ in range(5):
    start = time.perf_counter_ns()
    for call in range(calls):
      scale = 0.5 + ((call % 2) * 0.1)
      with sharded.model_update("geom_friction"):
        for group in sharded.groups:
          group.expand("geom_friction")[:] = values[group.name] * scale
    samples.append((time.perf_counter_ns() - start) / 1e6 / calls)
  elapsed = statistics.median(samples)
  bytes_per_call = total_values * np.dtype(np.float64).itemsize
  return elapsed, bytes_per_call / (elapsed / 1000) / (1024 * 1024)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-sims", type=int, default=512)
  parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--calls", type=int, default=200)
  parser.add_argument("--json", type=Path)
  args = parser.parse_args()
  if args.num_sims < 2 or args.threads < 1 or args.repeats < 1 or args.calls < 1:
    parser.error("num-sims >= 2, threads/repeats/calls must all be >= 1")

  models = [
    mujoco.MjModel.from_xml_string(ONE_JOINT_XML),
    mujoco.MjModel.from_xml_string(TWO_JOINT_XML),
  ]
  sharded = build(models, args.num_sims, args.threads)
  rng = np.random.default_rng(0)
  cold_ms, cold_rss_delta = timed_cold_start(models, args.num_sims, args.threads, args.repeats)
  step_rate = time_step(sharded, args.calls)
  field_ms, field_mb_per_s = time_field_update(sharded, rng, args.calls)
  result: dict[str, Any] = {
    "configuration": {
      "num_sims": args.num_sims,
      "threads": args.threads,
      "repeats": args.repeats,
      "calls": args.calls,
      "seed": 0,
      "python": platform.python_version(),
      "mujoco": mujoco.__version__,
      "numpy": np.__version__,
      "group_nstates": [group.nstate for group in sharded.groups],
    },
    "cold_start_ms_median": round(cold_ms, 3),
    "cold_start_rss_delta_kib_max": cold_rss_delta,
    "step_sim_substeps_per_s": round(step_rate),
    "model_field_update_ms_per_call_median": round(field_ms, 3),
    "model_field_update_mb_per_s": round(field_mb_per_s),
    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  }
  print(json.dumps(result, indent=2, sort_keys=True))
  if args.json:
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
  main()
