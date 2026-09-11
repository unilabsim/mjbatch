"""Explicit routing for simulations with incompatible model layouts."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import numpy as np

if TYPE_CHECKING:
  from mjbatch import Batch


@dataclass(frozen=True)
class TopologyGroup:
  """One model layout and the fixed global simulation ids assigned to it."""

  name: str
  batch: Batch
  global_ids: np.ndarray

  @property
  def num_sims(self) -> int:
    return int(self.global_ids.size)

  @property
  def nstate(self) -> int:
    return self.batch.nstate

  @property
  def state(self) -> np.ndarray:
    """This group's local ``(num_sims, nstate)`` integration-state rows."""
    return self.batch.bind("state")

  def bind(self, name: str, dtype: Any = None) -> np.ndarray:
    return self.batch.bind(name, dtype)

  def expand(self, name: str, dtype: Any = None) -> np.ndarray:
    return self.batch.expand(name, dtype)


class ModelAffineBatch:
  """Route global ids to batches that own incompatible model topologies."""

  def __init__(
    self,
    batches: Sequence[Batch],
    assignments: Sequence[np.ndarray] | None = None,
    *,
    names: Sequence[str] | None = None,
  ) -> None:
    if not batches:
      raise ValueError("at least one topology group is required")
    if names is None:
      names = [f"group_{i}" for i in range(len(batches))]
    if len(names) != len(batches) or len(set(names)) != len(names):
      raise ValueError("group names must be unique and match the group count")
    if assignments is None:
      offset = 0
      prepared = []
      for batch in batches:
        ids = np.arange(offset, offset + batch.num_sims, dtype=np.int32)
        offset += batch.num_sims
        prepared.append(ids)
    else:
      if len(assignments) != len(batches):
        raise ValueError("assignments must match the group count")
      prepared = [np.array(ids, copy=True) for ids in assignments]

    seen: list[int] = []
    groups: list[TopologyGroup] = []
    for name, batch, ids in zip(names, batches, prepared, strict=True):
      if ids.ndim != 1 or ids.shape[0] != batch.num_sims:
        raise ValueError(f"group {name!r} assignment must have one id per local sim")
      if ids.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise ValueError(f"group {name!r} assignment ids must be int32 or int64")
      checked = ids.astype(np.int64, copy=False)
      if np.any(np.diff(checked) <= 0):
        raise ValueError(f"group {name!r} assignment ids must be sorted and unique")
      seen.extend(checked.tolist())
      ids.setflags(write=False)
      groups.append(TopologyGroup(str(name), batch, ids))
    if sorted(seen) != list(range(len(seen))):
      raise ValueError("group assignments must cover every global id exactly once")

    self._groups = tuple(groups)
    self._by_name = {group.name: group for group in groups}

  @property
  def groups(self) -> tuple[TopologyGroup, ...]:
    return self._groups

  @property
  def num_groups(self) -> int:
    return len(self._groups)

  @property
  def num_sims(self) -> int:
    return sum(group.num_sims for group in self._groups)

  def group(self, key: str | int) -> TopologyGroup:
    if isinstance(key, str):
      try:
        return self._by_name[key]
      except KeyError as error:
        raise KeyError(f"unknown topology group {key!r}") from error
    if key < 0 or key >= self.num_groups:
      raise IndexError("topology group index out of range")
    return self._groups[key]

  def __getitem__(self, key: str | int) -> TopologyGroup:
    return self.group(key)

  def _normalize_ids(self, ids: Any) -> np.ndarray | None:
    if ids is None:
      return None
    array = np.ascontiguousarray(ids)
    if array.ndim != 1:
      raise ValueError("ids must be one-dimensional")
    if array.dtype == bool:
      if array.shape[0] != self.num_sims:
        raise ValueError("a boolean ids mask must have num_sims entries")
      return np.flatnonzero(array).astype(np.int32)
    if array.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
      raise ValueError("ids must be int32, int64 or a bool mask")
    checked = array.astype(np.int64, copy=False)
    if checked.size and (checked[0] < 0 or checked[-1] >= self.num_sims) or np.any(np.diff(checked) <= 0):
      raise ValueError("ids must be sorted, unique and in range")
    return array

  def _split_ids(self, ids: Any) -> list[np.ndarray | None]:
    global_ids = self._normalize_ids(ids)
    if global_ids is None:
      return [None] * self.num_groups
    result: list[np.ndarray | None] = []
    for group in self._groups:
      positions = np.searchsorted(group.global_ids, global_ids)
      valid = positions < group.global_ids.size
      valid[valid] = group.global_ids[positions[valid].astype(np.int64, copy=False)] == global_ids[valid]
      result.append(positions[valid].astype(np.int32))
    return result

  def step(self, ids: Any = None, nstep: int = 1, history: Any = None) -> None:
    if history is not None:
      raise ValueError("global history is unsupported across topologies; collect state from each group")
    self._dispatch(ids, lambda group, local_ids: group.batch.step(local_ids, nstep=nstep))

  def forward(self, ids: Any = None) -> None:
    self._dispatch(ids, lambda group, local_ids: group.batch.forward(local_ids))

  def reset(self, ids: Any = None, keyframe: int = -1) -> None:
    if keyframe >= 0 and any(keyframe >= group.batch.model.nkey for group in self.groups):
      raise ValueError("keyframe out of range for at least one topology group")
    self._dispatch(ids, lambda group, local_ids: group.batch.reset(local_ids, keyframe=keyframe))

  def set_const(self, ids: Any = None) -> None:
    self._dispatch(ids, lambda group, local_ids: group.batch.set_const(local_ids))

  @contextmanager
  def model_update(self, *fields: str, ids: Any = None) -> Iterator[None]:
    """Run declared per-group model updates with one global selection boundary."""
    selections = self._split_ids(ids)
    with ExitStack() as stack:
      for group, local_ids in zip(self.groups, selections, strict=True):
        stack.enter_context(group.batch.model_update(*fields, ids=local_ids))
      yield

  def _dispatch(
    self,
    ids: Any,
    operation: Callable[[TopologyGroup, np.ndarray | None], None],
  ) -> None:
    selections = self._split_ids(ids)
    for group, local_ids in zip(self.groups, selections, strict=True):
      if local_ids is not None and local_ids.size == 0:
        continue
      operation(group, local_ids)
