# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Iterator, Mapping, TypeAlias

import mujoco
import numpy as np

from mjbatch._bindings import Batch as _Batch

_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}  # by mjtJoint: free, ball, slide, hinge
_DOF_WIDTH = {0: 6, 1: 3, 2: 1, 3: 1}


class RecomputeLevel(IntEnum):
  """Derived model constants that must be refreshed after a model-field write."""

  NONE = 0
  SET_CONST_FIXED = 1
  SET_CONST_0 = 2
  SET_CONST = 3

  @property
  def derived_fields(self) -> tuple[str, ...]:
    return _DERIVED_FIELDS[self]


_DERIVED_FIELDS = {
  RecomputeLevel.NONE: (),
  RecomputeLevel.SET_CONST_FIXED: ("body_subtreemass",),
  RecomputeLevel.SET_CONST_0: (
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
  ),
}
_DERIVED_FIELDS[RecomputeLevel.SET_CONST] = (
  _DERIVED_FIELDS[RecomputeLevel.SET_CONST_FIXED] + _DERIVED_FIELDS[RecomputeLevel.SET_CONST_0]
)

_RECOMPUTE_BY_FIELD = {
  "body_gravcomp": RecomputeLevel.SET_CONST_FIXED,
  "body_pos": RecomputeLevel.SET_CONST_0,
  "body_quat": RecomputeLevel.SET_CONST_0,
  "qpos0": RecomputeLevel.SET_CONST_0,
  "dof_armature": RecomputeLevel.SET_CONST_0,
  "tendon_armature": RecomputeLevel.SET_CONST_0,
  "body_mass": RecomputeLevel.SET_CONST,
  "body_ipos": RecomputeLevel.SET_CONST,
  "body_inertia": RecomputeLevel.SET_CONST,
  "body_iquat": RecomputeLevel.SET_CONST,
}

_ASSET_PREFIXES = ("mesh_", "hfield_", "tex_", "skin_", "bvh_", "oct_")
_WRITABLE_ID_SUFFIXES = ("dataid", "matid", "texid")
_READ_ONLY_MODEL_FIELDS = {"names", "names_map", "paths", "plugin"}
_ModelUpdateState: TypeAlias = "tuple[np.ndarray | slice, dict[str, np.ndarray]]"


@dataclass(frozen=True)
class Joint:
  qpos: np.ndarray
  qvel: np.ndarray


@dataclass(frozen=True)
class Actuator:
  ctrl: np.ndarray
  force: np.ndarray


@dataclass(frozen=True)
class Body:
  xpos: np.ndarray
  xquat: np.ndarray
  cvel: np.ndarray


@dataclass(frozen=True)
class Site:
  xpos: np.ndarray
  xmat: np.ndarray


@dataclass(frozen=True)
class ModelFieldSpec:
  """Public metadata for one per-simulation model field."""

  name: str
  shape: tuple[int, ...]
  dtype: np.dtype[Any]
  writable: bool
  asset: bool
  recompute: RecomputeLevel
  derived_fields: tuple[str, ...]


def _is_writable_model_array(name: str, dtype: np.dtype[Any]) -> bool:
  if name in _READ_ONLY_MODEL_FIELDS or (name[:1].isupper() and name[1:2] == "_"):
    return False
  if dtype in (np.dtype(np.int8), np.dtype(np.int64)):
    return False
  if name.endswith(_WRITABLE_ID_SUFFIXES):
    return True
  if name.endswith(("adr", "num", "id")):
    return False
  if name.endswith(("sameframe", "simple", "signature")):
    return False
  if "_rowadr" in name or "_colind" in name or "_rownnz" in name or "_diag" in name:
    return False
  return True


class Batch(_Batch):
  """See mjbatch._bindings.Batch. The named accessors return live (N, ...) views of
  the corresponding bound fields, like MjData's sensor(), joint(), body() and site().

  model is the template, for names, ids and sizes. The batch copied it at construction,
  so writes to it do not reach the simulations; expand() a field to change it per sim."""

  def __init__(
    self,
    model: mujoco.MjModel,
    num_sims: int,
    num_threads: int = 0,
    forward: bool = False,
  ) -> None:
    super().__init__(model, num_sims, num_threads, forward)
    self.model = model
    self._model_fields = self._build_model_fields(model)
    self._expanded_fields: dict[str, np.ndarray] = {}
    self._active_model_update: _ModelUpdateState | None = None
    self._python_mutex = threading.RLock()

  def _build_model_fields(self, model: mujoco.MjModel) -> Mapping[str, ModelFieldSpec]:
    specs: dict[str, ModelFieldSpec] = {}
    for name in dir(model):
      if name.startswith("_"):
        continue
      value = getattr(model, name)
      if not isinstance(value, np.ndarray):
        continue
      asset = name.startswith(_ASSET_PREFIXES)
      writable = not asset and _is_writable_model_array(name, value.dtype)
      recompute = _RECOMPUTE_BY_FIELD.get(name, RecomputeLevel.NONE)
      specs[name] = ModelFieldSpec(
        name=name,
        shape=tuple(value.shape),
        dtype=value.dtype,
        writable=writable,
        asset=asset,
        recompute=recompute,
        derived_fields=recompute.derived_fields,
      )

    integer = np.dtype(np.int32)
    floating = np.dtype(np.float64)
    boolean = np.dtype(np.bool_)
    for name in dir(model.opt):
      if name.startswith("_") or name in specs:
        continue
      value = getattr(model.opt, name)
      if isinstance(value, np.ndarray):
        shape, dtype = tuple(value.shape), value.dtype
      elif isinstance(value, bool):
        shape, dtype = (), boolean
      elif isinstance(value, int):
        shape, dtype = (), integer
      elif isinstance(value, float):
        shape, dtype = (), floating
      else:
        continue
      specs[name] = ModelFieldSpec(
        name=name,
        shape=shape,
        dtype=dtype,
        writable=True,
        asset=False,
        recompute=RecomputeLevel.NONE,
        derived_fields=(),
      )
    return MappingProxyType(specs)

  def _normalize_ids(self, ids: Any) -> np.ndarray | None:
    if ids is None:
      return None
    array = np.ascontiguousarray(ids)
    if array.ndim != 1:
      raise ValueError("ids must be one-dimensional")
    if array.dtype == bool:
      if array.shape[0] != self.num_sims:
        raise ValueError("a boolean ids mask must have num_sims entries")
      return array
    if array.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
      raise ValueError("ids must be int32, int64 or a bool mask")
    checked = array.astype(np.int64, copy=False)
    if (checked.size and (checked[0] < 0 or checked[-1] >= self.num_sims)) or np.any(np.diff(checked) <= 0):
      raise ValueError("ids must be sorted, unique and in range")
    return array

  def model_field_specs(self) -> Mapping[str, ModelFieldSpec]:
    """Return immutable metadata for every model field exposed by ``expand``."""
    return self._model_fields

  def expand(self, name: str, dtype: Any = None) -> np.ndarray:
    spec = self._model_fields.get(name)
    if spec is not None and not spec.writable:
      reason = "asset data" if spec.asset else "read-only structural data"
      raise ValueError(f"{name} is {reason}")
    view = super().expand(name, dtype)
    with self._python_mutex:
      self._expanded_fields[name] = view
      update = self._active_model_update
      if update is not None:
        update[1][name] = view.copy()
    return view

  @contextmanager
  def model_update(self, ids: Any = None) -> Iterator[None]:
    """Write model fields, then perform one conservative recompute on exit.

    Selection semantics match the other batch operations. Changes are compared on
    the selected rows only: writes to unselected rows are restored on exit. Writing
    only fields whose recompute level is ``NONE`` does not call ``set_const``; any
    higher level currently uses one full stock MuJoCo ``mj_setConst`` pass, which is
    a conservative superset of that level.
    """
    ids_array = self._normalize_ids(ids)
    selection: np.ndarray | slice = slice(None) if ids_array is None else ids_array
    with self._python_mutex:
      if self._active_model_update is not None:
        raise RuntimeError("model_update calls cannot be nested")
      baseline = {name: view.copy() for name, view in self._expanded_fields.items()}
      self._active_model_update = (selection, baseline)
    try:
      yield
    finally:
      with self._python_mutex:
        update = self._active_model_update
        baseline = {} if update is None else update[1]
        self._active_model_update = None
        changed_levels = [
          self._model_fields[name].recompute
          for name, before in baseline.items()
          if not np.array_equal(self._expanded_fields[name][selection], before[selection])
        ]
        if ids_array is not None:
          unselected = ~ids_array if ids_array.dtype == bool else np.ones(self.num_sims, bool)
          unselected[ids_array] = False
          for name, before in baseline.items():
            self._expanded_fields[name][unselected] = before[unselected]
      level = max(changed_levels, default=RecomputeLevel.NONE)
      if level != RecomputeLevel.NONE and ids_array is not None and ids_array.size == 0:
        level = RecomputeLevel.NONE
      if level != RecomputeLevel.NONE:
        super().set_const(ids_array)

  def set_const(self, ids: Any = None) -> None:
    ids_array = self._normalize_ids(ids)
    with self._python_mutex:
      if self._active_model_update is not None:
        raise RuntimeError("call set_const after model_update exits, not inside it")
    super().set_const(ids_array)

  def sensor(self, name: str, dtype: Any = None) -> np.ndarray:
    s = self.model.sensor(name)
    return self.bind("sensordata", dtype)[:, s.adr[0] : s.adr[0] + s.dim[0]]

  def joint(self, name: str, dtype: Any = None) -> Joint:
    j = self.model.joint(name)
    nq, nv = _QPOS_WIDTH[int(j.type[0])], _DOF_WIDTH[int(j.type[0])]
    return Joint(
      self.bind("qpos", dtype)[:, j.qposadr[0] : j.qposadr[0] + nq],
      self.bind("qvel", dtype)[:, j.dofadr[0] : j.dofadr[0] + nv],
    )

  def actuator(self, name: str, dtype: Any = None) -> Actuator:
    i = self.model.actuator(name).id
    return Actuator(self.bind("ctrl", dtype)[:, i], self.bind("actuator_force", dtype)[:, i])

  def body(self, name: str, dtype: Any = None) -> Body:
    i = self.model.body(name).id
    return Body(
      self.bind("xpos", dtype)[:, i],
      self.bind("xquat", dtype)[:, i],
      self.bind("cvel", dtype)[:, i],
    )

  def site(self, name: str, dtype: Any = None) -> Site:
    i = self.model.site(name).id
    return Site(self.bind("site_xpos", dtype)[:, i], self.bind("site_xmat", dtype)[:, i])

  def _nsel(self, ids: Any) -> int:
    if ids is None:
      return self.num_sims
    ids = np.asarray(ids)
    return int(ids.sum()) if ids.dtype == bool else len(ids)

  def jac_site(  # pyright: ignore[reportIncompatibleMethodOverride]  # name-based allocating wrapper
    self, name: str, ids: Any = None
  ) -> tuple[np.ndarray, np.ndarray]:
    """World-frame position/rotation Jacobians of a site, per selected simulation.

    Returns (jacp, jacr) with shape (nsel, 3, nv). Runs kinematics and comPos
    only, not mj_forward."""
    i = self.model.site(name).id
    n = self._nsel(ids)
    jacp = np.zeros((n, 3, self.model.nv))
    jacr = np.zeros((n, 3, self.model.nv))
    super().jac_site(i, jacp, jacr, ids)
    return jacp, jacr

  def sample_hfield(  # pyright: ignore[reportIncompatibleMethodOverride]  # name-based allocating wrapper
    self, geom: str, body: str, offsets: np.ndarray, ids: Any = None
  ) -> np.ndarray:
    """Bilinear hfield heights at world-frame XY offsets around a body's origin.

    offsets: (npoint, 2). Returns (nsel, npoint) local heights. All simulations
    sample the template's hfield. Runs kinematics only, not mj_forward."""
    g = self.model.geom(geom).id
    b = self.model.body(body).id
    offsets = np.ascontiguousarray(offsets, dtype=np.float64)
    out = np.zeros((self._nsel(ids), offsets.shape[0]))
    super().sample_hfield(g, b, offsets, out, ids)
    return out
