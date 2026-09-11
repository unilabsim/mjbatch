"""Metadata and recompute semantics for per-simulation model fields."""

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping

import mujoco
import numpy as np


class RecomputeLevel(IntEnum):
  """Derived model constants refreshed after a model-field write."""

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


@dataclass(frozen=True)
class ModelFieldSpec:
  """Public metadata for one per-simulation model field."""

  name: str
  shape: tuple[int, ...]
  dtype: np.dtype[Any]
  writable: bool
  asset: bool
  recompute: RecomputeLevel

  @property
  def derived_fields(self) -> tuple[str, ...]:
    return self.recompute.derived_fields


def _is_writable(name: str, dtype: np.dtype[Any]) -> bool:
  if name in _READ_ONLY_MODEL_FIELDS or name.endswith("plugin") or (name[:1].isupper() and name[1:2] == "_"):
    return False
  if dtype in (np.dtype(np.int8), np.dtype(np.int64)):
    return False
  if name.endswith(_WRITABLE_ID_SUFFIXES):
    return True
  if name.endswith(("adr", "num", "id", "sameframe", "simple", "signature")):
    return False
  return "_rowadr" not in name and "_colind" not in name and "_rownnz" not in name and "_diag" not in name


def build_model_fields(model: mujoco.MjModel) -> Mapping[str, ModelFieldSpec]:
  """Build immutable metadata for the arrays and options accepted by ``expand``."""
  specs: dict[str, ModelFieldSpec] = {}
  for name in dir(model):
    if name.startswith("_"):
      continue
    value = getattr(model, name)
    if not isinstance(value, np.ndarray):
      continue
    asset = name.startswith(_ASSET_PREFIXES)
    specs[name] = ModelFieldSpec(
      name=name,
      shape=tuple(value.shape),
      dtype=value.dtype,
      writable=not asset and _is_writable(name, value.dtype),
      asset=asset,
      recompute=_RECOMPUTE_BY_FIELD.get(name, RecomputeLevel.NONE),
    )

  integer, floating = np.dtype(np.int32), np.dtype(np.float64)
  for name in dir(model.opt):
    if name.startswith("_") or name in specs:
      continue
    value = getattr(model.opt, name)
    if isinstance(value, np.ndarray):
      shape, dtype = tuple(value.shape), value.dtype
    elif isinstance(value, bool):
      shape, dtype = (), np.dtype(np.bool_)
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
    )
  return MappingProxyType(specs)
