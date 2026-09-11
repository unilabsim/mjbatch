# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from mjbatch._bindings import Batch as _Batch

_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}  # by mjtJoint: free, ball, slide, hinge
_DOF_WIDTH = {0: 6, 1: 3, 2: 1, 3: 1}


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
