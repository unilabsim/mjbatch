"""Compiler-coherent, same-layout mesh variant construction."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import mujoco
import numpy as np

_GEOM_VARIANT_FIELDS = (
  "geom_type",
  "geom_contype",
  "geom_conaffinity",
  "geom_matid",
  "geom_rgba",
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "geom_pos",
  "geom_quat",
)
_BODY_VARIANT_FIELDS = (
  "body_mass",
  "body_subtreemass",
  "body_inertia",
  "body_invweight0",
  "body_ipos",
  "body_iquat",
)
_DOF_VARIANT_FIELDS = ("dof_M0", "dof_invweight0", "dof_length")
_VARIANT_FIELDS = _GEOM_VARIANT_FIELDS + _BODY_VARIANT_FIELDS + _DOF_VARIANT_FIELDS

_LAYOUT_SCALARS = (
  "nq",
  "nv",
  "na",
  "nbody",
  "njnt",
  "nsite",
  "ncam",
  "nlight",
  "nmat",
  "npair",
  "nexclude",
  "neq",
  "ntendon",
  "nwrap",
  "nsensor",
  "nnumeric",
  "ntext",
  "ntuple",
  "nmocap",
  "nplugin",
  "nuser_body",
  "nuser_jnt",
  "nuser_geom",
  "nuser_site",
  "nuser_cam",
  "nuser_sensor",
  "nuser_tendon",
  "nuser_actuator",
)


@dataclass(frozen=True)
class VariantPack:
  """A canonical model plus per-variant, compiler-derived model-field rows."""

  model: mujoco.MjModel
  num_variants: int
  fields: Mapping[str, np.ndarray]

  def field_values(self, name: str) -> np.ndarray:
    """Return a copy of one field's ``(num_variants, ...)`` table."""
    if name not in self.fields:
      raise KeyError(f"unknown variant field {name!r}")
    return self.fields[name].copy()

  @classmethod
  def from_specs(
    cls,
    specs: Sequence[mujoco.MjSpec],
    *,
    slot_policy: str = "stable-name",
  ) -> "VariantPack":
    """Compile variants independently and merge their meshes into one model.

    Variants must have the same named structural layout. A canonical variant may
    contain optional mesh-geom slots that are absent in another variant; missing
    slots are disabled with ``mjGEOM_NONE``, ``geom_dataid=-1``, and zero contact
    bits. Non-mesh topology and parameters must not vary.
    """
    if slot_policy != "stable-name":
      raise ValueError("slot_policy must be 'stable-name'")
    if not specs:
      raise ValueError("at least one variant spec is required")

    reference_models = [spec.compile() for spec in specs]
    canonical_index = max(range(len(specs)), key=lambda i: (reference_models[i].ngeom, -i))
    canonical_spec = specs[canonical_index].copy()

    mesh_pool: dict[tuple[Any, ...], str] = {}
    for mesh in canonical_spec.meshes:
      signature = _mesh_signature(canonical_spec, mesh)
      mesh_pool[signature] = mesh.name

    mesh_names_by_variant: list[dict[str, str]] = []
    for spec in specs:
      mesh_names: dict[str, str] = {}
      for mesh in spec.meshes:
        signature = _mesh_signature(spec, mesh)
        pooled_name = mesh_pool.get(signature)
        if pooled_name is None:
          pooled_name = _add_pooled_mesh(canonical_spec, spec, mesh, mesh_pool)
          mesh_pool[signature] = pooled_name
        if mesh.name in mesh_names:
          raise ValueError(f"variant has duplicate mesh name {mesh.name!r}")
        mesh_names[mesh.name] = pooled_name
      mesh_names_by_variant.append(mesh_names)

    canonical = canonical_spec.compile()
    _validate_layout(reference_models, canonical)
    geom_maps = _validate_names_and_build_geom_maps(reference_models, canonical)
    body_maps = [np.arange(canonical.nbody, dtype=np.int32) for _ in specs]
    dof_maps = [np.arange(canonical.nv, dtype=np.int32) for _ in specs]
    mesh_id_maps = _build_mesh_id_maps(
      reference_models,
      canonical,
      specs,
      mesh_names_by_variant,
    )

    fields: dict[str, np.ndarray] = {}
    for name in _VARIANT_FIELDS:
      canonical_value = np.asarray(getattr(canonical, name))
      values = np.tile(canonical_value, (len(specs), *(1,) * canonical_value.ndim))
      if name in _GEOM_VARIANT_FIELDS:
        for variant, (reference, geom_map) in enumerate(zip(reference_models, geom_maps, strict=True)):
          values[variant][geom_map] = getattr(reference, name)
        values = _disable_missing_geom_slots(
          values,
          name,
          geom_maps,
          canonical,
        )
      elif name in _BODY_VARIANT_FIELDS:
        for variant, (reference, body_map) in enumerate(zip(reference_models, body_maps, strict=True)):
          values[variant][body_map] = getattr(reference, name)
      elif name in _DOF_VARIANT_FIELDS:
        for variant, (reference, dof_map) in enumerate(zip(reference_models, dof_maps, strict=True)):
          values[variant][dof_map] = getattr(reference, name)
      fields[name] = values

    dataids = fields["geom_dataid"] = np.full((len(specs), canonical.ngeom), -1, dtype=np.int32)
    for variant, (reference, geom_map) in enumerate(zip(reference_models, geom_maps, strict=True)):
      mesh_ids = mesh_id_maps[variant]
      for reference_geom, canonical_geom in enumerate(geom_map):
        dataid = int(reference.geom_dataid[reference_geom])
        dataids[variant, canonical_geom] = mesh_ids.get(dataid, -1)

    for values in fields.values():
      values.flags.writeable = False

    return cls(
      model=canonical,
      num_variants=len(specs),
      fields=MappingProxyType(fields),
    )


def _mesh_path(spec: mujoco.MjSpec, mesh: mujoco.MjsMesh) -> Path | None:
  if not mesh.file:
    return None
  path = Path(mesh.file)
  if path.is_absolute() or path.exists():
    return path.resolve()
  candidates = (spec.modelfiledir, spec.meshdir)
  for candidate in candidates:
    if candidate:
      resolved = (Path(candidate) / path).resolve()
      if resolved.exists():
        return resolved
  return path.resolve()


def _sequence(value: Iterable[Any]) -> tuple[Any, ...]:
  return tuple(np.asarray(value).tolist())


def _mesh_signature(spec: mujoco.MjSpec, mesh: mujoco.MjsMesh) -> tuple[Any, ...]:
  path = _mesh_path(spec, mesh)
  content = path.read_bytes() if path is not None and path.exists() else b""
  return (
    mesh.content_type,
    content,
    _sequence(mesh.refpos),
    _sequence(mesh.refquat),
    _sequence(mesh.scale),
    int(mesh.inertia),
    bool(mesh.smoothnormal),
    bool(mesh.needsdf),
    int(mesh.maxhullvert),
    int(mesh.octree_maxdepth),
    _sequence(mesh.uservert),
    _sequence(mesh.usernormal),
    _sequence(mesh.usertexcoord),
    _sequence(mesh.userface),
    _sequence(mesh.userfacenormal),
    _sequence(mesh.userfacetexcoord),
    mesh.material,
  )


def _add_pooled_mesh(
  target: mujoco.MjSpec,
  source_spec: mujoco.MjSpec,
  source: mujoco.MjsMesh,
  mesh_pool: Mapping[tuple[Any, ...], str],
) -> str:
  base_name = source.name or "mesh"
  index = len(mesh_pool)
  while True:
    name = f"mjbatch_mesh_{index}_{base_name}"
    if not any(mesh.name == name for mesh in target.meshes):
      break
    index += 1

  copied = target.add_mesh(name=name)
  path = _mesh_path(source_spec, source)
  copied.file = "" if path is None else str(path)
  copied.content_type = source.content_type
  copied.refpos = np.asarray(source.refpos)
  copied.refquat = np.asarray(source.refquat)
  copied.scale = np.asarray(source.scale)
  copied.inertia = source.inertia
  copied.smoothnormal = source.smoothnormal
  copied.needsdf = source.needsdf
  copied.maxhullvert = source.maxhullvert
  copied.octree_maxdepth = source.octree_maxdepth
  copied.uservert = list(source.uservert)
  copied.usernormal = list(source.usernormal)
  copied.usertexcoord = list(source.usertexcoord)
  copied.userface = list(source.userface)
  copied.userfacenormal = list(source.userfacenormal)
  copied.userfacetexcoord = list(source.userfacetexcoord)
  copied.material = source.material
  return name


def _layout(model: mujoco.MjModel) -> tuple[tuple[str, int], ...]:
  return tuple((name, int(getattr(model, name))) for name in _LAYOUT_SCALARS if hasattr(model, name))


def _validate_layout(references: Sequence[mujoco.MjModel], canonical: mujoco.MjModel) -> None:
  canonical_layout = dict(_layout(canonical))
  for variant, reference in enumerate(references):
    for name, value in _layout(reference):
      if value != canonical_layout[name]:
        raise ValueError(
          f"variant {variant} changes layout field {name}: {value} != {canonical_layout[name]}"
        )
    if reference.ngeom > canonical.ngeom:
      raise ValueError(f"variant {variant} has more geoms than the canonical model")


def _names(model: mujoco.MjModel, kind: str, count: int) -> list[str]:
  accessor = getattr(model, kind)
  return [str(accessor(i).name) for i in range(count)]


def _validate_names_and_build_geom_maps(
  references: Sequence[mujoco.MjModel], canonical: mujoco.MjModel
) -> list[np.ndarray]:
  canonical_geoms = _names(canonical, "geom", canonical.ngeom)
  if len(set(canonical_geoms)) != len(canonical_geoms) or "" in canonical_geoms:
    raise ValueError("canonical geoms must have unique, non-empty names")

  canonical_by_kind = {
    kind: _names(canonical, kind, getattr(canonical, count))
    for kind, count in (
      ("body", "nbody"),
      ("joint", "njnt"),
      ("site", "nsite"),
      ("cam", "ncam"),
      ("light", "nlight"),
      ("actuator", "nu"),
      ("sensor", "nsensor"),
      ("tendon", "ntendon"),
    )
  }
  canonical_geom_ids = {name: i for i, name in enumerate(canonical_geoms)}

  maps: list[np.ndarray] = []
  for variant, reference in enumerate(references):
    for kind, expected in canonical_by_kind.items():
      count = {
        "body": "nbody",
        "joint": "njnt",
        "site": "nsite",
        "cam": "ncam",
        "light": "nlight",
        "actuator": "nu",
        "sensor": "nsensor",
        "tendon": "ntendon",
      }[kind]
      actual = _names(reference, kind, getattr(reference, count))
      if actual != expected:
        raise ValueError(f"variant {variant} changes {kind} names or order")
    names = _names(reference, "geom", reference.ngeom)
    if len(set(names)) != len(names) or "" in names:
      raise ValueError(f"variant {variant} geoms must have unique, non-empty names")
    unknown = set(names) - set(canonical_geoms)
    if unknown:
      raise ValueError(f"variant {variant} has geoms absent from the canonical layout: {sorted(unknown)}")
    maps.append(np.asarray([canonical_geom_ids[name] for name in names], dtype=np.int32))
  return maps


def _build_mesh_id_maps(
  references: Sequence[mujoco.MjModel],
  canonical: mujoco.MjModel,
  specs: Sequence[mujoco.MjSpec],
  mesh_names_by_variant: Sequence[Mapping[str, str]],
) -> list[dict[int, int]]:
  maps: list[dict[int, int]] = []
  for reference, spec, mesh_names in zip(references, specs, mesh_names_by_variant, strict=True):
    mesh_ids: dict[int, int] = {}
    for mesh in spec.meshes:
      reference_id = reference.mesh(mesh.name).id
      pooled_name = mesh_names[mesh.name]
      mesh_ids[reference_id] = canonical.mesh(pooled_name).id
    maps.append(mesh_ids)
  return maps


def _disable_missing_geom_slots(
  values: np.ndarray,
  name: str,
  geom_maps: Sequence[np.ndarray],
  canonical: mujoco.MjModel,
) -> np.ndarray:
  identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=values.dtype)
  for variant, geom_map in enumerate(geom_maps):
    present = np.zeros(canonical.ngeom, dtype=bool)
    present[geom_map] = True
    missing = np.flatnonzero(~present)
    if missing.size == 0:
      continue
    if name == "geom_type":
      values[variant, missing] = int(mujoco.mjtGeom.mjGEOM_NONE)
    elif name in ("geom_contype", "geom_conaffinity", "geom_matid"):
      values[variant, missing] = -1 if name == "geom_matid" else 0
    elif name == "geom_rgba":
      values[variant, missing] = (0.0, 0.0, 0.0, 0.0)
    elif name == "geom_quat":
      values[variant, missing] = identity
    else:
      values[variant, missing] = 0
  return values
