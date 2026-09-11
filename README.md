# mjbatch

[![Build](https://img.shields.io/github/actions/workflow/status/kevinzakka/mjbatch/ci.yml?branch=main)](https://github.com/kevinzakka/mjbatch/actions)
[![PyPI version](https://img.shields.io/pypi/v/mjbatch)](https://pypi.org/project/mjbatch/)

`mjbatch` is a Python library for running thousands of MuJoCo simulations in parallel on CPU.

Features include:

* C++ thread pool execution, with the GIL released;
* Live array access to simulation state and controls across the batch, with `bind` for MjData fields;
* Per-simulation model parameters, with `expand` for MjModel fields and `set_const` to recompute derived constants.
* Same-layout compiler-coherent mesh variants, with `VariantPack.from_specs()` and `Batch.from_variant_pack()`.
* Batched queries beyond stepping: site Jacobians with `jac_site` and heightfield sampling with `sample_hfield`.

For example:

```python
import mujoco, numpy as np
from mjbatch import Batch

model = mujoco.MjModel.from_xml_path("scene.xml")
batch = Batch(model, num_sims=4096)  # threads default to every logical CPU
qpos, ctrl = batch.bind("qpos"), batch.bind("ctrl")
batch.expand("geom_friction")[:, :, 0] = np.random.uniform(0.4, 1.2, (4096, 1))
for _ in range(1000):
  ctrl[:] = policy(qpos)             # your controller, all 4096 at once
  batch.step()                       # step them in parallel; qpos updates in place
```

## Per-simulation model fields

`expand(field)` returns a live `(num_sims, ...)` view of a non-asset `MjModel` field.
Rows are applied to MuJoCo before that simulation's next call. Use `set_const(ids)`
after changing inputs such as mass or inertia; it runs `mj_setConst` per selected
simulation and expands the derived fields that MuJoCo changed.

```python
mass = batch.expand("body_mass")
mass[[1, 4, 7], body_id] *= 1.2
batch.set_const(np.array([1, 4, 7]))
```

Asset arrays (`mesh_*`, `hfield_*`, `tex_*`, and related large constant data) are
shared and immutable. A model can nevertheless pool multiple meshes in one canonical
layout and select one per simulation through `geom_dataid`:

```python
mesh_id = canonical.geom("mesh").id
dataid = batch.expand("geom_dataid")
dataid[tool_envs, mesh_id] = pooled_mesh_ids[tool_envs]
```

Mesh geometry also affects compiler-derived fields such as `geom_size`,
`geom_rbound`, `geom_aabb`, `geom_pos`, `geom_quat`, `body_inertia`,
`body_invweight0`, `body_ipos`, and `body_iquat`. Scatter those values from
independently compiled reference models before calling `set_const`. mjbatch does
not require the manual field scatter below to be handwritten for common mesh
variants, but all directly pooled variants must share the same model layout.

For same-layout mesh variants, `VariantPack.from_specs()` performs that construction:
it compiles every source spec independently, pools and deduplicates meshes, aligns
named geom slots, disables optional slots missing from a variant, and scatters the
compiler-derived geometry and inertia fields. `Batch.from_variant_pack()` applies a
fixed initialization-time assignment and performs the initial recompute.

```python
from mjbatch import Batch, VariantPack

pack = VariantPack.from_specs([tool_spec_0, tool_spec_1, tool_spec_2])
batch = Batch.from_variant_pack(pack, num_sims, np.arange(num_sims) % 3)
```

Variants must use the same named structural layout and differ only in mesh assets or
the presence of optional mesh-geom slots. Different topologies require the separate
topology-group API rather than silent padding.

`model_field_specs()` describes every field accepted by `expand`: its template shape,
native dtype, whether it is writable or pooled asset data, and which derived constants
must be refreshed after it changes. `model_update(ids)` is a transaction over those
fields. It compares only the selected rows and, on exit, invokes the strongest required
recompute once. Writes made to unselected rows are restored. Writing only fields with
`RecomputeLevel.NONE` performs no recompute; higher levels currently use one
conservative full `mj_setConst` pass.

```python
from mjbatch import RecomputeLevel

specs = batch.model_field_specs()
assert specs["body_mass"].recompute == RecomputeLevel.SET_CONST
with batch.model_update(reset_envs):
  batch.expand("body_mass")[reset_envs, body_id] *= 1.1
  batch.expand("geom_friction")[reset_envs, :, 0] = friction_samples
```

## Examples

We showcase a range of applications built using `mjbatch`: RL, MPC, SysID, and hardware
co-design. Each example is a self-contained, performant implementation. For instance, the Go1
RL controller learns to walk in under a minute on a five-year-old M1 laptop.

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/cartpole_swingup.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/cartpole_swingup.gif" alt="cart-pole swing-up"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/cartpole_mpc.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/cartpole_mpc.gif" alt="cart-pole MPC"></a>
    </td>
  </tr>
  <tr>
    <td align="center">A two-pole cart swung upright with <a href="https://ieeexplore.ieee.org/document/6386025">iLQR</a></td>
    <td align="center">A cart-pole swing-up controller using <a href="https://arxiv.org/abs/2212.00541">predictive sampling</a></td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/g1_flip.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/g1_flip.gif" alt="G1 backflip"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/go1_joystick.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/go1_joystick.gif" alt="Go1 joystick"></a>
    </td>
  </tr>
  <tr>
    <td align="center">A G1 humanoid tracking a reference backflip with receding-horizon iLQR</td>
    <td align="center">A Go1 quadruped joystick controller trained with PPO</td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/arm_throw.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/arm_throw.gif" alt="throwing arm co-design"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/rizon_inertia.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/rizon_inertia.gif" alt="Rizon inertia identification"></a>
    </td>
  </tr>
  <tr>
    <td align="center">CEM jointly optimizes a robot arm's proportions, gears, and controls</td>
    <td align="center">Damped Gauss–Newton fits a Rizon arm's inertial parameters to synthetic motion data</td>
  </tr>
</table>

Run with `uv run examples/<file>.py`; some need `uv sync --group examples`. The ones that open
a window need a display; `--headless` runs the solver without one.

## License

Apache-2.0.
