# SPDX-License-Identifier: Apache-2.0

import copy
import subprocess
import sys
import threading

import mujoco
import numpy as np
import pytest

from mjbatch import Batch

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="cam" pos="1 1 1"/>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
      <site name="base"/>
      <body name="pole" pos="0 0 .05" gravcomp="0">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".1"/>
        <site name="tip" pos="0 0 .5"/>
      </body>
    </body>
    <body name="puck" pos="0 1 .5">
      <joint type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>
    </body>
    <body name="mocap" mocap="true" pos="1 0 1">
      <geom type="sphere" size=".02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <tendon><spatial name="t"><site site="base"/><site site="tip"/></spatial></tendon>
  <equality><weld body1="mocap" body2="cart" active="false"/></equality>
  <actuator><motor joint="slide" gear="10"/><position joint="hinge" kp="1"/></actuator>
  <sensor><jointpos joint="hinge"/><framepos objtype="site" objname="tip"/></sensor>
  <keyframe><key qpos="0.5 0.2"/></keyframe>
</mujoco>
"""
N = 8

# Activation dynamics, a mocap weld, a keyframe, and sensors that set mjData's
# lazy-evaluation flags (accelerometer, subtreelinvel).
LOCKSTEP_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
      <body name="pole" pos="0 0 .05">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".1"/>
        <site name="tip" pos="0 0 .5"/>
      </body>
    </body>
    <body name="ball" pos="0 1 .5">
      <freejoint/>
      <geom type="sphere" size=".05" mass=".2"/>
    </body>
    <body name="mocap" mocap="true" pos="0 1 .5">
      <geom type="sphere" size=".02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality><weld body1="mocap" body2="ball"/></equality>
  <actuator>
    <motor joint="slide" gear="5"/>
    <general joint="hinge" dyntype="filter" dynprm="0.02" gainprm="5"
             biastype="affine" biasprm="0 -5 0"/>
  </actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <accelerometer site="tip"/>
    <subtreelinvel body="cart"/>
  </sensor>
  <keyframe>
    <key qpos="0.3 0.5 0 1 .5 1 0 0 0" ctrl="0.1 0.2" act="0.2" mpos="0 1 .5"/>
  </keyframe>
</mujoco>
"""
COMPARED = ("qpos", "qvel", "act", "time", "sensordata", "site_xpos")


@pytest.fixture
def model():
  return mujoco.MjModel.from_xml_string(XML)


def lockstep(nstep, num_threads, heavy=()):
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  models = [model] * N
  batch = Batch(model, N, num_threads=num_threads)
  if heavy:
    batch.expand("body_mass")[list(heavy), 1] *= 3.0
    batch.set_const(np.array(heavy))
    for i in heavy:
      models[i] = copy.copy(model)
      models[i].body_mass[1] *= 3.0
      mujoco.mj_setConst(models[i], mujoco.MjData(models[i]))
  datas = [mujoco.MjData(m) for m in models]
  pending = [[] for _ in range(N)]
  bound = {f: batch.bind(f) for f in COMPARED}
  rng = np.random.default_rng(0)

  def write(field, ids, index, value):
    batch.bind(field)[(ids, *index)] = value
    for j, i in enumerate(ids):
      pending[i].append((field, index, value[j]))

  def apply(i):
    for field, index, value in pending[i]:
      getattr(datas[i], field)[index] = value
    pending[i].clear()

  every = list(range(N))
  for call in range(50):
    write("ctrl", every, (slice(None),), rng.uniform(-1, 1, (N, model.nu)))
    if call == 10:
      write("qpos", [1, 4, 6], (slice(0, 2),), rng.uniform(-0.3, 0.3, (3, 2)))
    if call == 15:
      write("mocap_pos", every, (0,), rng.uniform(-0.2, 0.2, (N, 3)) + [0, 1, 0.5])
    if call == 20:
      write("xfrc_applied", [2, 3], (3, slice(0, 3)), rng.uniform(-1, 1, (2, 3)))
    if call == 30:
      write("eq_active", [0, 2, 4, 6], (0,), np.zeros(4, np.uint8))
    ids = [0, 2, 3, 7] if call % 4 == 3 else every
    if call == 25:
      ids = [1, 5]
      batch.reset(np.array(ids), keyframe=0)
      for i in ids:
        mujoco.mj_resetDataKeyframe(models[i], datas[i], 0)
        apply(i)
        mujoco.mj_forward(models[i], datas[i])
    else:
      batch.step(None if len(ids) == N else np.array(ids), nstep=nstep)
      for i in ids:
        apply(i)
        for _ in range(nstep):
          mujoco.mj_step(models[i], datas[i])
    for field, arr in bound.items():
      np.testing.assert_array_equal(arr, [getattr(d, field) for d in datas], field)


@pytest.mark.parametrize("nstep", [1, 3])
@pytest.mark.parametrize("num_threads", [1, 3, 7])
def test_lockstep_with_mj_step(nstep, num_threads):
  lockstep(nstep, num_threads)


@pytest.mark.parametrize("nstep", [1, 3])
def test_lockstep_with_expanded_mass(nstep):
  lockstep(nstep, num_threads=4, heavy=(1, 2, 5))


def test_sleep_is_rejected():
  xml = LOCKSTEP_XML.replace("<option", '<option><flag sleep="enable"/></option><option')
  with pytest.raises(ValueError, match="sleep"):
    Batch(mujoco.MjModel.from_xml_string(xml), N)


def test_per_sim_gravity_matches_separate_models():
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batch = Batch(model, N, num_threads=3)
  gravity = batch.expand("gravity")
  gravity[:, 2] = np.linspace(-9.81, -1.0, N)
  batch.bind("ctrl")[:, 0] = 0.5
  qpos, qvel = batch.bind("qpos"), batch.bind("qvel")
  batch.step(nstep=50)
  for i in range(N):
    m = copy.copy(model)
    m.opt.gravity[2] = gravity[i, 2]
    d = mujoco.MjData(m)
    d.ctrl[0] = 0.5
    for _ in range(50):
      mujoco.mj_step(m, d)
    np.testing.assert_array_equal(qpos[i], d.qpos)
    np.testing.assert_array_equal(qvel[i], d.qvel)


def test_per_sim_timestep(model):
  batch = Batch(model, N, num_threads=2)
  timestep = batch.expand("timestep")
  timestep[:] = 0.001 * (1 + np.arange(N))
  assert timestep.shape == (N,)
  time = batch.bind("time")
  batch.step(nstep=4)
  np.testing.assert_allclose(time, 4 * timestep)


def test_per_sim_integrator(model):
  batch = Batch(model, N)
  integrator = batch.expand("integrator")
  assert integrator.dtype == np.int32
  np.testing.assert_array_equal(integrator, model.opt.integrator)
  integrator[::2] = mujoco.mjtIntegrator.mjINT_RK4
  batch.bind("ctrl")[:, 0] = 1.0
  qvel = batch.bind("qvel")
  batch.step(nstep=20)
  np.testing.assert_array_equal(qvel[1], qvel[3])
  assert not np.array_equal(qvel[0], qvel[1])
  ref = copy.copy(model)
  ref.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
  d = mujoco.MjData(ref)
  d.ctrl[0] = 1.0
  for _ in range(20):
    mujoco.mj_step(ref, d)
  np.testing.assert_array_equal(qvel[0], d.qvel)


def test_expanded_option_seeds_from_the_template(model):
  batch, ref = Batch(model, N), Batch(model, N)
  gravity = batch.expand("gravity")
  np.testing.assert_array_equal(gravity, np.tile(model.opt.gravity, (N, 1)))
  np.testing.assert_array_equal(batch.expand("o_solref"), np.tile(model.opt.o_solref, (N, 1)))
  for b in (batch, ref):
    b.bind("ctrl")[:, 0] = 1.0
    b.step(nstep=20)
  np.testing.assert_array_equal(batch.bind("state"), ref.bind("state"))


def test_option_is_untouched_by_set_const(model):
  # mj_setConst never writes opt, so no opt field can be flagged as one of its
  # outputs and expanded behind the caller's back.
  ref = mujoco.MjModel.from_xml_string(XML)
  before = {f: np.array(getattr(ref.opt, f), copy=True) for f in dir(ref.opt) if f[0] != "_"}
  mujoco.mj_setConst(ref, mujoco.MjData(ref))
  for f, v in before.items():
    np.testing.assert_array_equal(getattr(ref.opt, f), v, f)
  batch = Batch(model, N)
  timestep = batch.expand("timestep")
  timestep[:] = 0.001 * (1 + np.arange(N))
  batch.expand("body_mass")[:, model.body("pole").id] = 0.5
  batch.set_const()
  np.testing.assert_array_equal(timestep, 0.001 * (1 + np.arange(N)))
  time = batch.bind("time")
  batch.step()
  np.testing.assert_allclose(time, timestep)


def test_per_sim_sleep_is_rejected(model):
  batch = Batch(model, N, num_threads=2)
  enableflags = batch.expand("enableflags")
  enableflags[3] |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  batch.step(np.array([0, 1]))  # sim 3 is not running
  for call in (batch.step, batch.forward, batch.set_const):
    with pytest.raises(ValueError, match="sim 3: sleep"):
      call()
  enableflags[3] = model.opt.enableflags
  batch.step()


def test_step_history_matches_a_loop(model):
  batch, ref = Batch(model, N, num_threads=3), Batch(model, N, num_threads=3)
  for b in (batch, ref):
    b.bind("ctrl")[:, 0] = np.linspace(-1, 1, N)
  history = np.empty((N, 7, batch.nstate))
  batch.step(nstep=7, history=history)
  np.testing.assert_array_equal(history[:, -1], batch.bind("state"))
  for k in range(7):
    ref.step()
    np.testing.assert_array_equal(history[:, k], ref.bind("state"))


def test_step_history_with_ids(model):
  batch = Batch(model, N)
  batch.bind("ctrl")[:, 0] = np.linspace(-1, 1, N)
  ids = np.array([1, 4, 6])
  history = np.empty((len(ids), 5, batch.nstate))
  batch.step(ids, nstep=5, history=history)
  state, time = batch.bind("state"), batch.bind("time")
  np.testing.assert_array_equal(history[:, -1], state[ids])
  np.testing.assert_allclose(history[:, :, 0], np.tile(np.arange(1, 6) * 0.002, (3, 1)))
  assert not np.any(time[[0, 2, 3, 5, 7]])


def test_step_history_validation(model):
  batch = Batch(model, N)
  nstate = batch.nstate
  for bad in (
    np.empty((N, 3)),
    np.empty((N, 2, nstate)),
    np.empty((N - 1, 3, nstate)),
    np.empty((N, 3, nstate), np.float32),
    np.empty((N, 3, nstate + 1))[:, :, :-1],
  ):
    with pytest.raises(ValueError, match="history"):
      batch.step(nstep=3, history=bad)
  with pytest.raises(ValueError, match="history"):
    batch.step(np.array([0, 1]), nstep=3, history=np.empty((N, 3, nstate)))
  assert not np.any(batch.bind("time"))


def test_step_history_allows_an_empty_selection(model):
  # A size-0 history has nothing written to it, so its strides are irrelevant
  # (numpy sets them to zero); it must not be rejected as non-contiguous.
  batch = Batch(model, N)
  batch.step(np.zeros(N, dtype=bool), nstep=3, history=np.empty((0, 3, batch.nstate)))
  batch.step(np.array([], dtype=np.int64), nstep=2, history=np.empty((0, 2, batch.nstate)))
  assert not np.any(batch.bind("time"))


def test_step_history_error_names_the_sim():
  batch = Batch(mujoco.MjModel.from_xml_string(LOCKSTEP_XML), N, num_threads=2)
  batch.expand("eq_type")[2] = 99
  history = np.zeros((N, 3, batch.nstate))
  with pytest.raises(RuntimeError, match="sim 2"):
    batch.step(nstep=3, history=history)
  assert np.any(history[0])  # the sims that ran still wrote their rows


def test_memory_does_not_scale_with_num_sims():
  # A fresh process, so ru_maxrss growth is this batch's. One mjData per sim
  # grows it by 712 MB here; 4096 state vectors are a few MB.
  code = """
import resource, sys, mujoco
from mjbatch import Batch
xml = '<mujoco><size memory="256K"/><worldbody><body><joint/><geom size=".02"/></body></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
scale = 1 if sys.platform == "darwin" else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
batch = Batch(model, 4096)
batch.step()
print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) * scale)
"""
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
  assert out.returncode == 0, out.stderr
  assert int(out.stdout) < 256 * 2**20


@pytest.mark.parametrize("num_threads", [1, 3])
def test_step_error_keeps_the_failing_sim(num_threads):
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batch = Batch(model, N, num_threads=num_threads)
  eq_type = batch.expand("eq_type")
  eq_type[2] = 99
  ctrl, qpos = batch.bind("ctrl"), batch.bind("qpos")
  ctrl[:, 0] = np.linspace(-1, 1, N)
  with pytest.raises(RuntimeError, match="sim 2"):
    batch.step(nstep=3)
  # The others ran; sim 2 kept its state and its pending write, and the worker
  # it failed on serves later sims correctly.
  eq_type[2] = model.eq_type
  batch.step()
  one, four = reference(model, ctrl, 1), reference(model, ctrl, 4)
  for i in range(N):
    np.testing.assert_array_equal(qpos[i], (one if i == 2 else four)[i].qpos)


def reference(model, ctrl, nstep):
  datas = [mujoco.MjData(model) for _ in range(N)]
  for i, d in enumerate(datas):
    d.ctrl[:] = ctrl[i]
    for _ in range(nstep):
      mujoco.mj_step(model, d)
  return datas


@pytest.mark.parametrize("num_threads", [1, 4])
def test_step_matches_reference(model, num_threads):
  batch = Batch(model, N, num_threads=num_threads)
  ctrl = batch.bind("ctrl")
  ctrl[:, 0] = np.linspace(-1, 1, N)
  qpos, qvel, sensordata = (batch.bind(f) for f in ("qpos", "qvel", "sensordata"))
  for _ in range(25):
    batch.step()
  batch.step(nstep=25)
  for i, d in enumerate(reference(model, ctrl, 50)):
    np.testing.assert_array_equal(qpos[i], d.qpos)
    np.testing.assert_array_equal(qvel[i], d.qvel)
    np.testing.assert_array_equal(sensordata[i], d.sensordata)


def test_forward_after_step(model):
  """With forward=True the derived fields are current after a step; without, one
  substep behind, as with mj_step."""
  for forward in (False, True):
    batch = Batch(model, N, forward=forward)
    ctrl, xpos, sensordata = (batch.bind(f) for f in ("ctrl", "xpos", "sensordata"))
    ctrl[:, 0] = np.linspace(-1, 1, N)
    batch.step(nstep=10)
    behind = xpos.copy(), sensordata.copy()
    batch.forward()
    same = np.array_equal(behind[0], xpos) and np.array_equal(behind[1], sensordata)
    assert same == forward


def test_shapes_and_dtypes_match_mjdata(model):
  batch = Batch(model, N)
  d = mujoco.MjData(model)
  for name in dir(d):
    if name.startswith("_"):
      continue
    val = getattr(d, name)
    if not isinstance(val, np.ndarray) or name in ("contact", "efc_type"):
      continue
    try:
      arr = batch.bind(name)
    except ValueError:
      continue
    assert arr.shape == (N, *val.shape), name
    assert arr.dtype == val.dtype, name
  assert batch.bind("time").shape == (N,)
  batch = Batch(model, N)
  assert batch.bind("qpos", np.float32).dtype == np.float32
  with pytest.raises(ValueError):
    batch.bind("qpos")  # already bound as float32
  with pytest.raises(ValueError):
    batch.bind("eq_active", np.float32)
  with pytest.raises(ValueError):
    batch.bind("nope")
  with pytest.raises(ValueError, match=r'use expand\("geom_friction"\)'):
    batch.bind("geom_friction")
  with pytest.raises(ValueError, match=r'use bind\("qpos"\)'):
    batch.expand("qpos")
  with pytest.raises(ValueError):
    batch.expand("mesh_vert")
  with pytest.raises(ValueError):
    Batch(model, 0)


def test_float32_rows_only_written_when_changed(model):
  batch = Batch(model, N, num_threads=2)
  ctrl = batch.bind("ctrl", np.float32)
  qpos = batch.bind("qpos", np.float32)
  ctrl[:, 0] = 0.3
  for _ in range(200):
    batch.step()
  # Physics ran in mjtNum precision: untouched float32 rows never round-trip.
  ref_ctrl = np.zeros((N, model.nu))
  ref_ctrl[:, 0] = np.float32(0.3)
  d = reference(model, ref_ctrl, 200)[0]
  np.testing.assert_array_equal(qpos[0], d.qpos.astype(np.float32))
  # A written row takes effect on the next call; a derived field is read-only.
  site_xpos = batch.bind("site_xpos")
  before = site_xpos[1].copy()
  qpos[1, 1] = 0.7
  site_xpos[0] = 42.0
  batch.forward()
  assert not np.array_equal(site_xpos[1], before)
  assert not np.any(site_xpos[0] == 42.0)


def test_expand_and_set_const_are_complete(model):
  pole = model.body("pole").id
  results = []
  for num_threads in (1, 4):
    batch = Batch(model, N, num_threads=num_threads)
    mass = batch.expand("body_mass")
    mass[:, pole] = np.linspace(0.1, 1.0, N)
    batch.set_const()
    # Everything mj_setConst derived is now per sim, in native dtype.
    subtree = batch.expand("body_subtreemass")
    np.testing.assert_allclose(subtree[:, pole], mass[:, pole])
    assert batch.expand("dof_invweight0").shape == (N, model.nv)
    batch.bind("ctrl")[:, 0] = 1.0
    qvel = batch.bind("qvel")
    batch.step(nstep=20)
    results.append(qvel.copy())
  np.testing.assert_array_equal(results[0], results[1])
  ref = mujoco.MjModel.from_xml_string(XML)
  ref.body_mass[pole] = 1.0
  mujoco.mj_setConst(ref, mujoco.MjData(ref))
  d = mujoco.MjData(ref)
  d.ctrl[0] = 1.0
  for _ in range(20):
    mujoco.mj_step(ref, d)
  np.testing.assert_array_equal(results[0][-1], d.qvel)


def test_gravcomp_flag_is_per_sim(model):
  pole = model.body("pole").id
  results = []
  for num_threads in (1, 4):
    batch = Batch(model, N, num_threads=num_threads)
    gravcomp = batch.expand("body_gravcomp")
    gravcomp[::2, pole] = 1.0
    batch.set_const()
    batch.bind("qpos")[:, 1] = 0.3
    qvel = batch.bind("qvel")
    batch.step(nstep=5)
    results.append(qvel.copy())
  np.testing.assert_array_equal(results[0], results[1])
  assert np.all(results[0][::2, 1] != results[0][1::2, 1])


def test_reset_keyframe_and_expanded_qpos0(model):
  batch = Batch(model, N)
  qpos0 = batch.expand("qpos0")
  qpos0[:, 1] = np.arange(N) * 0.1
  qpos = batch.bind("qpos")
  batch.bind("ctrl")[:, 0] = 1.0
  batch.step()
  qpos[5, 0] = 0.25  # A write survives a reset of that sim and lands after it.
  xpos = batch.bind("xpos")
  batch.reset(np.array([2, 5]))
  np.testing.assert_array_equal(qpos[2], qpos0[2])
  assert qpos[5, 0] == 0.25
  np.testing.assert_array_equal(qpos[5, 1:], qpos0[5, 1:])
  assert qpos[0, 0] != 0.0
  assert xpos[2, 1, 2] != 0.0  # reset forwards
  mask = np.zeros(N, dtype=bool)
  mask[0] = True
  batch.reset(mask)
  assert qpos[0, 0] == 0.0
  batch.reset(keyframe=0)
  np.testing.assert_array_equal(qpos, np.tile(model.key_qpos[0], (N, 1)))
  for bad in (1, -2):
    with pytest.raises(ValueError):
      batch.reset(keyframe=bad)


def test_ids_validation_and_time(model):
  batch = Batch(model, N, num_threads=3)
  time = batch.bind("time")
  batch.step(np.array([1, 3]))
  np.testing.assert_array_equal(time, [0, 0.002, 0, 0.002, 0, 0, 0, 0])
  batch.step(np.array([1, 3], dtype=np.int32))
  np.testing.assert_array_equal(time, [0, 0.004, 0, 0.004, 0, 0, 0, 0])
  for bad in ([N], [1, 1], [3, 1], [1.0, 2.0]):
    with pytest.raises(ValueError):
      batch.step(np.array(bad))


def test_warning_counters(model):
  batch = Batch(model, N)
  warning = batch.bind("warning")
  batch.bind("qpos")[2] = np.nan
  batch.step()
  assert warning.shape == (N, mujoco.mjtWarning.mjNWARNING.value, 2)
  assert warning[2, mujoco.mjtWarning.mjWARN_BADQPOS, 1] == 1
  assert warning[:, :, 1].sum() == 1


@pytest.mark.parametrize("num_threads", [1, 2])
def test_mujoco_error_becomes_exception(model, num_threads):
  batch = Batch(model, N, num_threads=num_threads)
  mass = batch.expand("body_mass")
  mass[3, model.body("puck").id] = 0.0
  mass[5, model.body("pole").id] = 2.0
  with pytest.raises(RuntimeError, match="sim 3"):
    batch.set_const()
  # The other sims still ran; a later call works after fixing the input.
  assert batch.expand("body_subtreemass")[5, model.body("pole").id] == 2.0
  mass[3, model.body("puck").id] = 1.0
  batch.set_const()


def test_set_const_subset_keeps_every_sim_consistent(model):
  batch = Batch(model, N)
  pole = model.body("pole").id
  batch.expand("body_mass")[:, pole] = 5.0
  batch.set_const(np.array([0, 1]))
  np.testing.assert_array_equal(batch.expand("body_subtreemass")[:, pole], 5.0)


def test_concurrent_calls_are_serialized(model):
  batch = Batch(model, N, num_threads=4)
  time = batch.bind("time")

  other = Batch(model, N, num_threads=4)
  other_time = other.bind("time")

  def work(b, nstep):
    for _ in range(50):
      b.step(nstep=nstep)

  threads = [threading.Thread(target=work, args=a) for a in ((batch, 1), (batch, 5), (other, 2))]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  np.testing.assert_allclose(time, 300 * 0.002)
  np.testing.assert_allclose(other_time, 100 * 0.002)


def test_named_views(model):
  batch = Batch(model, N)
  hinge, tip = batch.joint("hinge"), batch.site("tip")
  assert hinge.qpos.shape == (N, 1) and hinge.qvel.shape == (N, 1)
  batch.bind("ctrl")[:, 0] = 1.0
  batch.step(nstep=10)
  np.testing.assert_array_equal(hinge.qpos[:, 0], batch.bind("qpos")[:, 1])
  np.testing.assert_array_equal(tip.xpos, batch.bind("site_xpos")[:, model.site("tip").id])
  assert batch.body("pole").xquat.shape == (N, 4)
  assert batch.body("pole").xpos.shape == (N, 3)


def test_state_rows_copy_restore_and_compose(model):
  batch = Batch(model, N)
  state, qpos, xpos = batch.bind("state"), batch.bind("qpos"), batch.bind("xpos")
  assert batch.nstate == mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
  assert state.shape == (N, batch.nstate) and state.dtype == np.float64
  batch.bind("ctrl")[:] = np.arange(N)[:, None] * 0.1
  batch.step(nstep=5)
  # Copying a row copies the physics: sim 1 becomes sim 0 in every field.
  assert not np.array_equal(qpos[0], qpos[1])
  state[1] = state[0]
  batch.forward(np.array([0, 1]))  # sim 0's derived fields lag by a substep
  np.testing.assert_array_equal(qpos[1], qpos[0])
  np.testing.assert_array_equal(xpos[1], xpos[0])
  # A state write and a field write compose, the field write winning on its overlap,
  # and the step matches mj_setState + mj_step on the same data.
  data = mujoco.MjData(model)
  saved = state[3].copy()
  batch.step(nstep=3)
  state[3] = saved
  qpos[3, 0] = 0.7
  mujoco.mj_setState(model, data, saved, mujoco.mjtState.mjSTATE_INTEGRATION)
  data.qpos[0], data.ctrl[:] = 0.7, batch.bind("ctrl")[3]
  batch.step(np.array([3]))
  mujoco.mj_step(model, data)
  np.testing.assert_array_equal(qpos[3], data.qpos)
  np.testing.assert_array_equal(state[3, 1 : 1 + model.nq], data.qpos)
  # Round trip: restoring the rows replays the same steps bit for bit.
  before = state.copy()
  batch.step(nstep=2)
  once = state.copy()
  state[:] = before
  batch.step(nstep=2)
  np.testing.assert_array_equal(state, once)
  # reset discards a pending state write; float32 is refused; the view is the same.
  state[2] = once[5]
  batch.reset(np.array([2]))
  np.testing.assert_array_equal(qpos[2], 0.0)
  with pytest.raises(ValueError):
    batch.bind("state", np.float32)
  assert np.shares_memory(batch.bind("state"), state)


HFIELD_XML = """
<mujoco>
  <asset>
    <hfield name="hf" nrow="9" ncol="9" size="1 1 0.5 0.1"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="hf"/>
    <body name="cart" pos="0 0 0.2">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _ref_hfield(model, qpos_row, geom, body, offsets):
  d = mujoco.MjData(model)
  d.qpos[:] = qpos_row
  mujoco.mj_forward(model, d)
  hfield = model.geom_dataid[geom]
  nrow, ncol = model.hfield_nrow[hfield], model.hfield_ncol[hfield]
  size = model.hfield_size[hfield]
  data = model.hfield_data[hfield * nrow * ncol : (hfield + 1) * nrow * ncol].reshape(nrow, ncol)
  gpos, gmat = d.geom_xpos[geom], d.geom_xmat[geom].reshape(3, 3)
  bpos = d.xpos[body]
  out = np.zeros(len(offsets))
  for k, (ox, oy) in enumerate(offsets):
    w = np.array([bpos[0] + ox, bpos[1] + oy, gpos[2]])
    lp = gmat.T @ (w - gpos)
    fx = np.clip((lp[0] / size[0] + 1.0) * 0.5 * (ncol - 1), 0, ncol - 1.001)
    fy = np.clip((lp[1] / size[1] + 1.0) * 0.5 * (nrow - 1), 0, nrow - 1.001)
    ix, iy = int(fx), int(fy)
    sx, sy = fx - ix, fy - iy
    h = (
      (1 - sx) * (1 - sy) * data[iy, ix]
      + sx * (1 - sy) * data[iy, min(ix + 1, ncol - 1)]
      + (1 - sx) * sy * data[min(iy + 1, nrow - 1), ix]
      + sx * sy * data[min(iy + 1, nrow - 1), min(ix + 1, ncol - 1)]
    )
    out[k] = h * size[2]
  return out


def test_jac_site_matches_serial(model):
  batch = Batch(model, N, num_threads=3)
  rng = np.random.default_rng(0)
  qpos = batch.bind("qpos")
  qpos[:] = model.qpos0 + rng.uniform(-0.2, 0.2, (N, model.nq))
  expected_qpos = qpos.copy()
  site = model.site("tip").id
  jacp, jacr = batch.jac_site("tip")
  assert jacp.shape == (N, 3, model.nv) and jacr.shape == (N, 3, model.nv)
  for i in range(N):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos[i]
    mujoco.mj_kinematics(model, d)
    mujoco.mj_comPos(model, d)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, d, jp, jr, site)
    np.testing.assert_array_equal(jacp[i], jp)
    np.testing.assert_array_equal(jacr[i], jr)
  # A subset fills one row per selected sim, in selection order.
  ids = np.array([1, 3, 4])
  sub_p, sub_r = batch.jac_site("tip", ids=ids)
  np.testing.assert_array_equal(sub_p, jacp[ids])
  np.testing.assert_array_equal(sub_r, jacr[ids])
  mask = np.zeros(N, dtype=bool)
  mask[5] = True
  np.testing.assert_array_equal(batch.jac_site("tip", ids=mask)[0], jacp[5][None])
  # The call runs kinematics only: qpos is untouched.
  np.testing.assert_array_equal(qpos, expected_qpos)
  from mjbatch._bindings import Batch as RawBatch

  raw = RawBatch(model, 2)
  with pytest.raises(ValueError, match="out of range"):
    raw.jac_site(99, None, np.zeros((2, 3, model.nv)))
  with pytest.raises(ValueError, match="both"):
    raw.jac_site(0, None, None)
  with pytest.raises(ValueError, match="shape"):
    raw.jac_site(0, np.zeros((2, 3, model.nv + 1)), None)


def test_sample_hfield_matches_reference():
  model = mujoco.MjModel.from_xml_string(HFIELD_XML)
  hfield = 0
  nrow, ncol = model.hfield_nrow[hfield], model.hfield_ncol[hfield]
  model.hfield_data[: nrow * ncol] = np.linspace(0, 1, nrow * ncol)  # known ramp
  geom = model.geom("terrain").id
  body = model.body("cart").id
  batch = Batch(model, N, num_threads=3)
  rng = np.random.default_rng(1)
  qpos = batch.bind("qpos")
  qpos[:] = model.qpos0 + rng.uniform(-0.5, 0.5, (N, model.nq))
  offsets = np.array([[i * 0.06, j * 0.06] for i in (-1, 0, 1) for j in (-1, 0, 1)])
  out = batch.sample_hfield("terrain", "cart", offsets)
  assert out.shape == (N, len(offsets))
  for i in range(N):
    np.testing.assert_allclose(out[i], _ref_hfield(model, qpos[i], geom, body, offsets))
  # Points outside the grid clamp to the border height.
  far = np.array([[5.0, 5.0], [-5.0, -5.0]])
  out_far = batch.sample_hfield("terrain", "cart", far, ids=np.array([0]))
  np.testing.assert_allclose(out_far[0], _ref_hfield(model, qpos[0], geom, body, far))
  # Subset rows follow the selection.
  ids = np.array([2, 6])
  np.testing.assert_allclose(batch.sample_hfield("terrain", "cart", offsets, ids=ids), out[ids])
  from mjbatch._bindings import Batch as RawBatch

  raw = RawBatch(model, 2)
  with pytest.raises(ValueError, match="not a hfield"):
    raw.sample_hfield(1, body, offsets, np.zeros((2, len(offsets))))
  with pytest.raises(ValueError, match="out of range"):
    raw.sample_hfield(geom, 99, offsets, np.zeros((2, len(offsets))))
  with pytest.raises(ValueError, match="offsets"):
    raw.sample_hfield(geom, body, np.zeros(3), np.zeros((2, 3)))


# A hfield geom away from the origin, so its rotation moves the sampling frame.
HFIELD_ROT_XML = """
<mujoco>
  <asset>
    <hfield name="hf" nrow="9" ncol="9" size="1 1 0.5 0.1"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="hf" pos="0.3 -0.2 0.1"/>
    <body name="cart" pos="0 0 0.2">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _rot_hfield_model():
  model = mujoco.MjModel.from_xml_string(HFIELD_ROT_XML)
  nrow, ncol = model.hfield_nrow[0], model.hfield_ncol[0]
  model.hfield_data[: nrow * ncol] = np.linspace(0, 1, nrow * ncol)  # known ramp
  return model


def _ref_hfield_aligned(model, qpos_row, geom, body, offsets, alignment, output):
  d = mujoco.MjData(model)
  d.qpos[:] = qpos_row
  mujoco.mj_forward(model, d)
  hfield = model.geom_dataid[geom]
  nrow, ncol = model.hfield_nrow[hfield], model.hfield_ncol[hfield]
  size = model.hfield_size[hfield]
  data = model.hfield_data[hfield * nrow * ncol : (hfield + 1) * nrow * ncol].reshape(nrow, ncol)
  gpos, gmat = d.geom_xpos[geom], d.geom_xmat[geom].reshape(3, 3)
  bpos = d.xpos[body]
  out = np.zeros(len(offsets))
  for k, (ox, oy) in enumerate(offsets):
    r = np.array([ox, oy, 0.0])
    if alignment == "yaw":
      yaw = np.arctan2(gmat[1, 0], gmat[0, 0])
      c, s = np.cos(yaw), np.sin(yaw)
      r = np.array([c * ox - s * oy, s * ox + c * oy, 0.0])
    elif alignment == "body":
      r = gmat @ r
    w = bpos + r
    if alignment != "body":
      w[2] = gpos[2]  # the sampling plane passes through the geom center
    lp = gmat.T @ (w - gpos)
    fx = np.clip((lp[0] / size[0] + 1.0) * 0.5 * (ncol - 1), 0, ncol - 1.001)
    fy = np.clip((lp[1] / size[1] + 1.0) * 0.5 * (nrow - 1), 0, nrow - 1.001)
    ix, iy = int(fx), int(fy)
    sx, sy = fx - ix, fy - iy
    h = (
      (1 - sx) * (1 - sy) * data[iy, ix]
      + sx * (1 - sy) * data[iy, min(ix + 1, ncol - 1)]
      + (1 - sx) * sy * data[min(iy + 1, nrow - 1), ix]
      + sx * sy * data[min(iy + 1, nrow - 1), min(ix + 1, ncol - 1)]
    ) * size[2]
    if output == "height":
      out[k] = h
    else:
      # Signed distance along the geom z-axis from the hfield surface to the
      # query point (the sample point at the body origin's height).
      q = np.array([w[0], w[1], bpos[2] + r[2]])
      surface = gpos + gmat @ np.array([lp[0], lp[1], h])
      out[k] = gmat[:, 2] @ (q - surface)
  return out


def test_sample_hfield_yaw_alignment():
  model = _rot_hfield_model()
  geom, body = model.geom("terrain").id, model.body("cart").id
  batch = Batch(model, N, num_threads=3)
  rng = np.random.default_rng(2)
  qpos = batch.bind("qpos")
  qpos[:] = model.qpos0 + rng.uniform(-0.5, 0.5, (N, model.nq))
  # A per-sim terrain yaw through expanded geom_quat; the grid follows the geom.
  quat = batch.expand("geom_quat")
  yaw = rng.uniform(-np.pi, np.pi, N)
  quat[:, geom] = np.stack([np.cos(yaw / 2), np.zeros(N), np.zeros(N), np.sin(yaw / 2)], axis=1)
  offsets = np.array([[i * 0.06, j * 0.06] for i in (-1, 0, 1) for j in (-1, 0, 1)])
  out = batch.sample_hfield("terrain", "cart", offsets, alignment="yaw")
  assert out.shape == (N, len(offsets))
  for i in range(N):
    model.geom_quat[geom] = quat[i, geom]
    ref = _ref_hfield_aligned(model, qpos[i], geom, body, offsets, "yaw", "height")
    np.testing.assert_allclose(out[i], ref, rtol=1e-7, atol=1e-12)
  # For a pure yaw rotation, body alignment samples the same points.
  np.testing.assert_allclose(batch.sample_hfield("terrain", "cart", offsets, alignment="body"), out)
  # Subset rows follow the selection.
  ids = np.array([2, 6])
  sub = batch.sample_hfield("terrain", "cart", offsets, ids=ids, alignment="yaw")
  np.testing.assert_allclose(sub, out[ids])


def test_sample_hfield_body_alignment():
  model = _rot_hfield_model()
  geom, body = model.geom("terrain").id, model.body("cart").id
  batch = Batch(model, N, num_threads=3)
  rng = np.random.default_rng(3)
  qpos = batch.bind("qpos")
  qpos[:] = model.qpos0 + rng.uniform(-0.5, 0.5, (N, model.nq))
  # Random per-sim terrain orientations, not just yaws.
  quat = batch.expand("geom_quat")
  q = rng.normal(size=(N, 4))
  quat[:, geom] = q / np.linalg.norm(q, axis=1, keepdims=True)
  offsets = np.array([[i * 0.06, j * 0.06] for i in (-1, 0, 1) for j in (-1, 0, 1)])
  out = batch.sample_hfield("terrain", "cart", offsets, alignment="body")
  assert out.shape == (N, len(offsets))
  for i in range(N):
    model.geom_quat[geom] = quat[i, geom]
    ref = _ref_hfield_aligned(model, qpos[i], geom, body, offsets, "body", "height")
    np.testing.assert_allclose(out[i], ref, rtol=1e-7, atol=1e-12)


def test_sample_hfield_clearance():
  model = _rot_hfield_model()
  geom, body = model.geom("terrain").id, model.body("cart").id
  batch = Batch(model, N, num_threads=3)
  rng = np.random.default_rng(4)
  qpos = batch.bind("qpos")
  qpos[:] = model.qpos0 + rng.uniform(-0.5, 0.5, (N, model.nq))
  quat = batch.expand("geom_quat")
  q = rng.normal(size=(N, 4))
  quat[:, geom] = q / np.linalg.norm(q, axis=1, keepdims=True)
  offsets = np.array([[i * 0.06, j * 0.06] for i in (-1, 0, 1) for j in (-1, 0, 1)])
  for alignment in ("world", "yaw", "body"):
    out = batch.sample_hfield("terrain", "cart", offsets, alignment=alignment, output="clearance")
    for i in range(N):
      model.geom_quat[geom] = quat[i, geom]
      ref = _ref_hfield_aligned(model, qpos[i], geom, body, offsets, alignment, "clearance")
      np.testing.assert_allclose(out[i], ref, rtol=1e-7, atol=1e-12)
  # An unrotated geom: clearance is bpos_z - gpos_z - height.
  flat_model = mujoco.MjModel.from_xml_string(HFIELD_XML)
  nrow, ncol = flat_model.hfield_nrow[0], flat_model.hfield_ncol[0]
  flat_model.hfield_data[: nrow * ncol] = np.linspace(0, 1, nrow * ncol)
  flat = Batch(flat_model, N, num_threads=3)
  flat_qpos = flat.bind("qpos")
  flat_qpos[:] = flat_model.qpos0 + rng.uniform(-0.5, 0.5, (N, flat_model.nq))
  h = flat.sample_hfield("terrain", "cart", offsets)
  c = flat.sample_hfield("terrain", "cart", offsets, output="clearance")
  fgeom, fbody = flat_model.geom("terrain").id, flat_model.body("cart").id
  for i in range(N):
    d = mujoco.MjData(flat_model)
    d.qpos[:] = flat_qpos[i]
    mujoco.mj_kinematics(flat_model, d)
    dz = d.xpos[fbody, 2] - d.geom_xpos[fgeom, 2]
    np.testing.assert_allclose(c[i], dz - h[i], rtol=1e-7, atol=1e-12)


def test_sample_hfield_alignment_output_errors():
  model = _rot_hfield_model()
  geom, body = model.geom("terrain").id, model.body("cart").id
  offsets = np.zeros((3, 2))
  batch = Batch(model, 2)
  with pytest.raises(ValueError, match="alignment"):
    batch.sample_hfield("terrain", "cart", offsets, alignment="diagonal")
  with pytest.raises(ValueError, match="output"):
    batch.sample_hfield("terrain", "cart", offsets, output="depth")
  from mjbatch._bindings import Batch as RawBatch

  raw = RawBatch(model, 2)
  with pytest.raises(ValueError, match="alignment"):
    raw.sample_hfield(geom, body, offsets, np.zeros((2, 3)), None, "diagonal", "height")
  with pytest.raises(ValueError, match="output"):
    raw.sample_hfield(geom, body, offsets, np.zeros((2, 3)), None, "world", "depth")
  # Defaults are world/height: the same call spelled out matches the defaults.
  default = batch.sample_hfield("terrain", "cart", offsets)
  spelled = batch.sample_hfield("terrain", "cart", offsets, alignment="world", output="height")
  np.testing.assert_array_equal(default, spelled)
