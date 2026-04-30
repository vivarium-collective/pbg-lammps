"""Integration tests for LAMMPS composites including closed-loop control."""

import pytest
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter

from pbg_lammps import (
    LAMMPSProcess,
    PIDController,
    OscillatingForce,
    make_lammps_document,
    make_pid_controlled_document,
    make_force_driven_document,
    register_pbg_lammps,
)


SIMPLE_NVE_SCRIPT = """
units lj
atom_style atomic
dimension 3
boundary p p p
lattice sc 0.5
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_modify shift yes
velocity all create 1.0 87287 dist gaussian
timestep 0.005
fix integ all nve
"""

NVT_SCRIPT = """
units lj
atom_style atomic
dimension 3
boundary p p p
lattice sc 0.6
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_modify shift yes
velocity all create 1.5 87287 dist gaussian
timestep 0.005
fix integ all nvt temp 1.0 1.0 0.5
"""


@pytest.fixture
def core():
    c = allocate_core()
    register_pbg_lammps(c)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_composite_assembly(core):
    doc = make_lammps_document(input_script=SIMPLE_NVE_SCRIPT, interval=0.5)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


def test_composite_short_run(core):
    doc = make_lammps_document(input_script=SIMPLE_NVE_SCRIPT, interval=0.5)
    sim = Composite({'state': doc}, core=core)
    sim.run(1.0)
    stores = sim.state['stores']
    assert stores['temperature'] > 0
    assert stores['num_atoms'] == 27


def test_emitter_collects_timeseries(core):
    doc = make_lammps_document(input_script=SIMPLE_NVE_SCRIPT, interval=0.5)
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)
    raw = gather_emitter_results(sim)
    series = raw[('emitter',)]
    assert len(series) >= 2
    assert 'total_energy' in series[0]
    assert 'time' in series[0]


def test_document_factory_requires_input():
    with pytest.raises(ValueError):
        make_lammps_document(interval=1.0)


def test_pid_loop_drives_temperature(core):
    """The PIDController should pull temperature toward its target."""
    doc = make_pid_controlled_document(
        input_script=NVT_SCRIPT,
        target_temperature=2.5,
        interval=1.0,
        thermostat_damping=0.3,
        kp=1.0,
        initial_target_temperature=1.0,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(20.0)
    raw = gather_emitter_results(sim)
    series = raw[('emitter',)]

    # Setpoint should rise toward target; final temperature should be
    # closer to target than the starting NVT setpoint of 1.0.
    final = series[-1]
    assert final['target_temperature'] > 1.0
    assert final['temperature'] > 1.5


def test_force_driven_document_runs(core):
    doc = make_force_driven_document(
        input_script=SIMPLE_NVE_SCRIPT,
        interval=0.5,
        amplitude=0.5,
        frequency=0.2,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)
    stores = sim.state['stores']
    assert stores['num_atoms'] == 27
    assert stores['kinetic_energy'] > 0
