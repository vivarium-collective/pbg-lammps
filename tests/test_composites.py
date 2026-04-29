"""Integration tests for LAMMPS composites."""

import pytest
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from pbg_lammps.processes import LAMMPSProcess
from pbg_lammps.composites import make_lammps_document


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
velocity all create 2.0 87287 dist gaussian
timestep 0.005
fix integ all nvt temp 1.5 1.5 0.5
"""


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('LAMMPSProcess', LAMMPSProcess)
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
    assert isinstance(stores['total_energy'], float)
    assert len(stores['positions']) == 27


def test_emitter_collects_timeseries(core):
    doc = make_lammps_document(input_script=SIMPLE_NVE_SCRIPT, interval=0.5)
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)

    raw_results = gather_emitter_results(sim)
    emitter_data = raw_results[('emitter',)]
    assert len(emitter_data) >= 2
    assert 'total_energy' in emitter_data[0]
    assert 'time' in emitter_data[0]
    for entry in emitter_data[1:]:
        assert entry['total_energy'] is not None


def test_document_factory_nvt(core):
    doc = make_lammps_document(input_script=NVT_SCRIPT, interval=1.0)
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)
    stores = sim.state['stores']
    assert stores['temperature'] > 0
    assert stores['num_atoms'] == 27


def test_document_factory_requires_input():
    with pytest.raises(ValueError):
        make_lammps_document(interval=1.0)


def test_document_factory_input_file(core, tmp_path):
    in_path = tmp_path / 'simple.in'
    in_path.write_text(SIMPLE_NVE_SCRIPT)
    doc = make_lammps_document(input_file=str(in_path), interval=0.5)
    sim = Composite({'state': doc}, core=core)
    sim.run(1.0)
    assert sim.state['stores']['num_atoms'] == 27
