"""Integration tests for LAMMPS composites."""

import pytest
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from pbg_lammps.processes import LAMMPSProcess
from pbg_lammps.composites import make_lammps_document


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('LAMMPSProcess', LAMMPSProcess)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_composite_assembly(core):
    doc = make_lammps_document(
        num_atoms_per_dim=3,
        density=0.5,
        interval=0.5)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


def test_composite_short_run(core):
    doc = make_lammps_document(
        num_atoms_per_dim=3,
        density=0.5,
        interval=0.5)
    sim = Composite({'state': doc}, core=core)
    sim.run(1.0)

    stores = sim.state['stores']
    assert stores['temperature'] > 0
    assert stores['num_atoms'] == 27
    assert isinstance(stores['total_energy'], float)
    assert len(stores['positions']) == 27


def test_emitter_collects_timeseries(core):
    doc = make_lammps_document(
        num_atoms_per_dim=3,
        density=0.5,
        interval=0.5)
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
    doc = make_lammps_document(
        num_atoms_per_dim=3,
        density=0.6,
        ensemble='nvt',
        target_temp=1.5,
        tdamp=0.5,
        temperature=2.0,
        interval=1.0)
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)
    stores = sim.state['stores']
    assert stores['temperature'] > 0
    assert stores['num_atoms'] == 27
