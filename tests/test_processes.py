"""Unit tests for LAMMPSProcess."""

import pytest
from process_bigraph import allocate_core
from pbg_lammps.processes import LAMMPSProcess


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('LAMMPSProcess', LAMMPSProcess)
    return c


def test_instantiation(core):
    proc = LAMMPSProcess(
        config={'num_atoms_per_dim': 3, 'temperature': 1.0},
        core=core)
    assert proc.config['num_atoms_per_dim'] == 3
    assert proc.config['ensemble'] == 'nve'


def test_initial_state(core):
    proc = LAMMPSProcess(
        config={'num_atoms_per_dim': 3, 'density': 0.5},
        core=core)
    state = proc.initial_state()
    assert 'temperature' in state
    assert 'potential_energy' in state
    assert 'total_energy' in state
    assert 'pressure' in state
    assert 'positions' in state
    assert 'velocities' in state
    assert state['num_atoms'] == 27  # 3^3 for sc lattice
    assert len(state['positions']) == 27
    assert len(state['velocities']) == 27
    assert state['temperature'] >= 0
    proc.close()


def test_single_update(core):
    proc = LAMMPSProcess(
        config={'num_atoms_per_dim': 3, 'density': 0.5},
        core=core)
    proc.initial_state()
    result = proc.update({}, interval=0.5)
    assert 'temperature' in result
    assert 'total_energy' in result
    assert isinstance(result['temperature'], float)
    assert isinstance(result['total_energy'], float)
    assert isinstance(result['pressure'], float)
    assert result['num_atoms'] == 27
    proc.close()


def test_nve_energy_conservation(core):
    proc = LAMMPSProcess(
        config={
            'num_atoms_per_dim': 4,
            'density': 0.6,
            'temperature': 1.0,
            'ensemble': 'nve',
        },
        core=core)
    state0 = proc.initial_state()
    e0 = state0['total_energy']

    result = proc.update({}, interval=1.0)
    e1 = result['total_energy']
    # NVE should conserve energy within a few percent
    assert abs(e1 - e0) / max(abs(e0), 1e-10) < 0.05, \
        f'NVE energy drift too large: {e0} -> {e1}'
    proc.close()


def test_nvt_temperature(core):
    proc = LAMMPSProcess(
        config={
            'num_atoms_per_dim': 4,
            'density': 0.6,
            'temperature': 2.0,
            'ensemble': 'nvt',
            'target_temp': 1.5,
            'tdamp': 0.5,
        },
        core=core)
    proc.initial_state()
    # Run longer to let thermostat act
    result = proc.update({}, interval=5.0)
    # Temperature should be in the right ballpark (not exact due to fluctuations)
    assert 0.5 < result['temperature'] < 3.0, \
        f'NVT temperature out of range: {result["temperature"]}'
    proc.close()


def test_outputs_schema(core):
    proc = LAMMPSProcess(config={'num_atoms_per_dim': 3}, core=core)
    outputs = proc.outputs()
    expected_ports = [
        'temperature', 'potential_energy', 'kinetic_energy',
        'total_energy', 'pressure', 'num_atoms',
        'positions', 'velocities',
    ]
    for port in expected_ports:
        assert port in outputs, f'Missing output port: {port}'
    proc.close()


def test_config_defaults(core):
    proc = LAMMPSProcess(config={}, core=core)
    assert proc.config['num_atoms_per_dim'] == 5
    assert proc.config['density'] == 0.6
    assert proc.config['lattice_style'] == 'sc'
    assert proc.config['temperature'] == 1.0
    assert proc.config['timestep'] == 0.005
    assert proc.config['ensemble'] == 'nve'
    assert proc.config['mass'] == 1.0
    proc.close()


def test_positions_are_3d(core):
    proc = LAMMPSProcess(
        config={'num_atoms_per_dim': 3},
        core=core)
    state = proc.initial_state()
    for pos in state['positions']:
        assert len(pos) == 3, f'Position is not 3D: {pos}'
    for vel in state['velocities']:
        assert len(vel) == 3, f'Velocity is not 3D: {vel}'
    proc.close()


def test_multiple_updates(core):
    proc = LAMMPSProcess(
        config={'num_atoms_per_dim': 3, 'density': 0.5},
        core=core)
    proc.initial_state()
    results = []
    for _ in range(5):
        r = proc.update({}, interval=0.25)
        results.append(r['total_energy'])
    # Should get different energies (system is evolving)
    assert len(set(f'{e:.6f}' for e in results)) > 1
    proc.close()
