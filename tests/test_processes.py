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
    assert 'atom_types' in state
    assert 'volume' in state
    assert 'box_dimensions' in state
    assert state['num_atoms'] == 27  # 3^3 for sc lattice
    assert len(state['positions']) == 27
    assert len(state['atom_types']) == 27
    assert all(t == 1 for t in state['atom_types'])
    assert state['volume'] > 0
    assert len(state['box_dimensions']) == 3
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
    assert isinstance(result['pxx'], float)
    assert isinstance(result['pyy'], float)
    assert isinstance(result['pzz'], float)
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
    result = proc.update({}, interval=5.0)
    assert 0.5 < result['temperature'] < 3.0, \
        f'NVT temperature out of range: {result["temperature"]}'
    proc.close()


def test_outputs_schema(core):
    proc = LAMMPSProcess(config={'num_atoms_per_dim': 3}, core=core)
    outputs = proc.outputs()
    expected_ports = [
        'temperature', 'potential_energy', 'kinetic_energy',
        'total_energy', 'pressure', 'num_atoms',
        'positions', 'velocities', 'atom_types',
        'volume', 'pxx', 'pyy', 'pzz', 'box_dimensions',
    ]
    for port in expected_ports:
        assert port in outputs, f'Missing output port: {port}'
    proc.close()


def test_config_defaults(core):
    proc = LAMMPSProcess(config={}, core=core)
    assert proc.config['num_atoms_per_dim'] == 5
    assert proc.config['density'] == 0.6
    assert proc.config['lattice_style'] == 'sc'
    assert proc.config['ensemble'] == 'nve'
    assert proc.config['setup_commands'] == ''
    proc.close()


def test_positions_are_3d(core):
    proc = LAMMPSProcess(
        config={'num_atoms_per_dim': 3},
        core=core)
    state = proc.initial_state()
    for pos in state['positions']:
        assert len(pos) == 3
    for vel in state['velocities']:
        assert len(vel) == 3
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
    assert len(set(f'{e:.6f}' for e in results)) > 1
    proc.close()


def test_setup_commands_binary_system(core):
    """Test that setup_commands enables multi-type systems."""
    proc = LAMMPSProcess(
        config={
            'setup_commands': """
units lj
atom_style atomic
dimension 3
boundary p p p
lattice fcc 1.2
region box block 0 4 0 4 0 4
create_box 2 box
create_atoms 1 box
mass 1 1.0
mass 2 1.0
set type 1 type/fraction 2 0.2 12345
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
pair_coeff 1 2 1.5 0.8 2.0
pair_coeff 2 2 0.5 0.88 2.2
pair_modify shift yes
velocity all create 1.0 87287 dist gaussian
timestep 0.005
fix integ all nvt temp 1.0 1.0 0.5
""",
            'timestep': 0.005,
        },
        core=core)
    state = proc.initial_state()
    types = state['atom_types']
    assert 1 in types
    assert 2 in types
    assert state['num_atoms'] == 256  # 4^3 * 4 for fcc
    proc.close()


def test_setup_commands_2d(core):
    """Test that setup_commands enables 2D simulations."""
    proc = LAMMPSProcess(
        config={
            'setup_commands': """
units lj
atom_style atomic
dimension 2
boundary p p p
lattice hex 0.85
region box block 0 10 0 10 -0.5 0.5
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_modify shift yes
velocity all create 0.5 87287 dist gaussian
timestep 0.005
fix integ all nvt temp 0.5 0.5 0.5
fix enforce all enforce2d
""",
            'timestep': 0.005,
        },
        core=core)
    state = proc.initial_state()
    assert state['num_atoms'] > 0
    # All z-positions should be 0 in 2D
    for pos in state['positions']:
        assert abs(pos[2]) < 1e-10
    proc.close()
