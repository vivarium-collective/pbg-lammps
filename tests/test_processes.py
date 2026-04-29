"""Unit tests for LAMMPSProcess."""

import os
import tempfile

import pytest
from process_bigraph import allocate_core
from pbg_lammps.processes import LAMMPSProcess


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('LAMMPSProcess', LAMMPSProcess)
    return c


def _basic_lj_script(n=3, density=0.5, ensemble='nve', target_temp=1.0,
                     temperature=1.0, tdamp=0.5):
    """Minimal single-type LJ input script for tests."""
    lines = [
        "units lj",
        "atom_style atomic",
        "dimension 3",
        "boundary p p p",
        f"lattice sc {density}",
        f"region box block 0 {n} 0 {n} 0 {n}",
        "create_box 1 box",
        "create_atoms 1 box",
        "mass 1 1.0",
        "pair_style lj/cut 2.5",
        "pair_coeff 1 1 1.0 1.0",
        "pair_modify shift yes",
        f"velocity all create {temperature} 87287 dist gaussian",
        "timestep 0.005",
    ]
    if ensemble == 'nve':
        lines.append("fix integ all nve")
    elif ensemble == 'nvt':
        lines.append(
            f"fix integ all nvt temp {target_temp} {target_temp} {tdamp}")
    return "\n".join(lines) + "\n"


# ── Tests ───────────────────────────────────────────────────────────

def test_instantiation(core):
    proc = LAMMPSProcess(
        config={'input_script': _basic_lj_script(n=3)},
        core=core)
    assert 'fix integ all nve' in proc.config['input_script']


def test_initial_state(core):
    proc = LAMMPSProcess(
        config={'input_script': _basic_lj_script(n=3, density=0.5)},
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
        config={'input_script': _basic_lj_script(n=3, density=0.5)},
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
        config={'input_script': _basic_lj_script(
            n=4, density=0.6, ensemble='nve', temperature=1.0)},
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
        config={'input_script': _basic_lj_script(
            n=4, density=0.6, ensemble='nvt',
            target_temp=1.5, temperature=2.0, tdamp=0.5)},
        core=core)
    proc.initial_state()
    result = proc.update({}, interval=5.0)
    assert 0.5 < result['temperature'] < 3.0, \
        f'NVT temperature out of range: {result["temperature"]}'
    proc.close()


def test_outputs_schema(core):
    proc = LAMMPSProcess(
        config={'input_script': _basic_lj_script(n=3)}, core=core)
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
    """Empty config should fail — input_file or input_script is required."""
    proc = LAMMPSProcess(config={}, core=core)
    assert proc.config['input_file'] == ''
    assert proc.config['input_script'] == ''
    assert proc.config['working_directory'] == ''
    with pytest.raises(ValueError):
        proc.initial_state()


def test_positions_are_3d(core):
    proc = LAMMPSProcess(
        config={'input_script': _basic_lj_script(n=3)}, core=core)
    state = proc.initial_state()
    for pos in state['positions']:
        assert len(pos) == 3
    for vel in state['velocities']:
        assert len(vel) == 3
    proc.close()


def test_multiple_updates(core):
    proc = LAMMPSProcess(
        config={'input_script': _basic_lj_script(n=3, density=0.5)},
        core=core)
    proc.initial_state()
    results = []
    for _ in range(5):
        r = proc.update({}, interval=0.25)
        results.append(r['total_energy'])
    assert len(set(f'{e:.6f}' for e in results)) > 1
    proc.close()


def test_input_script_binary_system(core):
    """Multi-type setups via input_script."""
    script = """
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
"""
    proc = LAMMPSProcess(config={'input_script': script}, core=core)
    state = proc.initial_state()
    types = state['atom_types']
    assert 1 in types
    assert 2 in types
    assert state['num_atoms'] == 256  # 4^3 * 4 for fcc
    proc.close()


def test_input_script_2d(core):
    """2D simulations via input_script."""
    script = """
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
"""
    proc = LAMMPSProcess(config={'input_script': script}, core=core)
    state = proc.initial_state()
    assert state['num_atoms'] > 0
    for pos in state['positions']:
        assert abs(pos[2]) < 1e-10
    proc.close()


def test_run_commands_are_filtered(core):
    """User-written `run` commands must be stripped — the bridge controls timestepping."""
    script = _basic_lj_script(n=3) + "run 100000\nrun 50000\n"
    proc = LAMMPSProcess(config={'input_script': script}, core=core)
    state = proc.initial_state()  # would take a long time if run weren't filtered
    assert state['num_atoms'] == 27
    proc.close()


def test_filter_run_commands_static():
    script = (
        "fix integ all nve\n"
        "run 1000\n"
        "  run 500 # with leading whitespace\n"
        "rerun foo\n"
        "# run 200 (comment, keep)\n"
        "variable run_count equal 5\n"  # 'run_count' not 'run'
    )
    out = LAMMPSProcess._filter_run_commands(script)
    assert 'run 1000' not in out
    assert 'run 500' not in out
    assert 'rerun foo' not in out
    assert '# run 200' in out
    assert 'variable run_count equal 5' in out


def test_input_file_path(core, tmp_path):
    """Loading from an actual .in file on disk."""
    script = _basic_lj_script(n=3, density=0.5)
    in_path = tmp_path / 'test.in'
    in_path.write_text(script)
    proc = LAMMPSProcess(
        config={'input_file': str(in_path)}, core=core)
    state = proc.initial_state()
    assert state['num_atoms'] == 27
    proc.close()


def test_missing_input_raises(core):
    """No input_file and no input_script should raise on build."""
    proc = LAMMPSProcess(config={}, core=core)
    with pytest.raises(ValueError, match='input_file or input_script'):
        proc.initial_state()
