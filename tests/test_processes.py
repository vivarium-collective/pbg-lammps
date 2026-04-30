"""Unit tests for LAMMPSProcess, PIDController, OscillatingForce."""

import pytest
from process_bigraph import allocate_core
from pbg_lammps.processes import (
    LAMMPSProcess,
    PIDController,
    OscillatingForce,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('LAMMPSProcess', LAMMPSProcess)
    c.register_link('PIDController', PIDController)
    c.register_link('OscillatingForce', OscillatingForce)
    return c


def _basic_lj_script(n=3, density=0.5, ensemble='nve', target_temp=1.0,
                     temperature=1.0, tdamp=0.5, fix_name='integ'):
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
        lines.append(f"fix {fix_name} all nve")
    elif ensemble == 'nvt':
        lines.append(
            f"fix {fix_name} all nvt temp {target_temp} {target_temp} {tdamp}")
    return "\n".join(lines) + "\n"


# ── LAMMPSProcess tests ─────────────────────────────────────────────

def test_instantiation(core):
    proc = LAMMPSProcess(
        config={'input_script': _basic_lj_script(n=3)}, core=core)
    assert 'fix integ all nve' in proc.config['input_script']


def test_inputs_schema(core):
    proc = LAMMPSProcess(
        config={'input_script': _basic_lj_script(n=3)}, core=core)
    inputs = proc.inputs()
    assert inputs['target_temperature'] == 'float'
    assert inputs['target_pressure'] == 'float'
    assert inputs['external_force'] == 'list[float]'


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
        assert port in outputs
        assert outputs[port].startswith('overwrite')


def test_initial_state(core):
    proc = LAMMPSProcess(
        config={'input_script': _basic_lj_script(n=3, density=0.5)},
        core=core)
    state = proc.initial_state()
    assert state['num_atoms'] == 27
    assert len(state['positions']) == 27
    assert len(state['box_dimensions']) == 3
    assert state['volume'] > 0
    proc.close()


def test_single_update_no_inputs(core):
    proc = LAMMPSProcess(
        config={'input_script': _basic_lj_script(n=3, density=0.5)},
        core=core)
    proc.initial_state()
    result = proc.update({}, interval=0.5)
    assert isinstance(result['temperature'], float)
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
    assert abs(e1 - e0) / max(abs(e0), 1e-10) < 0.05
    proc.close()


def test_thermostat_input_port(core):
    """Writing target_temperature should cause LAMMPS to track it."""
    script = _basic_lj_script(
        n=4, density=0.6, ensemble='nvt',
        target_temp=1.0, temperature=1.0, tdamp=0.3)
    proc = LAMMPSProcess(
        config={
            'input_script': script,
            'thermostat_fix': 'integ',
            'thermostat_style': 'nvt',
            'thermostat_damping': 0.3,
        },
        core=core)
    proc.initial_state()

    # Drive the setpoint up.
    for _ in range(8):
        proc.update({'target_temperature': 2.5}, interval=1.0)
    hot = proc.update({'target_temperature': 2.5}, interval=1.0)
    assert hot['temperature'] > 1.5

    # Drive it down — should track to a lower temperature.
    for _ in range(8):
        proc.update({'target_temperature': 0.5}, interval=1.0)
    cold = proc.update({'target_temperature': 0.5}, interval=1.0)
    assert cold['temperature'] < hot['temperature']
    proc.close()


def test_external_force_input_port(core):
    """Writing external_force should perturb kinetic energy."""
    script = _basic_lj_script(n=4, density=0.6, ensemble='nve')
    proc = LAMMPSProcess(
        config={
            'input_script': script,
            'force_fix': 'ext',
        },
        core=core)
    proc.initial_state()

    # Apply zero force baseline.
    base = proc.update({'external_force': [0.0, 0.0, 0.0]}, interval=2.0)
    # Apply a strong force in x. Atoms accelerate; kinetic energy rises.
    for _ in range(5):
        result = proc.update({'external_force': [1.0, 0.0, 0.0]}, interval=2.0)
    assert result['kinetic_energy'] > base['kinetic_energy']
    proc.close()


def test_managed_fix_stripped_from_script(core):
    """When thermostat_fix is configured, a fix line in the script should be
    stripped — the bridge re-issues it from the input port."""
    script = _basic_lj_script(
        n=3, density=0.5, ensemble='nvt', target_temp=1.0,
        temperature=1.0, tdamp=0.5, fix_name='integ')
    proc = LAMMPSProcess(
        config={
            'input_script': script,
            'thermostat_fix': 'integ',
        },
        core=core)
    state = proc.initial_state()
    # If the script's nvt fix had been kept, a control loop wouldn't be able
    # to retarget it at runtime (would conflict with re-fix). The simulation
    # should still run with no thermostat applied yet because no
    # target_temperature was written.
    assert state['num_atoms'] == 27
    proc.close()


def test_run_commands_are_filtered(core):
    script = _basic_lj_script(n=3) + "run 100000\nrun 50000\n"
    proc = LAMMPSProcess(config={'input_script': script}, core=core)
    state = proc.initial_state()
    assert state['num_atoms'] == 27
    proc.close()


def test_filter_run_commands_static():
    out = LAMMPSProcess._filter_run_commands(
        "fix integ all nve\nrun 1000\n  run 500\nrerun foo\n# run 200\n")
    assert 'run 1000' not in out
    assert 'rerun foo' not in out
    assert '# run 200' in out


def test_strip_managed_fixes_static():
    out = LAMMPSProcess._strip_managed_fixes(
        "fix integ all nvt temp 1.0 1.0 0.5\nfix wall all wall/lj93 zlo EDGE 1 1 2\nunfix integ\n",
        ['integ'])
    assert 'fix integ all nvt' not in out
    assert 'unfix integ' not in out
    assert 'fix wall' in out


def test_missing_input_raises(core):
    proc = LAMMPSProcess(config={}, core=core)
    with pytest.raises(ValueError):
        proc.initial_state()


def test_input_file_path(core, tmp_path):
    script = _basic_lj_script(n=3, density=0.5)
    in_path = tmp_path / 'test.in'
    in_path.write_text(script)
    proc = LAMMPSProcess(
        config={'input_file': str(in_path)}, core=core)
    state = proc.initial_state()
    assert state['num_atoms'] == 27
    proc.close()


# ── PIDController tests ─────────────────────────────────────────────

def test_pid_inputs_outputs(core):
    pid = PIDController(config={'target': 2.0, 'kp': 0.5}, core=core)
    assert pid.inputs() == {'measurement': 'float'}
    assert pid.outputs() == {'setpoint': 'overwrite[float]'}


def test_pid_proportional_drives_setpoint(core):
    pid = PIDController(
        config={'target': 2.0, 'kp': 1.0, 'initial_setpoint': 1.0},
        core=core)
    out = pid.update({'measurement': 1.0}, interval=1.0)
    # error = +1.0, kp=1.0 => +1.0 * dt = +1.0 adjustment
    assert abs(out['setpoint'] - 2.0) < 1e-9


def test_pid_clamps_setpoint(core):
    pid = PIDController(
        config={'target': 100.0, 'kp': 1.0, 'setpoint_max': 5.0,
                'initial_setpoint': 1.0},
        core=core)
    out = pid.update({'measurement': 0.0}, interval=10.0)
    assert out['setpoint'] == 5.0


def test_pid_with_no_measurement_holds(core):
    pid = PIDController(
        config={'target': 2.0, 'initial_setpoint': 1.5}, core=core)
    out = pid.update({}, interval=1.0)
    assert out['setpoint'] == 1.5


# ── OscillatingForce tests ──────────────────────────────────────────

def test_oscillating_force_outputs(core):
    drv = OscillatingForce(
        config={'amplitude': 1.0, 'frequency': 0.25,
                'direction': [1.0, 0.0, 0.0]},
        core=core)
    assert drv.outputs() == {'force': 'overwrite[list]'}
    init = drv.initial_state()
    assert init['force'] == [0.0, 0.0, 0.0]
    out = drv.update({}, interval=1.0)
    assert len(out['force']) == 3
    # one quarter period should put us at amplitude.
    assert abs(out['force'][0] - 1.0) < 1e-6
    assert out['force'][1] == 0.0


def test_oscillating_force_returns_to_zero(core):
    drv = OscillatingForce(
        config={'amplitude': 1.0, 'frequency': 0.5,
                'direction': [0.0, 1.0, 0.0]},
        core=core)
    out = drv.update({}, interval=1.0)  # half period -> sin(pi) ≈ 0
    assert abs(out['force'][1]) < 1e-6
