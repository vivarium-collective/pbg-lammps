"""Pre-built composite document factories for LAMMPS simulations.

The factories below produce process-bigraph documents that wire the
LAMMPSProcess input ports (target_temperature, target_pressure,
external_force) to bigraph stores so that:

* ``make_lammps_document`` exposes those control ports as constant
  stores callers can pre-populate or that downstream code can write to,
* ``make_pid_controlled_document`` wires a PIDController next to the
  simulator and closes a feedback loop on temperature,
* ``make_force_driven_document`` wires an OscillatingForce process to
  drive the LAMMPS external_force input with a time-varying signal.
"""


_SENSOR_PORTS = (
    'temperature', 'potential_energy', 'kinetic_energy', 'total_energy',
    'pressure', 'num_atoms', 'positions', 'velocities', 'atom_types',
    'volume', 'pxx', 'pyy', 'pzz', 'box_dimensions',
)


def _lammps_node(input_file, input_script, working_directory, interval,
                 thermostat_fix='', thermostat_style='nvt',
                 thermostat_group='all', thermostat_damping=0.5,
                 barostat_fix='', barostat_style='iso',
                 barostat_group='all', barostat_damping=1.0,
                 force_fix='', force_group='all'):
    if not input_file and not input_script:
        raise ValueError(
            'LAMMPS document requires input_file or input_script')

    config = {
        'input_file': input_file,
        'input_script': input_script,
        'working_directory': working_directory,
        'thermostat_fix': thermostat_fix,
        'thermostat_style': thermostat_style,
        'thermostat_group': thermostat_group,
        'thermostat_damping': thermostat_damping,
        'barostat_fix': barostat_fix,
        'barostat_style': barostat_style,
        'barostat_group': barostat_group,
        'barostat_damping': barostat_damping,
        'force_fix': force_fix,
        'force_group': force_group,
    }
    inputs = {
        'target_temperature': ['controls', 'target_temperature'],
        'target_pressure': ['controls', 'target_pressure'],
        'external_force': ['controls', 'external_force'],
    }
    outputs = {p: ['stores', p] for p in _SENSOR_PORTS}
    return {
        '_type': 'process',
        'address': 'local:LAMMPSProcess',
        'config': config,
        'interval': interval,
        'inputs': inputs,
        'outputs': outputs,
    }


def _emitter_node(scalar_ports, controls=()):
    emit_schema = {p: 'float' for p in scalar_ports}
    emit_schema['time'] = 'float'
    inputs = {p: ['stores', p] for p in scalar_ports}
    inputs['time'] = ['global_time']
    for c in controls:
        emit_schema[c] = 'float'
        inputs[c] = ['controls', c]
    return {
        '_type': 'step',
        'address': 'local:ram-emitter',
        'config': {'emit': emit_schema},
        'inputs': inputs,
    }


def make_lammps_document(
    input_file='',
    input_script='',
    working_directory='',
    interval=1.0,
    initial_target_temperature=1.0,
    initial_target_pressure=1.0,
    initial_external_force=(0.0, 0.0, 0.0),
    thermostat_fix='',
    barostat_fix='',
    force_fix='',
    thermostat_damping=0.5,
):
    """Composite document with LAMMPS control ports exposed as stores.

    By default the bridge does not own any thermostat/barostat/addforce
    fix — pass ``thermostat_fix='integ'`` (etc.) to let the bridge
    take ownership of those fixes and let upstream code write to the
    matching control store.
    """
    return {
        'lammps': _lammps_node(
            input_file=input_file,
            input_script=input_script,
            working_directory=working_directory,
            interval=interval,
            thermostat_fix=thermostat_fix,
            thermostat_damping=thermostat_damping,
            barostat_fix=barostat_fix,
            force_fix=force_fix),
        'controls': {
            'target_temperature': float(initial_target_temperature),
            'target_pressure': float(initial_target_pressure),
            'external_force': list(initial_external_force),
        },
        'stores': {},
        'emitter': _emitter_node(
            scalar_ports=('temperature', 'potential_energy', 'kinetic_energy',
                          'total_energy', 'pressure', 'volume',
                          'pxx', 'pyy', 'pzz'),
            controls=('target_temperature',)),
    }


def make_pid_controlled_document(
    input_script,
    target_temperature,
    interval=1.0,
    thermostat_fix='integ',
    thermostat_damping=0.5,
    initial_target_temperature=None,
    kp=1.0,
    ki=0.0,
    kd=0.0,
    controller_interval=None,
):
    """Closed-loop document: PIDController.setpoint -> LAMMPS.target_temperature.

    The controller observes the current measured temperature from the
    bigraph state and adjusts the LAMMPS thermostat setpoint each step.
    Both processes share the ``controls/target_temperature`` store as
    a single source of truth.
    """
    initial = (initial_target_temperature
               if initial_target_temperature is not None
               else target_temperature)
    controller_interval = controller_interval or interval

    doc = make_lammps_document(
        input_script=input_script,
        interval=interval,
        thermostat_fix=thermostat_fix,
        thermostat_damping=thermostat_damping,
        initial_target_temperature=initial,
    )

    doc['controller'] = {
        '_type': 'process',
        'address': 'local:PIDController',
        'config': {
            'target': float(target_temperature),
            'kp': kp, 'ki': ki, 'kd': kd,
            'initial_setpoint': float(initial),
        },
        'interval': controller_interval,
        'inputs': {'measurement': ['stores', 'temperature']},
        'outputs': {'setpoint': ['controls', 'target_temperature']},
    }
    return doc


def make_force_driven_document(
    input_script,
    interval=1.0,
    force_fix='ext',
    amplitude=0.5,
    frequency=0.1,
    direction=(1.0, 0.0, 0.0),
):
    """Document where an OscillatingForce drives LAMMPS.external_force."""
    doc = make_lammps_document(
        input_script=input_script,
        interval=interval,
        force_fix=force_fix,
    )
    doc['driver'] = {
        '_type': 'process',
        'address': 'local:OscillatingForce',
        'config': {
            'amplitude': amplitude,
            'frequency': frequency,
            'direction': list(direction),
        },
        'interval': interval,
        'inputs': {},
        'outputs': {'force': ['controls', 'external_force']},
    }
    return doc
