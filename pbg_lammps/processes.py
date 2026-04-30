"""LAMMPS Process and helper Processes for process-bigraph.

LAMMPSProcess wraps the LAMMPS molecular dynamics simulator as a
time-driven, bidirectional Process. Beyond emitting thermodynamic
observables, it accepts per-step control inputs (target temperature,
target pressure, applied force vector) so a sibling controller, sensor,
or environment process can drive its behavior every step.

PIDController is a small Step that closes a feedback loop: it reads
the latest temperature from the bigraph state and writes an updated
target_temperature back into the bigraph. Wiring it next to a
LAMMPSProcess produces a closed-loop control experiment without any
custom plumbing.
"""

import os
from process_bigraph import Process, Step


def _strip_active_command(script, *cmds):
    """Strip lines whose first token is in cmds (ignoring comments/whitespace)."""
    out = []
    targets = set(cmds)
    for line in script.split('\n'):
        stripped = line.split('#', 1)[0].strip()
        tokens = stripped.split()
        if tokens and tokens[0] in targets:
            continue
        out.append(line)
    return '\n'.join(out)


class LAMMPSProcess(Process):
    """Bidirectional Process wrapping the LAMMPS MD engine.

    Configure with a standard LAMMPS input script (file or inline). The
    script defines the simulation box, atoms, pair styles and base
    integrator. Three classes of input commands are stripped at load
    time because the bridge owns them:

      * ``run`` / ``rerun`` — the orchestrator drives integration via
        ``update(interval=...)``
      * the named thermostat / barostat fix (only when the matching
        ``thermostat_fix`` / ``barostat_fix`` config is non-empty) — the
        bridge re-issues the fix each step from input ports so a
        controller can change the setpoint at runtime
      * the named addforce fix (only when ``force_fix`` is non-empty) —
        the bridge re-issues it from the ``external_force`` input port

    Inputs (per-step, written by sibling processes):
        target_temperature : float
            New thermostat setpoint. Applied only when ``thermostat_fix``
            is configured. The bridge unfixes/refixes the thermostat
            each step with the current value (and the configured damping
            and group).
        target_pressure : float
            New barostat setpoint. Applied only when ``barostat_fix``
            is configured.
        external_force : list[float]
            Three-component (Fx, Fy, Fz) external force applied to the
            ``force_group``. Applied only when ``force_fix`` is
            configured.

    Outputs (sensor readings, replace-on-write):
        temperature, potential_energy, kinetic_energy, total_energy,
        pressure, pxx, pyy, pzz, volume     : overwrite[float]
        num_atoms                            : overwrite[integer]
        positions, velocities, atom_types    : overwrite[list]
        box_dimensions                       : overwrite[list]

    Sensor outputs use ``overwrite[T]`` deliberately: each emission
    reports the *current* state of the simulator, not a delta to be
    accumulated.
    """

    config_schema = {
        'input_file': {'_type': 'string', '_default': ''},
        'input_script': {'_type': 'string', '_default': ''},
        'working_directory': {'_type': 'string', '_default': ''},

        # Names of LAMMPS fixes the bridge owns at runtime. If a name
        # is empty, the corresponding input port is ignored and any
        # script-defined fix of the same role is left untouched.
        'thermostat_fix': {'_type': 'string', '_default': ''},
        'thermostat_style': {'_type': 'string', '_default': 'nvt'},
        'thermostat_group': {'_type': 'string', '_default': 'all'},
        'thermostat_damping': {'_type': 'float', '_default': 0.5},

        'barostat_fix': {'_type': 'string', '_default': ''},
        'barostat_style': {'_type': 'string', '_default': 'iso'},
        'barostat_group': {'_type': 'string', '_default': 'all'},
        'barostat_damping': {'_type': 'float', '_default': 1.0},

        'force_fix': {'_type': 'string', '_default': ''},
        'force_group': {'_type': 'string', '_default': 'all'},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._lmp = None
        self._first_run = True
        self._dt = None
        self._last_target_temperature = None
        self._last_target_pressure = None
        self._last_external_force = None
        self._fixes_dirty = False  # True when a managed fix changed since last run

    def inputs(self):
        return {
            'target_temperature': 'float',
            'target_pressure': 'float',
            'external_force': 'list[float]',
        }

    def outputs(self):
        return {
            'temperature': 'overwrite[float]',
            'potential_energy': 'overwrite[float]',
            'kinetic_energy': 'overwrite[float]',
            'total_energy': 'overwrite[float]',
            'pressure': 'overwrite[float]',
            'num_atoms': 'overwrite[integer]',
            'positions': 'overwrite[list]',
            'velocities': 'overwrite[list]',
            'atom_types': 'overwrite[list]',
            'volume': 'overwrite[float]',
            'pxx': 'overwrite[float]',
            'pyy': 'overwrite[float]',
            'pzz': 'overwrite[float]',
            'box_dimensions': 'overwrite[list]',
        }

    @staticmethod
    def _filter_run_commands(script):
        return _strip_active_command(script, 'run', 'rerun')

    @staticmethod
    def _strip_managed_fixes(script, fix_names):
        """Remove ``fix <name> ...`` and ``unfix <name>`` for managed fixes."""
        managed = {n for n in fix_names if n}
        if not managed:
            return script
        out = []
        for line in script.split('\n'):
            stripped = line.split('#', 1)[0].strip()
            tokens = stripped.split()
            if len(tokens) >= 2 and tokens[0] == 'fix' and tokens[1] in managed:
                continue
            if len(tokens) >= 2 and tokens[0] == 'unfix' and tokens[1] in managed:
                continue
            out.append(line)
        return '\n'.join(out)

    def _resolve_script(self):
        cfg = self.config
        if cfg['input_file']:
            path = cfg['input_file']
            with open(path) as f:
                script = f.read()
            wd = cfg['working_directory'] or os.path.dirname(os.path.abspath(path))
            return script, wd
        if cfg['input_script']:
            return cfg['input_script'], cfg['working_directory']
        raise ValueError(
            'LAMMPSProcess requires either input_file or input_script')

    def _build_simulation(self):
        if self._lmp is not None:
            return

        from lammps import lammps

        script, wd = self._resolve_script()
        script = self._filter_run_commands(script)
        script = self._strip_managed_fixes(
            script,
            [self.config['thermostat_fix'],
             self.config['barostat_fix'],
             self.config['force_fix']])

        original_cwd = os.getcwd()
        if wd:
            os.chdir(wd)
        try:
            self._lmp = lammps(cmdargs=['-nocite', '-log', 'none', '-screen', 'none'])
            self._lmp.commands_string(script)
        finally:
            if wd:
                os.chdir(original_cwd)

        self._dt = self._lmp.extract_global('dt')

    def _apply_thermostat(self, target_temperature):
        cfg = self.config
        name = cfg['thermostat_fix']
        if not name or target_temperature is None:
            return
        if target_temperature == self._last_target_temperature:
            return
        if self._last_target_temperature is not None:
            self._lmp.command(f'unfix {name}')
        self._lmp.command(
            f"fix {name} {cfg['thermostat_group']} {cfg['thermostat_style']} "
            f"temp {target_temperature} {target_temperature} {cfg['thermostat_damping']}")
        self._last_target_temperature = target_temperature
        self._fixes_dirty = True

    def _apply_barostat(self, target_pressure):
        cfg = self.config
        name = cfg['barostat_fix']
        if not name or target_pressure is None:
            return
        if target_pressure == self._last_target_pressure:
            return
        if self._last_target_pressure is not None:
            self._lmp.command(f'unfix {name}')
        self._lmp.command(
            f"fix {name} {cfg['barostat_group']} {cfg['barostat_style']} "
            f"{target_pressure} {target_pressure} {cfg['barostat_damping']}")
        self._last_target_pressure = target_pressure
        self._fixes_dirty = True

    def _apply_external_force(self, force_vec):
        cfg = self.config
        name = cfg['force_fix']
        if not name or force_vec is None:
            return
        f = list(force_vec) + [0.0] * max(0, 3 - len(force_vec))
        f = f[:3]
        if f == self._last_external_force:
            return
        if self._last_external_force is not None:
            self._lmp.command(f'unfix {name}')
        self._lmp.command(
            f"fix {name} {cfg['force_group']} addforce {f[0]} {f[1]} {f[2]}")
        self._last_external_force = f
        self._fixes_dirty = True

    def _read_state(self):
        lmp = self._lmp
        natoms = lmp.get_natoms()
        nlocal = lmp.extract_setting('nlocal')

        x = lmp.numpy.extract_atom('x')[:nlocal].copy()
        v = lmp.numpy.extract_atom('v')[:nlocal].copy()
        types = lmp.numpy.extract_atom('type')[:nlocal].copy()

        boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
        lx = boxhi[0] - boxlo[0]
        ly = boxhi[1] - boxlo[1]
        lz = boxhi[2] - boxlo[2]

        return {
            'temperature': float(lmp.get_thermo('temp')),
            'potential_energy': float(lmp.get_thermo('pe')),
            'kinetic_energy': float(lmp.get_thermo('ke')),
            'total_energy': float(lmp.get_thermo('etotal')),
            'pressure': float(lmp.get_thermo('press')),
            'num_atoms': int(natoms),
            'positions': x.tolist(),
            'velocities': v.tolist(),
            'atom_types': types.tolist(),
            'volume': float(lmp.get_thermo('vol')),
            'pxx': float(lmp.get_thermo('pxx')),
            'pyy': float(lmp.get_thermo('pyy')),
            'pzz': float(lmp.get_thermo('pzz')),
            'box_dimensions': [lx, ly, lz],
        }

    def initial_state(self):
        self._build_simulation()
        # Apply any defaults supplied via config-time setpoints so the
        # initial readout reflects the configured controllers.
        self._lmp.command('run 0')
        self._first_run = False
        return self._read_state()

    def update(self, state, interval):
        self._build_simulation()

        # Push upstream state into the simulator before integrating.
        if state:
            self._apply_thermostat(state.get('target_temperature'))
            self._apply_barostat(state.get('target_pressure'))
            self._apply_external_force(state.get('external_force'))

        n_steps = max(1, int(round(interval / self._dt)))

        if self._first_run or self._fixes_dirty:
            # `pre yes` (the default for first_run; explicit here when fixes
            # changed) re-initializes any newly added/modified fixes — without
            # it, a thermostat retargeted via input_port has no effect.
            self._lmp.command(f'run {n_steps}')
            self._first_run = False
            self._fixes_dirty = False
        else:
            self._lmp.command(f'run {n_steps} pre no post no')

        return self._read_state()

    def close(self):
        if self._lmp is not None:
            self._lmp.close()
            self._lmp = None

    def __del__(self):
        try:
            self.close()
        except (ImportError, TypeError):
            pass


class PIDController(Process):
    """Closed-loop PID controller for a scalar setpoint.

    Reads a scalar ``measurement`` from the bigraph and emits an updated
    ``setpoint`` each step. Designed to wire alongside LAMMPSProcess
    (e.g., measurement <- temperature, setpoint -> target_temperature)
    but the math is generic and the schemas are bare floats so the
    output composes with anything else writing to the same store.

    Outputs are emitted as ``overwrite[float]`` because a setpoint is
    a current target, not an accumulating delta — this matches the
    "controller publishing the current setpoint" guidance in the
    process-bigraph port-design rules.
    """

    config_schema = {
        'target': {'_type': 'float', '_default': 1.0},
        'kp': {'_type': 'float', '_default': 1.0},
        'ki': {'_type': 'float', '_default': 0.0},
        'kd': {'_type': 'float', '_default': 0.0},
        'setpoint_min': {'_type': 'float', '_default': 0.05},
        'setpoint_max': {'_type': 'float', '_default': 100.0},
        'initial_setpoint': {'_type': 'float', '_default': 1.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._integral = 0.0
        self._last_error = None
        self._setpoint = self.config['initial_setpoint']

    def inputs(self):
        return {'measurement': 'float'}

    def outputs(self):
        return {'setpoint': 'overwrite[float]'}

    def initial_state(self):
        return {'setpoint': self._setpoint}

    def update(self, state, interval):
        cfg = self.config
        measurement = state.get('measurement')
        if measurement is None:
            return {'setpoint': self._setpoint}

        error = cfg['target'] - measurement
        self._integral += error * interval
        if self._last_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._last_error) / interval if interval > 0 else 0.0
        self._last_error = error

        adjustment = (cfg['kp'] * error
                      + cfg['ki'] * self._integral
                      + cfg['kd'] * derivative)
        new_setpoint = self._setpoint + adjustment * interval
        new_setpoint = max(cfg['setpoint_min'],
                           min(cfg['setpoint_max'], new_setpoint))
        self._setpoint = new_setpoint
        return {'setpoint': new_setpoint}


class OscillatingForce(Process):
    """Time-varying external force vector — useful for driving demos.

    Computes ``force = amplitude * sin(2*pi*frequency*t + phase) * direction``
    and writes it as a ``list[float]`` of length 3. Wire its ``force``
    output to a LAMMPSProcess ``external_force`` input.
    """

    config_schema = {
        'amplitude': {'_type': 'float', '_default': 0.5},
        'frequency': {'_type': 'float', '_default': 0.1},
        'phase': {'_type': 'float', '_default': 0.0},
        'direction': {'_type': 'list[float]', '_default': [1.0, 0.0, 0.0]},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._t = 0.0

    def inputs(self):
        return {}

    def outputs(self):
        return {'force': 'overwrite[list]'}

    def initial_state(self):
        return {'force': [0.0, 0.0, 0.0]}

    def update(self, state, interval):
        import math
        self._t += interval
        cfg = self.config
        s = cfg['amplitude'] * math.sin(
            2.0 * math.pi * cfg['frequency'] * self._t + cfg['phase'])
        d = cfg['direction']
        d = list(d) + [0.0] * max(0, 3 - len(d))
        d = d[:3]
        return {'force': [s * d[0], s * d[1], s * d[2]]}
