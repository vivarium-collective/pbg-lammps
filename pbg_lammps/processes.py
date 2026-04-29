"""LAMMPS Process wrapper for process-bigraph.

Wraps the LAMMPS molecular dynamics simulator as a time-driven Process.
The process accepts a standard LAMMPS input script (either a path to a
.in file or an inline string), executes the setup, and then advances
the integrator on each update(). Any `run` / `rerun` commands present
in the script are filtered out — the process-bigraph orchestrator
drives integration via update(interval=...).
"""

import os
from process_bigraph import Process


class LAMMPSProcess(Process):
    """Bridge Process wrapping the LAMMPS molecular dynamics engine.

    Configure with a standard LAMMPS input script (the same format
    LAMMPS itself reads, e.g. https://docs.lammps.org/Commands_input.html).
    `run` and `rerun` commands are stripped out at load time — the
    bridge issues `run N` calls itself based on the requested interval.

    Config:
        input_file: path to a LAMMPS .in input file
        input_script: inline LAMMPS script content (alternative to input_file)
        working_directory: directory used for resolving relative paths
            in commands like `read_data` (defaults to the directory
            containing input_file, or CWD for input_script)
    """

    config_schema = {
        'input_file': {'_type': 'string', '_default': ''},
        'input_script': {'_type': 'string', '_default': ''},
        'working_directory': {'_type': 'string', '_default': ''},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._lmp = None
        self._first_run = True
        self._dt = None

    def inputs(self):
        return {}

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
        """Strip `run` / `rerun` commands so the bridge can drive integration."""
        out = []
        for line in script.split('\n'):
            stripped = line.split('#', 1)[0].strip()
            tokens = stripped.split()
            if tokens and tokens[0] in ('run', 'rerun'):
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

        original_cwd = os.getcwd()
        if wd:
            os.chdir(wd)
        try:
            self._lmp = lammps(cmdargs=['-nocite', '-log', 'none', '-screen', 'none'])
            self._lmp.commands_string(script)
        finally:
            if wd:
                os.chdir(original_cwd)

        self._dt = self._lmp.extract_global("dt")

    def _read_state(self):
        """Read current thermodynamic state and positions from LAMMPS."""
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
        self._lmp.command('run 0')
        self._first_run = False
        return self._read_state()

    def update(self, state, interval):
        self._build_simulation()

        n_steps = max(1, int(round(interval / self._dt)))

        if self._first_run:
            self._lmp.command(f'run {n_steps}')
            self._first_run = False
        else:
            self._lmp.command(f'run {n_steps} pre no post no')

        return self._read_state()

    def close(self):
        """Explicitly close the LAMMPS instance."""
        if self._lmp is not None:
            self._lmp.close()
            self._lmp = None

    def __del__(self):
        try:
            self.close()
        except (ImportError, TypeError):
            pass
