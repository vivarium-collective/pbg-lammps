"""LAMMPS Process wrapper for process-bigraph.

Wraps the LAMMPS molecular dynamics simulator as a time-driven Process
using the bridge pattern. The internal LAMMPS instance is lazily
initialized on the first update() call.
"""

import numpy as np
from process_bigraph import Process


class LAMMPSProcess(Process):
    """Bridge Process wrapping the LAMMPS molecular dynamics engine.

    Simulates atomic/molecular systems using classical force fields.
    On each update(), advances the LAMMPS integrator by the requested
    time interval, then returns updated thermodynamic quantities and
    particle positions as overwrites.

    Two modes of setup:
    1. **Config-based** (simple): Set individual parameters like density,
       ensemble, temperature. The process auto-generates LAMMPS commands.
    2. **Script-based** (advanced): Provide raw LAMMPS commands via
       setup_commands. This enables multi-type systems, custom potentials,
       fix deform, 2D simulations, etc.

    Config:
        setup_commands: Raw LAMMPS setup script (overrides auto-generation)
        timestep: Integration timestep (used to convert interval to steps)
        num_atoms_per_dim: Lattice repeats per dimension (auto mode)
        density: Number density (auto mode)
        lattice_style: Lattice type (auto mode)
        temperature: Initial temperature (auto mode)
        ensemble: Integration ensemble (auto mode)
        ... (see config_schema for full list)
    """

    config_schema = {
        # Advanced: raw LAMMPS script (overrides all auto-generation below)
        'setup_commands': {'_type': 'string', '_default': ''},
        # System geometry
        'num_atoms_per_dim': {'_type': 'integer', '_default': 5},
        'density': {'_type': 'float', '_default': 0.6},
        'lattice_style': {'_type': 'string', '_default': 'sc'},
        # Interaction potential
        'pair_style': {'_type': 'string', '_default': 'lj/cut'},
        'epsilon': {'_type': 'float', '_default': 1.0},
        'sigma': {'_type': 'float', '_default': 1.0},
        'cutoff': {'_type': 'float', '_default': 2.5},
        'mass': {'_type': 'float', '_default': 1.0},
        # Temperature / velocities
        'temperature': {'_type': 'float', '_default': 1.0},
        'seed': {'_type': 'integer', '_default': 87287},
        # Integration
        'timestep': {'_type': 'float', '_default': 0.005},
        'ensemble': {'_type': 'string', '_default': 'nve'},
        'target_temp': {'_type': 'float', '_default': 1.0},
        'tdamp': {'_type': 'float', '_default': 0.5},
        'target_press': {'_type': 'float', '_default': 0.0},
        'pdamp': {'_type': 'float', '_default': 5.0},
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

    def _build_simulation(self):
        """Lazily initialize the LAMMPS instance."""
        if self._lmp is not None:
            return

        from lammps import lammps

        cfg = self.config
        self._lmp = lammps(cmdargs=['-nocite', '-log', 'none', '-screen', 'none'])

        if cfg['setup_commands']:
            self._lmp.commands_string(cfg['setup_commands'])
        else:
            n = cfg['num_atoms_per_dim']
            setup = f"""
units lj
atom_style atomic
dimension 3
boundary p p p

lattice {cfg['lattice_style']} {cfg['density']}
region box block 0 {n} 0 {n} 0 {n}
create_box 1 box
create_atoms 1 box
mass 1 {cfg['mass']}

pair_style {cfg['pair_style']} {cfg['cutoff']}
pair_coeff 1 1 {cfg['epsilon']} {cfg['sigma']}
pair_modify shift yes

velocity all create {cfg['temperature']} {cfg['seed']} dist gaussian

timestep {cfg['timestep']}
"""
            self._lmp.commands_string(setup)

            if cfg['ensemble'] == 'nve':
                self._lmp.command('fix integ all nve')
            elif cfg['ensemble'] == 'nvt':
                self._lmp.command(
                    f"fix integ all nvt temp {cfg['target_temp']} "
                    f"{cfg['target_temp']} {cfg['tdamp']}")
            elif cfg['ensemble'] == 'npt':
                self._lmp.command(
                    f"fix integ all npt temp {cfg['target_temp']} "
                    f"{cfg['target_temp']} {cfg['tdamp']} "
                    f"iso {cfg['target_press']} {cfg['target_press']} "
                    f"{cfg['pdamp']}")
            else:
                raise ValueError(
                    f"Unknown ensemble: {cfg['ensemble']}. "
                    f"Use 'nve', 'nvt', or 'npt'.")

        # Read actual timestep from LAMMPS
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
