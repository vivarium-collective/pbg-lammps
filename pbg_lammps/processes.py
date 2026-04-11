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

    Config:
        num_atoms_per_dim: Number of lattice repeats per dimension
        density: Number density (for lattice command)
        lattice_style: Lattice type ('sc', 'fcc', 'bcc')
        temperature: Initial temperature for velocity creation
        timestep: Integration timestep (LJ units)
        pair_style: LAMMPS pair_style command string
        pair_coeff: LAMMPS pair_coeff arguments
        epsilon: LJ energy parameter
        sigma: LJ length parameter
        cutoff: Pair interaction cutoff
        mass: Atom mass
        ensemble: Integration ensemble ('nve', 'nvt', 'npt')
        target_temp: Target temperature for NVT/NPT
        tdamp: Thermostat damping parameter
        target_press: Target pressure for NPT
        pdamp: Barostat damping parameter
        seed: Random seed for velocity initialization
    """

    config_schema = {
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
        }

    def _build_simulation(self):
        """Lazily initialize the LAMMPS instance."""
        if self._lmp is not None:
            return

        from lammps import lammps

        cfg = self.config
        n = cfg['num_atoms_per_dim']

        self._lmp = lammps(cmdargs=['-nocite', '-log', 'none', '-screen', 'none'])

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

        # Set up ensemble
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

    def _read_state(self):
        """Read current thermodynamic state and positions from LAMMPS."""
        lmp = self._lmp
        natoms = lmp.get_natoms()
        nlocal = lmp.extract_setting('nlocal')

        x = lmp.numpy.extract_atom('x')[:nlocal].copy()
        v = lmp.numpy.extract_atom('v')[:nlocal].copy()

        return {
            'temperature': float(lmp.get_thermo('temp')),
            'potential_energy': float(lmp.get_thermo('pe')),
            'kinetic_energy': float(lmp.get_thermo('ke')),
            'total_energy': float(lmp.get_thermo('etotal')),
            'pressure': float(lmp.get_thermo('press')),
            'num_atoms': int(natoms),
            'positions': x.tolist(),
            'velocities': v.tolist(),
        }

    def initial_state(self):
        self._build_simulation()
        # Run 0 steps to initialize thermo
        self._lmp.command('run 0')
        self._first_run = False
        return self._read_state()

    def update(self, state, interval):
        self._build_simulation()

        dt = self.config['timestep']
        n_steps = max(1, int(round(interval / dt)))

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
