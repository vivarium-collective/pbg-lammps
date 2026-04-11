"""Pre-built composite document factories for LAMMPS simulations."""


def make_lammps_document(
    num_atoms_per_dim=5,
    density=0.6,
    lattice_style='sc',
    temperature=1.0,
    timestep=0.005,
    pair_style='lj/cut',
    epsilon=1.0,
    sigma=1.0,
    cutoff=2.5,
    mass=1.0,
    ensemble='nve',
    target_temp=1.0,
    tdamp=0.5,
    target_press=0.0,
    pdamp=5.0,
    seed=87287,
    setup_commands='',
    interval=1.0,
):
    """Create a composite document for a LAMMPS molecular dynamics simulation.

    Returns a document dict ready for use with Composite().

    For simple single-type LJ simulations, use the individual parameters.
    For complex setups (multi-type, custom potentials, fix deform, 2D),
    pass a raw LAMMPS script via setup_commands.
    """
    config = {
        'num_atoms_per_dim': num_atoms_per_dim,
        'density': density,
        'lattice_style': lattice_style,
        'temperature': temperature,
        'timestep': timestep,
        'pair_style': pair_style,
        'epsilon': epsilon,
        'sigma': sigma,
        'cutoff': cutoff,
        'mass': mass,
        'ensemble': ensemble,
        'target_temp': target_temp,
        'tdamp': tdamp,
        'target_press': target_press,
        'pdamp': pdamp,
        'seed': seed,
        'setup_commands': setup_commands,
    }

    return {
        'lammps': {
            '_type': 'process',
            'address': 'local:LAMMPSProcess',
            'config': config,
            'interval': interval,
            'inputs': {},
            'outputs': {
                'temperature': ['stores', 'temperature'],
                'potential_energy': ['stores', 'potential_energy'],
                'kinetic_energy': ['stores', 'kinetic_energy'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
                'num_atoms': ['stores', 'num_atoms'],
                'positions': ['stores', 'positions'],
                'velocities': ['stores', 'velocities'],
                'atom_types': ['stores', 'atom_types'],
                'volume': ['stores', 'volume'],
                'pxx': ['stores', 'pxx'],
                'pyy': ['stores', 'pyy'],
                'pzz': ['stores', 'pzz'],
                'box_dimensions': ['stores', 'box_dimensions'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'temperature': 'float',
                    'potential_energy': 'float',
                    'kinetic_energy': 'float',
                    'total_energy': 'float',
                    'pressure': 'float',
                    'volume': 'float',
                    'time': 'float',
                },
            },
            'inputs': {
                'temperature': ['stores', 'temperature'],
                'potential_energy': ['stores', 'potential_energy'],
                'kinetic_energy': ['stores', 'kinetic_energy'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
                'volume': ['stores', 'volume'],
                'time': ['global_time'],
            },
        },
    }
