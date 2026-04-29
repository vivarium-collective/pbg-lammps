"""Pre-built composite document factories for LAMMPS simulations."""


def make_lammps_document(
    input_file='',
    input_script='',
    working_directory='',
    interval=1.0,
):
    """Create a composite document for a LAMMPS molecular dynamics simulation.

    Provide either a path to a LAMMPS .in file (`input_file`) or an
    inline script (`input_script`). `run` / `rerun` commands in the
    script are stripped — the orchestrator drives integration based
    on `interval`.

    Returns a document dict ready for use with Composite().
    """
    if not input_file and not input_script:
        raise ValueError(
            'make_lammps_document requires input_file or input_script')

    config = {
        'input_file': input_file,
        'input_script': input_script,
        'working_directory': working_directory,
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
