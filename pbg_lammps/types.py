"""Custom bigraph-schema types for LAMMPS wrapper."""


def register_lammps_types(core):
    """Register custom types used by LAMMPS processes.

    Currently LAMMPS uses only built-in types (float, integer, list,
    overwrite). This hook exists for future extensions (e.g., typed
    force-field parameters or unit-bearing quantities).
    """
    pass
