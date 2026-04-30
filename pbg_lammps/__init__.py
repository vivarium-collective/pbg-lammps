"""Process-bigraph wrapper for LAMMPS molecular dynamics simulator."""

from pbg_lammps.processes import (
    LAMMPSProcess,
    PIDController,
    OscillatingForce,
)
from pbg_lammps.composites import (
    make_lammps_document,
    make_pid_controlled_document,
    make_force_driven_document,
)

__all__ = [
    'LAMMPSProcess',
    'PIDController',
    'OscillatingForce',
    'make_lammps_document',
    'make_pid_controlled_document',
    'make_force_driven_document',
]


def register_pbg_lammps(core):
    """Register all pbg_lammps processes on a process-bigraph core."""
    core.register_link('LAMMPSProcess', LAMMPSProcess)
    core.register_link('PIDController', PIDController)
    core.register_link('OscillatingForce', OscillatingForce)
    return core
