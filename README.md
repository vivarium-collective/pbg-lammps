# pbg-lammps

Process-bigraph wrapper for the [LAMMPS](https://www.lammps.org/) molecular dynamics simulator.

Wraps LAMMPS as a time-driven `Process` using the bridge pattern, enabling LAMMPS simulations to participate in process-bigraph composite models. The wrapper manages a LAMMPS instance internally, translating between process-bigraph state ports and LAMMPS thermodynamic/atomic data.

## Installation

```bash
# Clone and install in a virtual environment
git clone https://github.com/vivarium-collective/pbg-lammps.git
cd pbg-lammps
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# LAMMPS requires MPI:
uv pip install "lammps[mpi]"
```

## Quick Start

```python
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from pbg_lammps import LAMMPSProcess, make_lammps_document

core = allocate_core()
core.register_link('LAMMPSProcess', LAMMPSProcess)
core.register_link('ram-emitter', RAMEmitter)

# Create a Lennard-Jones liquid simulation
doc = make_lammps_document(
    num_atoms_per_dim=5,
    density=0.85,
    lattice_style='fcc',
    ensemble='nvt',
    target_temp=1.0,
    interval=1.0,
)

sim = Composite({'state': doc}, core=core)
sim.run(10.0)

# Access final state
print(sim.state['stores']['temperature'])
print(sim.state['stores']['total_energy'])

# Or collect time series via emitter
results = gather_emitter_results(sim)
for entry in results[('emitter',)]:
    print(f"t={entry['time']:.1f}  T={entry['temperature']:.3f}")
```

## API Reference

### LAMMPSProcess

| Port | Type | Direction | Description |
|------|------|-----------|-------------|
| `temperature` | `overwrite[float]` | output | Instantaneous temperature |
| `potential_energy` | `overwrite[float]` | output | Potential energy |
| `kinetic_energy` | `overwrite[float]` | output | Kinetic energy |
| `total_energy` | `overwrite[float]` | output | Total energy (PE + KE) |
| `pressure` | `overwrite[float]` | output | Hydrostatic pressure |
| `num_atoms` | `overwrite[integer]` | output | Number of atoms |
| `positions` | `overwrite[list]` | output | Atom positions [[x,y,z], ...] |
| `velocities` | `overwrite[list]` | output | Atom velocities [[vx,vy,vz], ...] |

### Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_atoms_per_dim` | int | 5 | Lattice repeats per dimension |
| `density` | float | 0.6 | Number density |
| `lattice_style` | string | "sc" | Lattice type (sc, fcc, bcc) |
| `temperature` | float | 1.0 | Initial temperature |
| `timestep` | float | 0.005 | Integration timestep |
| `pair_style` | string | "lj/cut" | LAMMPS pair style |
| `epsilon` | float | 1.0 | LJ energy parameter |
| `sigma` | float | 1.0 | LJ length parameter |
| `cutoff` | float | 2.5 | Pair cutoff distance |
| `mass` | float | 1.0 | Atom mass |
| `ensemble` | string | "nve" | Ensemble (nve, nvt, npt) |
| `target_temp` | float | 1.0 | Thermostat target T |
| `tdamp` | float | 0.5 | Thermostat damping |
| `target_press` | float | 0.0 | Barostat target P |
| `pdamp` | float | 5.0 | Barostat damping |
| `seed` | int | 87287 | RNG seed |
| `setup_commands` | string | "" | Raw LAMMPS script (overrides auto-generation) |

### Advanced: Custom LAMMPS Scripts

For complex setups (multi-type systems, custom potentials, fix deform, 2D), pass raw LAMMPS commands via `setup_commands`:

```python
doc = make_lammps_document(
    setup_commands="""
units lj
atom_style atomic
dimension 3
boundary p p p
lattice fcc 1.2
region box block 0 6 0 6 0 6
create_box 2 box
create_atoms 1 box
mass 1 1.0
mass 2 1.0
set type 1 type/fraction 2 0.2 12345
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
pair_coeff 1 2 1.5 0.8 2.0
pair_coeff 2 2 0.5 0.88 2.2
pair_modify shift yes
velocity all create 2.0 87287 dist gaussian
timestep 0.005
fix integ all nvt temp 2.0 0.4 1.0
""",
    timestep=0.005,
    interval=1.0,
)
```

## Architecture

The wrapper uses the **bridge pattern**: a single `LAMMPSProcess` owns a LAMMPS instance that is lazily initialized on the first `update()` call. Each update cycle follows Push-Run-Read:

1. **Push**: (future) inject external state into LAMMPS via `scatter_atoms()`
2. **Run**: advance LAMMPS by `interval / timestep` steps
3. **Read**: extract thermodynamic quantities via `get_thermo()` and atomic data via `numpy.extract_atom()`

All outputs use `overwrite` types since LAMMPS manages absolute state internally.

### Additional Outputs

| Port | Type | Description |
|------|------|-------------|
| `atom_types` | `overwrite[list]` | Per-atom type indices |
| `volume` | `overwrite[float]` | Simulation box volume |
| `pxx`, `pyy`, `pzz` | `overwrite[float]` | Stress tensor diagonals |
| `box_dimensions` | `overwrite[list]` | Box lengths [Lx, Ly, Lz] |

## Demo

```bash
source .venv/bin/activate
python demo/demo_report.py
```

Generates `demo/report.html` with four canonical simulation configurations:
- **Spinodal Decomposition** (2048 atoms): 50:50 binary LJ mixture demixing below the critical solution temperature. Domain formation via the same mechanism as biomolecular condensates. Colored by species.
- **Kremer-Grest Polymer Melt** (720 atoms): 36 chains of 20 beads with FENE bonds and WCA repulsion. The foundational coarse-grained polymer model, showing chain randomization from initial rod configurations.
- **Liquid-Vapor Slab Interface** (2688 atoms): Dense LJ slab in vacuum at T=0.85. Shows stable two-phase coexistence, surface tension measurable from pressure tensor anisotropy (P_N vs P_T).
- **Nanoparticle Sintering** (1042 atoms): Two spherical FCC clusters merging under surface energy minimization. Neck formation and coalescence relevant to powder metallurgy and nanoparticle synthesis.

Each section includes interactive 3D particle viewers (Three.js with InstancedMesh), context-specific Plotly charts (demixing energy, pressure tensor, neck growth), colored bigraph-viz architecture diagrams, and collapsible PBG document trees.
