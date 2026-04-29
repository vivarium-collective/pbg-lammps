# pbg-lammps

Process-bigraph wrapper for the [LAMMPS](https://www.lammps.org/) molecular dynamics simulator.

**[View Interactive Demo Report](https://vivarium-collective.github.io/pbg-lammps/)** -- Lennard-Jones fluids, polymer chains, and thermal quench simulations with 3D viewers, Plotly charts, and bigraph architecture diagrams.

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

`LAMMPSProcess` is configured with a standard LAMMPS input script — the same
format LAMMPS itself reads (see [LAMMPS Commands](https://docs.lammps.org/Commands_input.html)).
You can either point it at a `.in` file on disk or pass the script inline.
Any `run` / `rerun` commands in the script are stripped at load time — the
process-bigraph orchestrator drives integration based on the requested
`interval`.

```python
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from pbg_lammps import LAMMPSProcess, make_lammps_document

core = allocate_core()
core.register_link('LAMMPSProcess', LAMMPSProcess)
core.register_link('ram-emitter', RAMEmitter)

# Option A: load from a .in file on disk
doc = make_lammps_document(
    input_file='path/to/simulation.in',
    interval=1.0,
)

# Option B: pass the script inline
doc = make_lammps_document(
    input_script="""
units lj
atom_style atomic
dimension 3
boundary p p p
lattice fcc 0.85
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_modify shift yes
velocity all create 1.0 87287 dist gaussian
timestep 0.005
fix integ all nvt temp 1.0 1.0 0.5
""",
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
| `input_file` | string | "" | Path to a LAMMPS `.in` input file |
| `input_script` | string | "" | Inline LAMMPS script (alternative to `input_file`) |
| `working_directory` | string | "" | Directory for resolving relative paths in commands like `read_data` (defaults to the directory of `input_file`) |

Exactly one of `input_file` or `input_script` must be provided.

### LAMMPS Input File Format

The wrapper accepts any standard LAMMPS input file. For example, the
following script (saved as `lj_nvt.in` and loaded via `input_file=`) sets
up a Lennard-Jones liquid in NVT:

```text
units           lj
atom_style      atomic
dimension       3
boundary        p p p

lattice         fcc 0.85
region          box block 0 5 0 5 0 5
create_box      1 box
create_atoms    1 box
mass            1 1.0

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0
pair_modify     shift yes

velocity        all create 1.0 87287 dist gaussian
timestep        0.005
fix             integ all nvt temp 1.0 1.0 0.5
```

Multi-type systems, custom potentials, FENE bonds, 2D simulations,
`read_data`-based setups, etc. are all supported — anything LAMMPS itself
accepts. Just leave out the `run` command (or include it; it will be
filtered out automatically).

## Architecture

The wrapper uses the **bridge pattern**: a single `LAMMPSProcess` owns a LAMMPS instance that is lazily initialized on the first `update()` call. On the first call, the input script is loaded with `commands_string()` (with `run` / `rerun` lines filtered out so the orchestrator drives integration). Each subsequent update cycle follows Run-Read:

1. **Run**: advance LAMMPS by `interval / timestep` steps
2. **Read**: extract thermodynamic quantities via `get_thermo()` and atomic data via `numpy.extract_atom()`

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
