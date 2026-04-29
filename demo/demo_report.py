"""Demo: LAMMPS multi-configuration molecular dynamics report.

Runs four canonical MD simulations demonstrating LAMMPS's range:
  1. Spinodal decomposition — binary LJ demixing / condensate formation
  2. Kremer-Grest polymer melt — bead-spring chains with FENE bonds
  3. Liquid-vapor slab interface — two-phase coexistence
  4. Nanoparticle sintering — two clusters merging under surface tension

Each simulation is driven by a standard LAMMPS input (.in) file. The
input file is displayed in the report alongside the 3D viewer, charts,
bigraph diagram, and PBG composite document tree.
"""

import json
import os
import base64
import time as _time
import tempfile
import numpy as np
from process_bigraph import allocate_core
from pbg_lammps.processes import LAMMPSProcess
from pbg_lammps.composites import make_lammps_document


# ── Helper: polymer data file generation ──────────────────────────

def _polymer_data_text(n_chains, chain_len, box_size, bond_len=0.97):
    """Generate a LAMMPS data file with straight-rod polymer chains on a grid."""
    rng = np.random.RandomState(42)
    atoms = []
    bonds = []
    aid = 0
    n_per_dim = int(np.ceil(np.sqrt(n_chains)))
    spacing = box_size / n_per_dim
    chain_idx = 0

    for ix in range(n_per_dim):
        for iy in range(n_per_dim):
            if chain_idx >= n_chains:
                break
            x0 = (ix + 0.5) * spacing
            y0 = (iy + 0.5) * spacing
            z0 = (box_size - chain_len * bond_len) / 2
            for b in range(chain_len):
                aid += 1
                px = x0 + rng.uniform(-0.1, 0.1)
                py = y0 + rng.uniform(-0.1, 0.1)
                pz = z0 + b * bond_len
                atoms.append((aid, chain_idx + 1, 1, px, py, pz))
                if b > 0:
                    bonds.append((len(bonds) + 1, 1, aid - 1, aid))
            chain_idx += 1

    lines = ['LAMMPS polymer data\n']
    lines.append(f'\n{len(atoms)} atoms\n{len(bonds)} bonds\n')
    lines.append(f'\n1 atom types\n1 bond types\n')
    lines.append(f'\n0.0 {box_size} xlo xhi\n0.0 {box_size} ylo yhi\n')
    lines.append(f'0.0 {box_size} zlo zhi\n')
    lines.append(f'\nMasses\n\n1 1.0\n')
    lines.append(f'\nAtoms # bond\n\n')
    for a in atoms:
        lines.append(f'{a[0]} {a[1]} {a[2]} {a[3]:.6f} {a[4]:.6f} {a[5]:.6f}\n')
    lines.append(f'\nBonds\n\n')
    for b in bonds:
        lines.append(f'{b[0]} {b[1]} {b[2]} {b[3]}\n')

    return ''.join(lines), len(atoms)


# ── LAMMPS Input Files ─────────────────────────────────────────────

SPINODAL_IN = """\
# Spinodal decomposition: 50:50 binary LJ mixture quenched below T_c
# Same-species attractions are stronger than cross-species, driving
# spontaneous phase separation.

units           lj
atom_style      atomic
dimension       3
boundary        p p p

lattice         fcc 0.85
region          box block 0 8 0 8 0 8
create_box      2 box
create_atoms    1 box
mass            1 1.0
mass            2 1.0

# Randomly relabel half the atoms as type 2 (50:50 mixture)
set             type 1 type/fraction 2 0.5 48392

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5     # A-A attraction
pair_coeff      2 2 1.0 1.0 2.5     # B-B attraction
pair_coeff      1 2 0.5 1.0 2.5     # A-B weaker -> demixing
pair_modify     shift yes

velocity        all create 2.0 87287 dist gaussian
timestep        0.005

# NVT well below the consolute temperature
fix             integ all nvt temp 0.7 0.7 0.5
"""

POLYMER_IN_TEMPLATE = """\
# Kremer-Grest polymer melt: bead-spring chains with FENE bonds.
# 36 chains of 20 beads each; WCA repulsion + FENE bonds.

units           lj
atom_style      bond
dimension       3
boundary        p p p

read_data       {data_file}

# Purely repulsive Weeks-Chandler-Andersen potential
pair_style      lj/cut 1.122462
pair_coeff      1 1 1.0 1.0 1.122462
pair_modify     shift yes

# Finitely extensible nonlinear elastic bonds
bond_style      fene
bond_coeff      1 30.0 1.5 1.0 1.0
special_bonds   fene

velocity        all create 1.0 87287 dist gaussian
timestep        0.005

fix             integ all nvt temp 1.0 1.0 0.5
"""

SLAB_IN = """\
# Liquid-vapor slab: dense LJ liquid sandwiched between vacuum.
# Long cutoff captures surface tension via pressure tensor anisotropy.

units           lj
atom_style      atomic
dimension       3
boundary        p p p

lattice         fcc 0.84
region          box block 0 8 0 8 0 30
region          slab block 0 8 0 8 10 20
create_box      1 box
create_atoms    1 region slab
mass            1 1.0

pair_style      lj/cut 3.5
pair_coeff      1 1 1.0 1.0

velocity        all create 0.85 87287 dist gaussian
timestep        0.005

# Stable liquid-vapor coexistence at T=0.85 (well below T_c ~ 1.3)
fix             integ all nvt temp 0.85 0.85 0.5
"""

SINTER_IN = """\
# Nanoparticle sintering: two crystalline LJ clusters merging.
# Shrink-wrapped boundary; surface diffusion forms a neck over time.

units           lj
atom_style      atomic
dimension       3
boundary        s s s

lattice         fcc 1.0
region          box block -2 24 -2 24 -2 24
create_box      1 box

# Two spherical clusters separated by a small gap
region          sphere1 sphere 7 11 11 5 units box
region          sphere2 sphere 17 11 11 5 units box
create_atoms    1 region sphere1
create_atoms    1 region sphere2
mass            1 1.0

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0
pair_modify     shift yes

velocity        all create 0.4 87287 dist gaussian
timestep        0.005

fix             integ all nvt temp 0.4 0.4 0.5
"""


# ── Simulation Configs ──────────────────────────────────────────────

def _materialize_inputs(workdir):
    """Write all .in (and companion .data) files into workdir.

    Returns a list of config dicts with input file paths and the inline
    .in content for display in the report.
    """
    # Polymer needs a generated .data file alongside the .in
    poly_data_text, _ = _polymer_data_text(
        n_chains=36, chain_len=20, box_size=20.0)
    poly_data_path = os.path.join(workdir, 'polymer_melt.data')
    with open(poly_data_path, 'w') as f:
        f.write(poly_data_text)

    polymer_in = POLYMER_IN_TEMPLATE.format(data_file='polymer_melt.data')

    inputs = [
        ('spinodal', 'spinodal.in', SPINODAL_IN),
        ('polymer', 'polymer.in', polymer_in),
        ('slab', 'slab.in', SLAB_IN),
        ('sinter', 'sinter.in', SINTER_IN),
    ]
    paths = {}
    contents = {}
    for sid, fname, text in inputs:
        path = os.path.join(workdir, fname)
        with open(path, 'w') as f:
            f.write(text)
        paths[sid] = path
        contents[sid] = text
    return paths, contents


def _get_configs(input_paths, input_contents):
    """Build config list referencing on-disk .in files."""
    return [
        {
            'id': 'spinodal',
            'title': 'Spinodal Decomposition',
            'subtitle': 'Binary LJ mixture demixing into condensate-like domains',
            'description': (
                'A 50:50 binary Lennard-Jones mixture is quenched below its critical '
                'solution temperature. Same-species attractions (epsilon_AA = epsilon_BB = 1.0) '
                'are stronger than cross-species (epsilon_AB = 0.5), driving spontaneous '
                'phase separation via spinodal decomposition. Composition fluctuations '
                'amplify and coarsen into macroscopic A-rich and B-rich domains — the same '
                'mechanism underlying biomolecular condensate formation in cells. '
                'The domain growth follows the Lifshitz-Slyozov t^(1/3) scaling law.'
            ),
            'input_file': input_paths['spinodal'],
            'input_filename': 'spinodal.in',
            'input_content': input_contents['spinodal'],
            'n_snapshots': 40,
            'total_time': 80.0,
            'camera': [22, 16, 22],
            'color_scheme': 'indigo',
            'color_mode': 'type',
            'chart_set': 'spinodal',
        },
        {
            'id': 'polymer',
            'title': 'Kremer-Grest Polymer Melt',
            'subtitle': 'Bead-spring chains with FENE bonds — the canonical coarse-grained model',
            'description': (
                'The Kremer-Grest model is the foundation of computational polymer physics. '
                '36 chains of 20 beads each interact via a purely repulsive WCA potential '
                '(shifted LJ with cutoff at 2^(1/6) sigma) and are connected by finitely '
                'extensible nonlinear elastic (FENE) bonds. Starting from straight-rod '
                'configurations, the chains rapidly randomize into a disordered melt. '
                'This model captures universal polymer dynamics — Rouse relaxation, '
                'reptation, and entanglement — and is widely used to study polymer '
                'blends, gels, and biomolecular assemblies.'
            ),
            'input_file': input_paths['polymer'],
            'input_filename': 'polymer.in',
            'input_content': input_contents['polymer'],
            'n_snapshots': 40,
            'total_time': 100.0,
            'camera': [30, 22, 30],
            'color_scheme': 'emerald',
            'color_mode': 'speed',
            'chart_set': 'standard',
        },
        {
            'id': 'slab',
            'title': 'Liquid-Vapor Slab Interface',
            'subtitle': 'Two-phase coexistence with surface tension and capillary fluctuations',
            'description': (
                'A dense Lennard-Jones liquid slab is placed in the center of an '
                'elongated simulation box with vacuum above and below. At T=0.85 '
                '(well below the critical temperature T_c ~ 1.3), the system maintains '
                'stable liquid-vapor coexistence. The pair cutoff is extended to 3.5 sigma '
                'to capture long-range interactions that determine surface tension. '
                'The interfaces exhibit thermal capillary fluctuations, and occasional '
                'atoms evaporate into the vapor phase. This geometry is the standard '
                'method for computing surface tension via the pressure tensor anisotropy: '
                'gamma = L_z/2 * (P_N - P_T).'
            ),
            'input_file': input_paths['slab'],
            'input_filename': 'slab.in',
            'input_content': input_contents['slab'],
            'n_snapshots': 35,
            'total_time': 60.0,
            'camera': [22, 15, 50],
            'color_scheme': 'rose',
            'color_mode': 'speed',
            'chart_set': 'slab',
        },
        {
            'id': 'sinter',
            'title': 'Nanoparticle Sintering',
            'subtitle': 'Two crystalline clusters merge under surface energy minimization',
            'description': (
                'Two spherical Lennard-Jones nanoparticles (radius ~ 5 sigma, ~500 atoms '
                'each) are placed with a small gap between them. At moderate temperature '
                '(T=0.4), surface atoms diffuse across the gap, forming a neck that grows '
                'over time as the system minimizes its total surface energy. This sintering '
                'process is fundamental to powder metallurgy, nanoparticle synthesis, and '
                'additive manufacturing. The shrinking boundary box visible in the viewer '
                'is an artifact of the non-periodic boundaries (shrink-wrapped).'
            ),
            'input_file': input_paths['sinter'],
            'input_filename': 'sinter.in',
            'input_content': input_contents['sinter'],
            'n_snapshots': 35,
            'total_time': 80.0,
            'camera': [35, 25, 35],
            'color_scheme': 'amber',
            'color_mode': 'speed',
            'chart_set': 'sinter',
        },
    ]


def run_simulation(cfg_entry):
    """Run a single simulation, returning snapshots and wall-clock runtime."""
    core = allocate_core()
    core.register_link('LAMMPSProcess', LAMMPSProcess)

    t0 = _time.perf_counter()
    proc = LAMMPSProcess(
        config={'input_file': cfg_entry['input_file']},
        core=core)
    state0 = proc.initial_state()

    interval = cfg_entry['total_time'] / cfg_entry['n_snapshots']
    snapshots = [_snap(0.0, state0)]

    t = 0.0
    for i in range(cfg_entry['n_snapshots']):
        result = proc.update({}, interval=interval)
        t += interval
        snapshots.append(_snap(round(t, 4), result))

    runtime = _time.perf_counter() - t0
    proc.close()
    return snapshots, runtime


def _snap(t, s):
    return {
        'time': t,
        'positions': s['positions'],
        'velocities': s['velocities'],
        'atom_types': s['atom_types'],
        'temperature': s['temperature'],
        'potential_energy': s['potential_energy'],
        'kinetic_energy': s['kinetic_energy'],
        'total_energy': s['total_energy'],
        'pressure': s['pressure'],
        'num_atoms': s['num_atoms'],
        'volume': s['volume'],
        'pxx': s['pxx'],
        'pyy': s['pyy'],
        'pzz': s['pzz'],
        'box_dimensions': s['box_dimensions'],
    }


def generate_bigraph_image(cfg_entry):
    """Generate a colored bigraph-viz PNG for the composite document."""
    from bigraph_viz import plot_bigraph

    doc = {
        'lammps': {
            '_type': 'process',
            'address': 'local:LAMMPSProcess',
            'interval': cfg_entry['total_time'] / cfg_entry['n_snapshots'],
            'inputs': {},
            'outputs': {
                'temperature': ['stores', 'temperature'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
                'positions': ['stores', 'positions'],
                'volume': ['stores', 'volume'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {
                'temperature': 'float',
                'total_energy': 'float',
                'pressure': 'float',
                'volume': 'float',
                'time': 'float',
            }},
            'inputs': {
                'temperature': ['stores', 'temperature'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
                'volume': ['stores', 'volume'],
                'time': ['global_time'],
            },
        },
    }

    node_colors = {
        ('lammps',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def build_pbg_document(cfg_entry):
    """Build the PBG composite document dict for display.

    Uses the .in filename (not the temp path) so the displayed doc
    is portable.
    """
    doc = make_lammps_document(
        input_file=cfg_entry['input_filename'],
        interval=cfg_entry['total_time'] / cfg_entry['n_snapshots'],
    )
    return doc


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
    'amber': {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#d97706'},
}


def _escape_html(s):
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;'))


def generate_html(sim_results, output_path):
    """Generate comprehensive HTML report."""

    sections_html = []
    all_js_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        n_atoms = snapshots[0]['num_atoms']
        color_mode = cfg.get('color_mode', 'speed')
        chart_set = cfg.get('chart_set', 'standard')

        # Compute speed stats
        all_speeds = []
        for s in snapshots:
            vels = np.array(s['velocities'])
            speeds = np.linalg.norm(vels, axis=1)
            all_speeds.extend(speeds.tolist())
        speed_min = float(np.percentile(all_speeds, 2))
        speed_max = float(np.percentile(all_speeds, 98))

        # Count atom types
        type_counts = {}
        for t in snapshots[0]['atom_types']:
            type_counts[t] = type_counts.get(t, 0) + 1

        # Time series
        times = [s['time'] for s in snapshots]
        temps = [s['temperature'] for s in snapshots]
        pe = [s['potential_energy'] for s in snapshots]
        ke = [s['kinetic_energy'] for s in snapshots]
        etotal = [s['total_energy'] for s in snapshots]
        press = [s['pressure'] for s in snapshots]
        vols = [s['volume'] for s in snapshots]
        pn_vals = [s['pzz'] for s in snapshots]
        pt_vals = [0.5 * (s['pxx'] + s['pyy']) for s in snapshots]

        # JS data — positions + speeds + types per snapshot
        js_snapshots = []
        for s in snapshots:
            vels = np.array(s['velocities'])
            speeds = np.linalg.norm(vels, axis=1).tolist()
            js_snapshots.append({
                'time': s['time'],
                'positions': s['positions'],
                'speeds': speeds,
                'types': s['atom_types'],
                'box': s['box_dimensions'],
            })

        all_js_data[sid] = {
            'snapshots': js_snapshots,
            'speed_range': [speed_min, speed_max],
            'camera': cfg['camera'],
            'color_mode': color_mode,
            'charts': {
                'times': times, 'temperature': temps,
                'pe': pe, 'ke': ke, 'etotal': etotal,
                'pressure': press, 'volume': vols,
                'pn': pn_vals, 'pt': pt_vals,
            },
            'chart_set': chart_set,
        }

        # Bigraph PNG
        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        # Colorbar
        if color_mode == 'type':
            n1 = type_counts.get(1, 0)
            n2 = type_counts.get(2, 0)
            cb_html = (
                f'<div class="cb-title">Species</div>'
                f'<div class="cb-val">B ({n2})</div>'
                f'<div class="cb-gradient" style="background:linear-gradient(to bottom, #f43f5e, #6366f1);"></div>'
                f'<div class="cb-val">A ({n1})</div>'
            )
        else:
            cb_html = (
                f'<div class="cb-title">Speed |v|</div>'
                f'<div class="cb-val">{speed_max:.2f}</div>'
                f'<div class="cb-gradient" style="background:linear-gradient(to bottom, #e61a0d, #e6c01a, #4dd94d, #12b5c9, #3112cc);"></div>'
                f'<div class="cb-val">{speed_min:.2f}</div>'
            )

        type_str = ', '.join(f'type {k}: {v}' for k, v in sorted(type_counts.items()))

        # LAMMPS .in file (escaped, with line numbers)
        in_lines = cfg['input_content'].rstrip('\n').split('\n')
        numbered = '\n'.join(
            f'<span class="in-line"><span class="in-ln">{i+1:>3}</span> {_escape_html(line)}</span>'
            for i, line in enumerate(in_lines))

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Atoms</span><span class="metric-value">{n_atoms:,}</span></div>
        <div class="metric"><span class="metric-label">T (final)</span><span class="metric-value">{temps[-1]:.3f}</span></div>
        <div class="metric"><span class="metric-label">PE/atom</span><span class="metric-value">{pe[-1]/n_atoms:.3f}</span></div>
        <div class="metric"><span class="metric-label">P (final)</span><span class="metric-value">{press[-1]:.1f}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">LAMMPS Input File &middot; <code class="in-fname">{cfg['input_filename']}</code></h3>
      <div class="in-file-wrap"><pre class="in-file">{numbered}</pre></div>

      <h3 class="subsection-title">3D Particle Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="particle-canvas"></canvas>
        <div class="viewer-info">
          <strong>{n_atoms}</strong> atoms &middot; {type_str}<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="colorbar-box">{cb_html}</div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(snapshots)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Thermodynamics</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-a-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-b-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-c-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-d-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append(section)

    # Navigation
    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LAMMPS Molecular Dynamics Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:700px; }}
.nav {{ display:flex; gap:.6rem; padding:.8rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; flex-wrap:wrap; }}
.nav-link {{ padding:.4rem .8rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.8rem; font-weight:600; transition:all .15s; white-space:nowrap; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem; padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:800px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155; margin:1.5rem 0 .8rem; }}
.in-fname {{ font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;
             font-size:.85rem; background:#eef2ff; color:#4338ca; padding:.1rem .45rem;
             border-radius:5px; font-weight:500; }}
.in-file-wrap {{ background:#0f172a; border:1px solid #334155; border-radius:10px;
                 overflow:auto; max-height:420px; margin-bottom:1rem; }}
.in-file {{ font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;
            font-size:.78rem; line-height:1.55; color:#cbd5e1; padding:.9rem 1rem;
            white-space:pre; }}
.in-line {{ display:block; }}
.in-ln {{ display:inline-block; width:2.5em; color:#475569; user-select:none;
          margin-right:.7rem; text-align:right; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.15rem; font-weight:700; color:#1e293b; }}
.viewer-wrap {{ position:relative; background:#0f172a; border:1px solid #334155; border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.particle-canvas {{ width:100%; height:520px; display:block; cursor:grab; }}
.particle-canvas:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(15,23,42,.85);
                border:1px solid #334155; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#94a3b8; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#e2e8f0; }}
.colorbar-box {{ position:absolute; top:.8rem; right:.8rem; background:rgba(15,23,42,.85);
                 border:1px solid #334155; border-radius:8px; padding:.6rem;
                 display:flex; flex-direction:column; align-items:center; gap:.2rem; backdrop-filter:blur(4px); }}
.cb-title {{ font-size:.65rem; text-transform:uppercase; letter-spacing:.04em; color:#94a3b8; }}
.cb-gradient {{ width:16px; height:100px; border-radius:3px; }}
.cb-val {{ font-size:.65rem; color:#64748b; text-align:center; max-width:80px; }}
.slider-controls {{ position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(15,23,42,.95));
                    padding:1.5rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }}
.slider-controls label {{ font-size:.8rem; color:#94a3b8; }}
.time-slider {{ flex:1; height:5px; }}
.time-val {{ font-size:.95rem; font-weight:600; color:#e2e8f0; min-width:100px; text-align:right; }}
.play-btn {{ background:rgba(15,23,42,.6); border:1.5px solid; padding:.3rem .8rem; border-radius:7px;
             cursor:pointer; font-size:.8rem; font-weight:600; transition:all .15s; }}
.play-btn:hover {{ transform:scale(1.05); }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:280px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px; padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem; border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>LAMMPS Molecular Dynamics Report</h1>
  <p>Four canonical molecular dynamics simulations wrapped as <strong>process-bigraph</strong>
  Processes using the LAMMPS engine. Each simulation is driven by a standard
  LAMMPS <code>.in</code> input file (shown inline below).</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-lammps</strong> &mdash;
  LAMMPS + process-bigraph &mdash;
  Classical Molecular Dynamics
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;').replace(/\\n/g,'\\\\n') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Color Utilities ───
const TYPE_COLORS = [
  new THREE.Color(0.39, 0.40, 0.95),
  new THREE.Color(0.95, 0.25, 0.37),
  new THREE.Color(0.06, 0.73, 0.51),
  new THREE.Color(0.96, 0.62, 0.04),
];

function speedToColor(t) {{
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.25) {{ const s=t/0.25; r=0.19; g=0.07+0.63*s; b=0.99-0.19*s; }}
  else if (t < 0.5) {{ const s=(t-0.25)/0.25; r=0.19+0.11*s; g=0.70+0.15*s; b=0.80-0.55*s; }}
  else if (t < 0.75) {{ const s=(t-0.5)/0.25; r=0.30+0.60*s; g=0.85-0.10*s; b=0.25-0.15*s; }}
  else {{ const s=(t-0.75)/0.25; r=0.90+0.10*s; g=0.75-0.55*s; b=0.10-0.05*s; }}
  return new THREE.Color(r, g, b);
}}

// ─── Three.js Particle Viewers ───
const viewers = {{}};
const playStates = {{}};

function initViewer(sid) {{
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 520;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0x0f172a);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.1, 500);
  cam.position.set(...d.camera);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;

  const snap0 = d.snapshots[0];
  const nAtoms = snap0.positions.length;

  // Center camera on atom cloud
  let cx=0, cy=0, cz=0;
  for (let i = 0; i < nAtoms; i++) {{
    cx += snap0.positions[i][0]; cy += snap0.positions[i][1]; cz += snap0.positions[i][2];
  }}
  cx /= nAtoms; cy /= nAtoms; cz /= nAtoms;
  controls.target.set(cx, cy, cz);

  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dl1.position.set(30, 50, 40); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0x8b9cc7, 0.3);
  dl2.position.set(-20, -10, -30); scene.add(dl2);

  const sphereGeo = new THREE.SphereGeometry(0.32, 10, 6);
  const sphereMat = new THREE.MeshPhongMaterial({{ shininess: 60, specular: 0x444444 }});
  const mesh = new THREE.InstancedMesh(sphereGeo, sphereMat, nAtoms);
  scene.add(mesh);

  const dummy = new THREE.Object3D();

  function updateParticles(idx) {{
    const snap = d.snapshots[idx];
    const [smin, smax] = d.speed_range;
    for (let i = 0; i < nAtoms; i++) {{
      dummy.position.set(snap.positions[i][0], snap.positions[i][1], snap.positions[i][2]);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      let col;
      if (d.color_mode === 'type') {{
        col = TYPE_COLORS[Math.min((snap.types[i]||1)-1, TYPE_COLORS.length-1)];
      }} else {{
        col = speedToColor((snap.speeds[i]-smin)/(smax-smin+1e-12));
      }}
      mesh.setColorAt(i, col);
    }}
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
  }}
  updateParticles(0);

  // Bounding box
  const boxGeo = new THREE.BoxGeometry(1,1,1);
  const boxEdges = new THREE.EdgesGeometry(boxGeo);
  const boxLine = new THREE.LineSegments(boxEdges,
    new THREE.LineBasicMaterial({{color:0x475569, transparent:true, opacity:0.4}}));
  scene.add(boxLine);

  function updateBox(idx) {{
    const bx=d.snapshots[idx].box[0], by=d.snapshots[idx].box[1], bz=d.snapshots[idx].box[2];
    boxLine.scale.set(bx, by, bz);
    boxLine.position.set(bx/2, by/2, bz/2);
  }}
  updateBox(0);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    updateParticles(idx); updateBox(idx);
    tval.textContent = 't = ' + d.snapshots[idx].time.toFixed(1);
  }});

  viewers[sid] = {{ renderer, scene, cam, controls, updateParticles, updateBox, slider, tval }};
  playStates[sid] = {{ playing: false, interval: null }};

  (function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }})();
}}

function togglePlay(sid) {{
  const ps = playStates[sid], v = viewers[sid], d = DATA[sid];
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {{
    btn.textContent = 'Pause'; v.controls.autoRotate = false;
    ps.interval = setInterval(() => {{
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= d.snapshots.length) idx = 0;
      v.slider.value = idx;
      v.updateParticles(idx); v.updateBox(idx);
      v.tval.textContent = 't = ' + d.snapshots[idx].time.toFixed(1);
    }}, 200);
  }} else {{
    btn.textContent = 'Play'; v.controls.autoRotate = true;
    clearInterval(ps.interval);
  }}
}}

Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly Charts ───
const pL = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:15, t:35, b:40 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0',
           title:{{ text:'Time (LJ units)', font:{{ size:10 }} }} }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pC = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const c = DATA[sid].charts, cs = DATA[sid].chart_set;

  if (cs === 'spinodal') {{
    Plotly.newPlot('chart-a-'+sid, [{{
      x:c.times, y:c.pe, type:'scatter', mode:'lines', line:{{ color:'#6366f1', width:2.5 }},
    }}], {{...pL, title:{{ text:'Potential Energy (demixing)', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'PE', font:{{ size:10 }} }} }}
    }}, pC);
    Plotly.newPlot('chart-b-'+sid, [
      {{ x:c.times, y:c.pe, type:'scatter', mode:'lines', line:{{ color:'#6366f1', width:1.5 }}, name:'PE' }},
      {{ x:c.times, y:c.ke, type:'scatter', mode:'lines', line:{{ color:'#f43f5e', width:1.5 }}, name:'KE' }},
      {{ x:c.times, y:c.etotal, type:'scatter', mode:'lines', line:{{ color:'#1e293b', width:2, dash:'dot' }}, name:'Total' }},
    ], {{...pL, title:{{ text:'Energy Components', font:{{ size:12, color:'#334155' }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
    }}, pC);
    Plotly.newPlot('chart-c-'+sid, [{{
      x:c.times, y:c.temperature, type:'scatter', mode:'lines', line:{{ color:'#10b981', width:2 }},
      fill:'tozeroy', fillcolor:'rgba(16,185,129,0.06)',
    }}], {{...pL, title:{{ text:'Temperature', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'T', font:{{ size:10 }} }} }}
    }}, pC);
    Plotly.newPlot('chart-d-'+sid, [{{
      x:c.times, y:c.pressure, type:'scatter', mode:'lines', line:{{ color:'#f59e0b', width:2 }},
    }}], {{...pL, title:{{ text:'Pressure', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'P', font:{{ size:10 }} }} }}
    }}, pC);

  }} else if (cs === 'slab') {{
    // Slab: PE, Energy components, Pressure tensor anisotropy, Temperature
    Plotly.newPlot('chart-a-'+sid, [{{
      x:c.times, y:c.pe, type:'scatter', mode:'lines', line:{{ color:'#f43f5e', width:2.5 }},
    }}], {{...pL, title:{{ text:'Potential Energy', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'PE', font:{{ size:10 }} }} }}
    }}, pC);
    Plotly.newPlot('chart-b-'+sid, [
      {{ x:c.times, y:c.pn, type:'scatter', mode:'lines', line:{{ color:'#6366f1', width:1.5 }}, name:'P_N (Pzz)' }},
      {{ x:c.times, y:c.pt, type:'scatter', mode:'lines', line:{{ color:'#f43f5e', width:1.5 }}, name:'P_T (Pxx+Pyy)/2' }},
    ], {{...pL, title:{{ text:'Pressure Tensor (surface tension)', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'P component', font:{{ size:10 }} }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
    }}, pC);
    Plotly.newPlot('chart-c-'+sid, [{{
      x:c.times, y:c.temperature, type:'scatter', mode:'lines', line:{{ color:'#10b981', width:2 }},
      fill:'tozeroy', fillcolor:'rgba(16,185,129,0.06)',
    }}], {{...pL, title:{{ text:'Temperature', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'T', font:{{ size:10 }} }} }}
    }}, pC);
    Plotly.newPlot('chart-d-'+sid, [{{
      x:c.times, y:c.pressure, type:'scatter', mode:'lines', line:{{ color:'#f59e0b', width:2 }},
    }}], {{...pL, title:{{ text:'Bulk Pressure', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'P', font:{{ size:10 }} }} }}
    }}, pC);

  }} else if (cs === 'sinter') {{
    // Sintering: PE (neck growth), Temperature, Energy components, Pressure
    Plotly.newPlot('chart-a-'+sid, [{{
      x:c.times, y:c.pe, type:'scatter', mode:'lines', line:{{ color:'#f59e0b', width:2.5 }},
      fill:'tozeroy', fillcolor:'rgba(245,158,11,0.06)',
    }}], {{...pL, title:{{ text:'Potential Energy (neck growth)', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'PE', font:{{ size:10 }} }} }}
    }}, pC);
    Plotly.newPlot('chart-b-'+sid, [
      {{ x:c.times, y:c.pe, type:'scatter', mode:'lines', line:{{ color:'#f59e0b', width:1.5 }}, name:'PE' }},
      {{ x:c.times, y:c.ke, type:'scatter', mode:'lines', line:{{ color:'#6366f1', width:1.5 }}, name:'KE' }},
    ], {{...pL, title:{{ text:'Energy Components', font:{{ size:12, color:'#334155' }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
    }}, pC);
    Plotly.newPlot('chart-c-'+sid, [{{
      x:c.times, y:c.temperature, type:'scatter', mode:'lines', line:{{ color:'#10b981', width:2 }},
    }}], {{...pL, title:{{ text:'Temperature', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'T', font:{{ size:10 }} }} }}
    }}, pC);
    Plotly.newPlot('chart-d-'+sid, [{{
      x:c.times, y:c.volume, type:'scatter', mode:'lines', line:{{ color:'#8b5cf6', width:2 }},
    }}], {{...pL, title:{{ text:'Bounding Volume', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'V', font:{{ size:10 }} }} }}
    }}, pC);

  }} else {{
    // Standard: Total energy, Components, Temperature, Pressure
    Plotly.newPlot('chart-a-'+sid, [{{
      x:c.times, y:c.etotal, type:'scatter', mode:'lines', line:{{ color:'#10b981', width:2.5 }},
    }}], {{...pL, title:{{ text:'Total Energy', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }}
    }}, pC);
    Plotly.newPlot('chart-b-'+sid, [
      {{ x:c.times, y:c.pe, type:'scatter', mode:'lines', line:{{ color:'#6366f1', width:1.5 }}, name:'PE' }},
      {{ x:c.times, y:c.ke, type:'scatter', mode:'lines', line:{{ color:'#f43f5e', width:1.5 }}, name:'KE' }},
    ], {{...pL, title:{{ text:'Energy Components', font:{{ size:12, color:'#334155' }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
    }}, pC);
    Plotly.newPlot('chart-c-'+sid, [{{
      x:c.times, y:c.temperature, type:'scatter', mode:'lines', line:{{ color:'#10b981', width:2 }},
      fill:'tozeroy', fillcolor:'rgba(16,185,129,0.06)',
    }}], {{...pL, title:{{ text:'Temperature', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'T', font:{{ size:10 }} }} }}
    }}, pC);
    Plotly.newPlot('chart-d-'+sid, [{{
      x:c.times, y:c.pressure, type:'scatter', mode:'lines', line:{{ color:'#f59e0b', width:2 }},
    }}], {{...pL, title:{{ text:'Pressure', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pL.yaxis, title:{{ text:'P', font:{{ size:10 }} }} }}
    }}, pC);
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


def run_demo():
    import shutil
    import subprocess
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    workdir = tempfile.mkdtemp(prefix='pbg_lammps_demo_')
    print(f'LAMMPS input files written to: {workdir}')

    try:
        input_paths, input_contents = _materialize_inputs(workdir)
        configs = _get_configs(input_paths, input_contents)

        sim_results = []
        for cfg in configs:
            print(f'Running: {cfg["title"]}...')
            snapshots, runtime = run_simulation(cfg)
            sim_results.append((cfg, (snapshots, runtime)))
            print(f'  Runtime: {runtime:.2f}s')
            print(f'  {len(snapshots)} snapshots, {snapshots[0]["num_atoms"]} atoms')

        print('Generating HTML report...')
        generate_html(sim_results, output_path)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
