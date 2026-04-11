"""Demo: LAMMPS multi-configuration molecular dynamics report.

Runs four canonical MD simulations demonstrating LAMMPS's range:
  1. Kob-Andersen binary glass former (NVT quench, multi-type)
  2. Uniaxial tensile deformation of FCC crystal (fix deform, mechanics)
  3. 2D hexatic melting (2D physics, phase transition)
  4. Isobaric compression (NPT, equation of state)

Generates interactive 3D/2D particle viewers with Three.js,
Plotly charts, bigraph-viz architecture diagrams, and navigatable
PBG document trees — all in a single self-contained HTML.
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


# ── Simulation Configs ──────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'glass',
        'title': 'Kob-Andersen Binary Glass Former',
        'subtitle': 'Two-component LJ mixture quenched through the glass transition',
        'description': (
            'The Kob-Andersen model is the canonical glass-forming system in '
            'computational physics. An 80:20 binary mixture of large (A) and '
            'small (B) Lennard-Jones particles with non-additive cross-interactions '
            '(epsilon_AB = 1.5) frustrates crystallization. Starting from a '
            'high-temperature liquid (T=2.0), the system is quenched to T=0.4 — '
            'below the mode-coupling temperature T_c ~ 0.435. The dynamics slow '
            'dramatically as the system falls out of equilibrium into a glassy state, '
            'visible as a plateau in the potential energy.'
        ),
        'config': {
            'setup_commands': (
                "units lj\n"
                "atom_style atomic\n"
                "dimension 3\n"
                "boundary p p p\n"
                "lattice fcc 1.2\n"
                "region box block 0 6 0 6 0 6\n"
                "create_box 2 box\n"
                "create_atoms 1 box\n"
                "mass 1 1.0\n"
                "mass 2 1.0\n"
                "set type 1 type/fraction 2 0.2 48392\n"
                "pair_style lj/cut 2.5\n"
                "pair_coeff 1 1 1.0 1.0 2.5\n"
                "pair_coeff 1 2 1.5 0.8 2.0\n"
                "pair_coeff 2 2 0.5 0.88 2.2\n"
                "pair_modify shift yes\n"
                "velocity all create 2.0 87287 dist gaussian\n"
                "timestep 0.005\n"
                "fix integ all nvt temp 2.0 0.4 1.0\n"
            ),
            'timestep': 0.005,
        },
        'n_snapshots': 40,
        'total_time': 50.0,
        'camera': [20, 15, 20],
        'color_scheme': 'indigo',
        'color_mode': 'type',
        'is_2d': False,
        'chart_set': 'glass',
    },
    {
        'id': 'tension',
        'title': 'Uniaxial Tensile Deformation',
        'subtitle': 'FCC crystal pulled apart to reveal yield and fracture',
        'description': (
            'A Lennard-Jones FCC crystal (elongated 6x6x12 unit cells, 1728 atoms) '
            'is equilibrated at low temperature (T=0.01) then subjected to uniaxial '
            'tensile strain in the z-direction at a constant engineering strain rate. '
            'The simulation captures the full mechanical response: linear elastic '
            'regime, yield point, plastic flow with dislocation activity, and '
            'ultimately void nucleation and fracture. The stress-strain curve and '
            'per-atom kinetic energy reveal where deformation localizes.'
        ),
        'config': {
            'setup_commands': (
                "units lj\n"
                "atom_style atomic\n"
                "dimension 3\n"
                "boundary p p p\n"
                "lattice fcc 1.0\n"
                "region box block 0 6 0 6 0 12\n"
                "create_box 1 box\n"
                "create_atoms 1 box\n"
                "mass 1 1.0\n"
                "pair_style lj/cut 2.5\n"
                "pair_coeff 1 1 1.0 1.0\n"
                "pair_modify shift yes\n"
                "velocity all create 0.01 54321 dist gaussian\n"
                "timestep 0.002\n"
                "fix integ all nvt temp 0.01 0.01 0.5\n"
            ),
            'timestep': 0.002,
        },
        'equilibrate_time': 2.0,
        'deform_command': 'fix pull all deform 1 z erate 0.01',
        'n_snapshots': 40,
        'total_time': 40.0,
        'camera': [20, 15, 35],
        'color_scheme': 'rose',
        'color_mode': 'speed',
        'is_2d': False,
        'chart_set': 'tension',
    },
    {
        'id': 'hexatic',
        'title': '2D Hexatic Melting',
        'subtitle': 'Kosterlitz-Thouless-Halperin-Nelson-Young transition in 2D',
        'description': (
            'A 2D system of Lennard-Jones particles on a hexagonal lattice is '
            'heated from T=0.1 through the KTHNY melting transition. In 2D, melting '
            'proceeds via two continuous transitions: solid -> hexatic -> isotropic '
            'liquid, mediated by unbinding of topological defects (dislocations and '
            'disclinations). Watch the ordered hexagonal lattice progressively '
            'disorder as temperature rises, with the hexatic phase characterized by '
            'orientational order without translational order.'
        ),
        'config': {
            'setup_commands': (
                "units lj\n"
                "atom_style atomic\n"
                "dimension 2\n"
                "boundary p p p\n"
                "lattice hex 0.88\n"
                "region box block 0 25 0 25 -0.5 0.5\n"
                "create_box 1 box\n"
                "create_atoms 1 box\n"
                "mass 1 1.0\n"
                "pair_style lj/cut 2.5\n"
                "pair_coeff 1 1 1.0 1.0\n"
                "pair_modify shift yes\n"
                "velocity all create 0.1 12345 dist gaussian\n"
                "timestep 0.005\n"
                "fix integ all nvt temp 0.1 1.2 1.0\n"
                "fix enforce all enforce2d\n"
            ),
            'timestep': 0.005,
        },
        'n_snapshots': 40,
        'total_time': 60.0,
        'camera': [12.5, 12.5, 35],
        'camera_target': [12.5, 12.5, 0],
        'color_scheme': 'emerald',
        'color_mode': 'speed',
        'is_2d': True,
        'chart_set': 'standard',
    },
    {
        'id': 'compress',
        'title': 'Isobaric Compression',
        'subtitle': 'NPT equation of state from gas to dense liquid',
        'description': (
            'An FCC Lennard-Jones system at moderate temperature (T=1.5) is '
            'progressively compressed from near-zero pressure to P=25 using the '
            'NPT ensemble with a Nose-Hoover barostat. The simulation box shrinks '
            'isotropically as the system traverses the equation of state from a '
            'low-density gas/expanded liquid through the coexistence region to a '
            'dense compressed liquid. The volume and density evolve continuously, '
            'and the particle viewer shows the box contracting in real time.'
        ),
        'config': {
            'setup_commands': (
                "units lj\n"
                "atom_style atomic\n"
                "dimension 3\n"
                "boundary p p p\n"
                "lattice fcc 0.55\n"
                "region box block 0 7 0 7 0 7\n"
                "create_box 1 box\n"
                "create_atoms 1 box\n"
                "mass 1 1.0\n"
                "pair_style lj/cut 2.5\n"
                "pair_coeff 1 1 1.0 1.0\n"
                "pair_modify shift yes\n"
                "velocity all create 1.5 99999 dist gaussian\n"
                "timestep 0.005\n"
                "fix integ all npt temp 1.5 1.5 0.5 iso 0.0 25.0 5.0\n"
            ),
            'timestep': 0.005,
        },
        'n_snapshots': 40,
        'total_time': 50.0,
        'camera': [25, 18, 25],
        'color_scheme': 'amber',
        'color_mode': 'speed',
        'is_2d': False,
        'chart_set': 'npt',
    },
]


def run_simulation(cfg_entry):
    """Run a single simulation, returning snapshots and wall-clock runtime."""
    core = allocate_core()
    core.register_link('LAMMPSProcess', LAMMPSProcess)

    t0 = _time.perf_counter()
    proc = LAMMPSProcess(config=cfg_entry['config'], core=core)
    state0 = proc.initial_state()

    # Handle tension: equilibrate first, then add deformation
    if 'equilibrate_time' in cfg_entry:
        eq_time = cfg_entry['equilibrate_time']
        proc.update({}, interval=eq_time)
        proc._lmp.command(cfg_entry['deform_command'])

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
    """Build the PBG composite document dict for display."""
    cfg = cfg_entry['config']
    doc = make_lammps_document(
        setup_commands=cfg.get('setup_commands', ''),
        timestep=cfg.get('timestep', 0.005),
        interval=cfg_entry['total_time'] / cfg_entry['n_snapshots'],
    )
    return doc


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
    'amber': {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#d97706',
              'bg': '#fffbeb', 'accent': '#fbbf24', 'text': '#78350f'},
}


def generate_html(sim_results, output_path):
    """Generate comprehensive HTML report."""

    sections_html = []
    all_js_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        n_atoms = snapshots[0]['num_atoms']
        is_2d = cfg.get('is_2d', False)
        color_mode = cfg.get('color_mode', 'speed')
        chart_set = cfg.get('chart_set', 'standard')

        # Compute speed stats for colorbar
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
        pzz_vals = [s['pzz'] for s in snapshots]
        lz_vals = [s['box_dimensions'][2] for s in snapshots]

        # Strain for tension
        lz0 = lz_vals[0]
        strains = [(lz - lz0) / lz0 * 100 for lz in lz_vals]

        # Density from volume
        densities = [n_atoms / v if v > 0 else 0 for v in vols]

        # JS data
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
            'camera_target': cfg.get('camera_target', None),
            'color_mode': color_mode,
            'is_2d': is_2d,
            'charts': {
                'times': times, 'temperature': temps,
                'pe': pe, 'ke': ke, 'etotal': etotal,
                'pressure': press, 'volume': vols,
                'pzz': pzz_vals, 'strain': strains,
                'density': densities,
            },
            'chart_set': chart_set,
        }

        # Bigraph PNG
        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        # PBG document JSON
        pbg_doc = build_pbg_document(cfg)

        # Metrics
        t0_val, t1_val = temps[0], temps[-1]
        e0_val, e1_val = etotal[0], etotal[-1]
        type_str = ', '.join(f'type {k}: {v}' for k, v in sorted(type_counts.items()))

        # Colorbar info
        if color_mode == 'type':
            cb_title = 'Atom Type'
            cb_top = f'B (n={type_counts.get(2, 0)})'
            cb_bot = f'A (n={type_counts.get(1, 0)})'
            cb_gradient = 'linear-gradient(to bottom, #f43f5e, #6366f1)'
        else:
            cb_title = 'Speed |v|'
            cb_top = f'{speed_max:.2f}'
            cb_bot = f'{speed_min:.2f}'
            cb_gradient = 'linear-gradient(to bottom, #e61a0d, #e6c01a, #4dd94d, #12b5c9, #3112cc)'

        # Per-config metrics row
        dim_label = '2D' if is_2d else '3D'
        metrics = f"""
        <div class="metric"><span class="metric-label">Atoms</span><span class="metric-value">{n_atoms:,}</span></div>
        <div class="metric"><span class="metric-label">Dimension</span><span class="metric-value">{dim_label}</span></div>
        <div class="metric"><span class="metric-label">T (init &rarr; final)</span><span class="metric-value">{t0_val:.2f} &rarr; {t1_val:.2f}</span></div>
        <div class="metric"><span class="metric-label">PE (final)</span><span class="metric-value">{pe[-1]:.2f}</span></div>
        <div class="metric"><span class="metric-label">P (final)</span><span class="metric-value">{press[-1]:.1f}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
"""

        canvas_class = 'particle-canvas-2d' if is_2d else 'particle-canvas'
        viewer_hint = 'Drag to pan &middot; Scroll to zoom' if is_2d else 'Drag to rotate &middot; Scroll to zoom'

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

      <div class="metrics-row">{metrics}</div>

      <h3 class="subsection-title">{'2D' if is_2d else '3D'} Particle Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="{canvas_class}"></canvas>
        <div class="viewer-info">
          <strong>{n_atoms}</strong> atoms &middot; {type_str}<br>
          {viewer_hint}
        </div>
        <div class="colorbar-box">
          <div class="cb-title">{cb_title}</div>
          <div class="cb-val">{cb_top}</div>
          <div class="cb-gradient" style="background:{cb_gradient};"></div>
          <div class="cb-val">{cb_bot}</div>
        </div>
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

    # PBG docs for JSON viewer
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
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100;
        flex-wrap:wrap; }}
.nav-link {{ padding:.4rem .8rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.8rem; font-weight:600;
             transition:all .15s; white-space:nowrap; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:800px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.15rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.7rem; color:#94a3b8; }}
.viewer-wrap {{ position:relative; background:#0f172a; border:1px solid #334155;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.particle-canvas {{ width:100%; height:520px; display:block; cursor:grab; }}
.particle-canvas:active {{ cursor:grabbing; }}
.particle-canvas-2d {{ width:100%; height:520px; display:block; cursor:grab; }}
.particle-canvas-2d:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(15,23,42,.85);
                border:1px solid #334155; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#94a3b8; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#e2e8f0; }}
.colorbar-box {{ position:absolute; top:.8rem; right:.8rem; background:rgba(15,23,42,.85);
                 border:1px solid #334155; border-radius:8px; padding:.6rem;
                 display:flex; flex-direction:column; align-items:center; gap:.2rem;
                 backdrop-filter:blur(4px); }}
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
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
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
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
  .nav {{ padding:.6rem 1rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>LAMMPS Molecular Dynamics Report</h1>
  <p>Four canonical molecular dynamics simulations wrapped as <strong>process-bigraph</strong>
  Processes, demonstrating multi-component systems, mechanical deformation, 2D phase
  transitions, and equation-of-state compression.</p>
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
  new THREE.Color(0.39, 0.40, 0.95),   // type 1: indigo
  new THREE.Color(0.95, 0.25, 0.37),   // type 2: rose
  new THREE.Color(0.06, 0.73, 0.51),   // type 3: emerald
  new THREE.Color(0.96, 0.62, 0.04),   // type 4: amber
];

function speedToColor(t) {{
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.25) {{
    const s = t / 0.25;
    r = 0.19; g = 0.07 + 0.63*s; b = 0.99 - 0.19*s;
  }} else if (t < 0.5) {{
    const s = (t - 0.25) / 0.25;
    r = 0.19 + 0.11*s; g = 0.70 + 0.15*s; b = 0.80 - 0.55*s;
  }} else if (t < 0.75) {{
    const s = (t - 0.5) / 0.25;
    r = 0.30 + 0.60*s; g = 0.85 - 0.10*s; b = 0.25 - 0.15*s;
  }} else {{
    const s = (t - 0.75) / 0.25;
    r = 0.90 + 0.10*s; g = 0.75 - 0.55*s; b = 0.10 - 0.05*s;
  }}
  return new THREE.Color(r, g, b);
}}

// ─── Three.js Particle Viewers ───
const viewers = {{}};
const playStates = {{}};

function initViewer(sid) {{
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = d.is_2d ? 520 : 520;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0x0f172a);

  const scene = new THREE.Scene();

  let cam;
  if (d.is_2d) {{
    const aspect = W / H;
    const frustum = 16;
    cam = new THREE.OrthographicCamera(-frustum*aspect, frustum*aspect, frustum, -frustum, 0.1, 500);
    cam.position.set(d.camera[0], d.camera[1], d.camera[2]);
    cam.up.set(0, 1, 0);
  }} else {{
    cam = new THREE.PerspectiveCamera(45, W/H, 0.1, 500);
    cam.position.set(...d.camera);
  }}

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  if (d.is_2d) {{
    controls.enableRotate = false;
    controls.autoRotate = false;
  }} else {{
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.6;
  }}

  if (d.camera_target) {{
    controls.target.set(d.camera_target[0], d.camera_target[1], d.camera_target[2]);
  }}

  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dl1.position.set(30, 50, 40); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0x8b9cc7, 0.3);
  dl2.position.set(-20, -10, -30); scene.add(dl2);

  const snap0 = d.snapshots[0];
  const nAtoms = snap0.positions.length;

  // Compute center
  if (!d.camera_target) {{
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < nAtoms; i++) {{
      cx += snap0.positions[i][0];
      cy += snap0.positions[i][1];
      cz += snap0.positions[i][2];
    }}
    cx /= nAtoms; cy /= nAtoms; cz /= nAtoms;
    controls.target.set(cx, cy, cz);
  }}

  // Particle radius
  const pRadius = d.is_2d ? 0.4 : 0.3;
  const sphereGeo = new THREE.SphereGeometry(pRadius, d.is_2d ? 16 : 10, d.is_2d ? 8 : 6);
  const sphereMat = new THREE.MeshPhongMaterial({{ shininess: 60, specular: 0x444444 }});
  const instancedMesh = new THREE.InstancedMesh(sphereGeo, sphereMat, nAtoms);
  scene.add(instancedMesh);

  const dummy = new THREE.Object3D();

  function updateParticles(idx) {{
    const snap = d.snapshots[idx];
    const [smin, smax] = d.speed_range;
    for (let i = 0; i < nAtoms; i++) {{
      dummy.position.set(snap.positions[i][0], snap.positions[i][1], snap.positions[i][2]);
      dummy.updateMatrix();
      instancedMesh.setMatrixAt(i, dummy.matrix);

      let col;
      if (d.color_mode === 'type') {{
        const typeIdx = (snap.types[i] || 1) - 1;
        col = TYPE_COLORS[Math.min(typeIdx, TYPE_COLORS.length - 1)];
      }} else {{
        const t = (snap.speeds[i] - smin) / (smax - smin + 1e-12);
        col = speedToColor(t);
      }}
      instancedMesh.setColorAt(i, col);
    }}
    instancedMesh.instanceMatrix.needsUpdate = true;
    instancedMesh.instanceColor.needsUpdate = true;
  }}

  updateParticles(0);

  // Bounding box (updates with snapshots for NPT/deform)
  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const boxEdges = new THREE.EdgesGeometry(boxGeo);
  const boxLine = new THREE.LineSegments(boxEdges,
    new THREE.LineBasicMaterial({{color:0x475569, transparent:true, opacity:0.5}}));
  scene.add(boxLine);

  function updateBox(idx) {{
    const snap = d.snapshots[idx];
    const bx = snap.box[0], by = snap.box[1], bz = snap.box[2];
    boxLine.scale.set(bx, by, d.is_2d ? 0.01 : bz);
    boxLine.position.set(bx/2, by/2, d.is_2d ? 0 : bz/2);
  }}
  updateBox(0);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    updateParticles(idx);
    updateBox(idx);
    tval.textContent = 't = ' + d.snapshots[idx].time.toFixed(2);
  }});

  viewers[sid] = {{ renderer, scene, cam, controls, updateParticles, updateBox, slider, tval }};
  playStates[sid] = {{ playing: false, interval: null }};

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }}
  animate();
}}

function togglePlay(sid) {{
  const ps = playStates[sid];
  const v = viewers[sid];
  const d = DATA[sid];
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {{
    btn.textContent = 'Pause';
    if (v.controls.autoRotate !== undefined) v.controls.autoRotate = false;
    ps.interval = setInterval(() => {{
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= d.snapshots.length) idx = 0;
      v.slider.value = idx;
      v.updateParticles(idx);
      v.updateBox(idx);
      v.tval.textContent = 't = ' + d.snapshots[idx].time.toFixed(2);
    }}, 200);
  }} else {{
    btn.textContent = 'Play';
    if (!d.is_2d) v.controls.autoRotate = true;
    clearInterval(ps.interval);
  }}
}}

// Init all viewers
Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly Charts ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:15, t:35, b:40 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0',
           title:{{ text:'Time (LJ units)', font:{{ size:10 }} }} }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const c = DATA[sid].charts;
  const cs = DATA[sid].chart_set;

  if (cs === 'glass') {{
    // Glass: Temperature quench, PE plateau, Energy components, Pressure
    Plotly.newPlot('chart-a-'+sid, [{{
      x:c.times, y:c.temperature, type:'scatter', mode:'lines',
      line:{{ color:'#6366f1', width:2.5 }},
    }}], {{...pLayout, title:{{ text:'Temperature (quench)', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'T', font:{{ size:10 }} }} }}
    }}, pCfg);

    Plotly.newPlot('chart-b-'+sid, [{{
      x:c.times, y:c.pe, type:'scatter', mode:'lines',
      line:{{ color:'#8b5cf6', width:2.5 }},
      fill:'tozeroy', fillcolor:'rgba(139,92,246,0.06)',
    }}], {{...pLayout, title:{{ text:'Potential Energy', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'PE', font:{{ size:10 }} }} }}
    }}, pCfg);

    Plotly.newPlot('chart-c-'+sid, [
      {{ x:c.times, y:c.pe, type:'scatter', mode:'lines',
         line:{{ color:'#6366f1', width:1.5 }}, name:'PE' }},
      {{ x:c.times, y:c.ke, type:'scatter', mode:'lines',
         line:{{ color:'#f43f5e', width:1.5 }}, name:'KE' }},
      {{ x:c.times, y:c.etotal, type:'scatter', mode:'lines',
         line:{{ color:'#1e293b', width:2, dash:'dot' }}, name:'Total' }},
    ], {{...pLayout, title:{{ text:'Energy Components', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
    }}, pCfg);

    Plotly.newPlot('chart-d-'+sid, [{{
      x:c.times, y:c.pressure, type:'scatter', mode:'lines',
      line:{{ color:'#f59e0b', width:2 }},
    }}], {{...pLayout, title:{{ text:'Pressure', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'P', font:{{ size:10 }} }} }}
    }}, pCfg);

  }} else if (cs === 'tension') {{
    // Tension: Stress-strain, Energy, Temperature, Box Lz
    Plotly.newPlot('chart-a-'+sid, [{{
      x:c.strain, y:c.pzz.map(v => -v), type:'scatter', mode:'lines',
      line:{{ color:'#f43f5e', width:2.5 }},
    }}], {{...pLayout,
      title:{{ text:'Stress-Strain Curve', font:{{ size:12, color:'#334155' }} }},
      xaxis:{{...pLayout.xaxis, title:{{ text:'Strain (%)', font:{{ size:10 }} }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'-sigma_zz', font:{{ size:10 }} }} }}
    }}, pCfg);

    Plotly.newPlot('chart-b-'+sid, [
      {{ x:c.times, y:c.pe, type:'scatter', mode:'lines',
         line:{{ color:'#6366f1', width:1.5 }}, name:'PE' }},
      {{ x:c.times, y:c.ke, type:'scatter', mode:'lines',
         line:{{ color:'#f43f5e', width:1.5 }}, name:'KE' }},
    ], {{...pLayout, title:{{ text:'Energy Components', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
    }}, pCfg);

    Plotly.newPlot('chart-c-'+sid, [{{
      x:c.times, y:c.temperature, type:'scatter', mode:'lines',
      line:{{ color:'#10b981', width:2 }},
    }}], {{...pLayout, title:{{ text:'Temperature', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'T', font:{{ size:10 }} }} }}
    }}, pCfg);

    Plotly.newPlot('chart-d-'+sid, [{{
      x:c.times, y:c.volume, type:'scatter', mode:'lines',
      line:{{ color:'#8b5cf6', width:2 }},
      fill:'tozeroy', fillcolor:'rgba(139,92,246,0.06)',
    }}], {{...pLayout, title:{{ text:'Cell Volume', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'V', font:{{ size:10 }} }} }}
    }}, pCfg);

  }} else if (cs === 'npt') {{
    // NPT: Volume, Density, Energy, Pressure
    Plotly.newPlot('chart-a-'+sid, [{{
      x:c.times, y:c.volume, type:'scatter', mode:'lines',
      line:{{ color:'#f59e0b', width:2.5 }},
      fill:'tozeroy', fillcolor:'rgba(245,158,11,0.06)',
    }}], {{...pLayout, title:{{ text:'Volume', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'V', font:{{ size:10 }} }} }}
    }}, pCfg);

    Plotly.newPlot('chart-b-'+sid, [{{
      x:c.times, y:c.density, type:'scatter', mode:'lines',
      line:{{ color:'#6366f1', width:2.5 }},
    }}], {{...pLayout, title:{{ text:'Number Density', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'rho', font:{{ size:10 }} }} }}
    }}, pCfg);

    Plotly.newPlot('chart-c-'+sid, [
      {{ x:c.times, y:c.pe, type:'scatter', mode:'lines',
         line:{{ color:'#6366f1', width:1.5 }}, name:'PE' }},
      {{ x:c.times, y:c.ke, type:'scatter', mode:'lines',
         line:{{ color:'#f43f5e', width:1.5 }}, name:'KE' }},
    ], {{...pLayout, title:{{ text:'Energy Components', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
    }}, pCfg);

    Plotly.newPlot('chart-d-'+sid, [{{
      x:c.times, y:c.pressure, type:'scatter', mode:'lines',
      line:{{ color:'#10b981', width:2 }},
    }}], {{...pLayout, title:{{ text:'Pressure', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'P', font:{{ size:10 }} }} }}
    }}, pCfg);

  }} else {{
    // Standard: Total Energy, Components, Temperature, Pressure
    Plotly.newPlot('chart-a-'+sid, [{{
      x:c.times, y:c.etotal, type:'scatter', mode:'lines+markers',
      line:{{ color:'#6366f1', width:2 }}, marker:{{ size:3 }},
    }}], {{...pLayout, title:{{ text:'Total Energy', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }}
    }}, pCfg);

    Plotly.newPlot('chart-b-'+sid, [
      {{ x:c.times, y:c.pe, type:'scatter', mode:'lines',
         line:{{ color:'#6366f1', width:1.5 }}, name:'PE' }},
      {{ x:c.times, y:c.ke, type:'scatter', mode:'lines',
         line:{{ color:'#f43f5e', width:1.5 }}, name:'KE' }},
    ], {{...pLayout, title:{{ text:'Energy Components', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
    }}, pCfg);

    Plotly.newPlot('chart-c-'+sid, [{{
      x:c.times, y:c.temperature, type:'scatter', mode:'lines',
      line:{{ color:'#10b981', width:2 }},
      fill:'tozeroy', fillcolor:'rgba(16,185,129,0.06)',
    }}], {{...pLayout, title:{{ text:'Temperature', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'T', font:{{ size:10 }} }} }}
    }}, pCfg);

    Plotly.newPlot('chart-d-'+sid, [{{
      x:c.times, y:c.pressure, type:'scatter', mode:'lines',
      line:{{ color:'#f59e0b', width:2 }},
    }}], {{...pLayout, title:{{ text:'Pressure', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'P', font:{{ size:10 }} }} }}
    }}, pCfg);
  }}
}});

</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


def run_demo():
    import subprocess
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        snapshots, runtime = run_simulation(cfg)
        sim_results.append((cfg, (snapshots, runtime)))
        print(f'  Runtime: {runtime:.2f}s')
        print(f'  {len(snapshots)} snapshots, {snapshots[0]["num_atoms"]} atoms')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
