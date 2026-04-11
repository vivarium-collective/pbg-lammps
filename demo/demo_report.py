"""Demo: LAMMPS multi-configuration molecular dynamics report.

Runs three distinct MD simulations (LJ gas NVE, LJ liquid NVT, crystal
melting NVT), generates interactive 3D particle viewers with Three.js,
Plotly charts, bigraph-viz architecture diagrams, and navigatable PBG
document trees — all in a single self-contained HTML.
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
        'id': 'gas_nve',
        'title': 'Lennard-Jones Gas (NVE)',
        'subtitle': 'Dilute gas with microcanonical energy conservation',
        'description': (
            'A dilute Lennard-Jones gas on a simple cubic lattice at low density '
            '(rho=0.3) evolves under NVE dynamics. With no thermostat, total energy '
            'is conserved while kinetic and potential energy exchange freely. '
            'This demonstrates the symplectic Verlet integrator preserving the '
            'Hamiltonian in a gas-phase system.'
        ),
        'config': {
            'num_atoms_per_dim': 6,
            'density': 0.3,
            'lattice_style': 'sc',
            'temperature': 2.0,
            'timestep': 0.005,
            'cutoff': 2.5,
            'ensemble': 'nve',
            'seed': 12345,
        },
        'n_snapshots': 30,
        'total_time': 15.0,
        'camera': [25, 18, 25],
        'color_scheme': 'indigo',
    },
    {
        'id': 'liquid_nvt',
        'title': 'Lennard-Jones Liquid (NVT)',
        'subtitle': 'Dense liquid equilibration with Nose-Hoover thermostat',
        'description': (
            'A dense Lennard-Jones liquid (FCC lattice, rho=0.85) is equilibrated '
            'at T=1.0 using a Nose-Hoover thermostat. The system starts from a '
            'perfect crystal and rapidly disorders into a liquid phase. The '
            'thermostat controls temperature fluctuations while allowing natural '
            'pressure and energy evolution.'
        ),
        'config': {
            'num_atoms_per_dim': 5,
            'density': 0.85,
            'lattice_style': 'fcc',
            'temperature': 1.5,
            'timestep': 0.005,
            'cutoff': 2.5,
            'ensemble': 'nvt',
            'target_temp': 1.0,
            'tdamp': 0.5,
            'seed': 54321,
        },
        'n_snapshots': 30,
        'total_time': 25.0,
        'camera': [18, 13, 18],
        'color_scheme': 'emerald',
    },
    {
        'id': 'melt',
        'title': 'Crystal Melting (NVT)',
        'subtitle': 'FCC crystal superheated above the melting point',
        'description': (
            'An FCC crystal at moderate density (rho=0.80) is superheated to '
            'T=2.5, well above the LJ melting temperature (~0.7). The ordered '
            'lattice structure rapidly breaks down as atoms gain enough kinetic '
            'energy to escape their lattice sites. This demonstrates the '
            'solid-to-liquid phase transition in a classical system.'
        ),
        'config': {
            'num_atoms_per_dim': 5,
            'density': 0.80,
            'lattice_style': 'fcc',
            'temperature': 0.5,
            'timestep': 0.005,
            'cutoff': 2.5,
            'ensemble': 'nvt',
            'target_temp': 2.5,
            'tdamp': 0.5,
            'seed': 99999,
        },
        'n_snapshots': 30,
        'total_time': 30.0,
        'camera': [18, 13, 18],
        'color_scheme': 'rose',
    },
]


def run_simulation(cfg_entry):
    """Run a single simulation, returning snapshots and wall-clock runtime."""
    core = allocate_core()
    core.register_link('LAMMPSProcess', LAMMPSProcess)

    t0 = _time.perf_counter()
    proc = LAMMPSProcess(config=cfg_entry['config'], core=core)
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
        'temperature': s['temperature'],
        'potential_energy': s['potential_energy'],
        'kinetic_energy': s['kinetic_energy'],
        'total_energy': s['total_energy'],
        'pressure': s['pressure'],
        'num_atoms': s['num_atoms'],
    }


def generate_bigraph_image(cfg_entry):
    """Generate a colored bigraph-viz PNG for the composite document."""
    from bigraph_viz import plot_bigraph

    doc = {
        'lammps': {
            '_type': 'process',
            'address': 'local:LAMMPSProcess',
            'config': {k: v for k, v in cfg_entry['config'].items()
                       if k in ('ensemble', 'density', 'temperature')},
            'interval': cfg_entry['total_time'] / cfg_entry['n_snapshots'],
            'inputs': {},
            'outputs': {
                'temperature': ['stores', 'temperature'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
                'positions': ['stores', 'positions'],
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
                'time': 'float',
            }},
            'inputs': {
                'temperature': ['stores', 'temperature'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
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
        num_atoms_per_dim=cfg.get('num_atoms_per_dim', 5),
        density=cfg.get('density', 0.6),
        lattice_style=cfg.get('lattice_style', 'sc'),
        temperature=cfg.get('temperature', 1.0),
        timestep=cfg.get('timestep', 0.005),
        ensemble=cfg.get('ensemble', 'nve'),
        target_temp=cfg.get('target_temp', 1.0),
        tdamp=cfg.get('tdamp', 0.5),
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
}


def generate_html(sim_results, output_path):
    """Generate comprehensive HTML report."""

    sections_html = []
    all_js_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        n_atoms = snapshots[0]['num_atoms']

        # Compute speed stats
        all_speeds = []
        for s in snapshots:
            vels = np.array(s['velocities'])
            speeds = np.linalg.norm(vels, axis=1)
            all_speeds.extend(speeds.tolist())
        speed_min = float(np.percentile(all_speeds, 2))
        speed_max = float(np.percentile(all_speeds, 98))

        # Time series
        times = [s['time'] for s in snapshots]
        temps = [s['temperature'] for s in snapshots]
        pe = [s['potential_energy'] for s in snapshots]
        ke = [s['kinetic_energy'] for s in snapshots]
        etotal = [s['total_energy'] for s in snapshots]
        press = [s['pressure'] for s in snapshots]

        # JS data — only send positions and speeds per snapshot (not full velocities)
        js_snapshots = []
        for s in snapshots:
            vels = np.array(s['velocities'])
            speeds = np.linalg.norm(vels, axis=1).tolist()
            js_snapshots.append({
                'time': s['time'],
                'positions': s['positions'],
                'speeds': speeds,
            })

        all_js_data[sid] = {
            'snapshots': js_snapshots,
            'speed_range': [speed_min, speed_max],
            'camera': cfg['camera'],
            'charts': {
                'times': times, 'temperature': temps,
                'pe': pe, 'ke': ke, 'etotal': etotal,
                'pressure': press,
            },
        }

        # Bigraph PNG
        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        # PBG document JSON
        pbg_doc = build_pbg_document(cfg)

        # Summary stats
        t0_val, t1_val = temps[0], temps[-1]
        e0_val, e1_val = etotal[0], etotal[-1]
        e_drift = abs(e1_val - e0_val) / max(abs(e0_val), 1e-10) * 100

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
        <div class="metric"><span class="metric-label">Ensemble</span><span class="metric-value">{cfg['config']['ensemble'].upper()}</span></div>
        <div class="metric"><span class="metric-label">Temp (final)</span><span class="metric-value">{t1_val:.3f}</span><span class="metric-sub">{t0_val:.2f} &rarr; {t1_val:.2f}</span></div>
        <div class="metric"><span class="metric-label">Total Energy</span><span class="metric-value">{e1_val:.3f}</span></div>
        <div class="metric"><span class="metric-label">E Drift</span><span class="metric-value">{e_drift:.2f}%</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">3D Particle Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="particle-canvas"></canvas>
        <div class="viewer-info">
          <strong>{n_atoms}</strong> atoms &middot; {cfg['config']['lattice_style'].upper()} lattice<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="colorbar-box">
          <div class="cb-title">Speed |v|</div>
          <div class="cb-val">{speed_max:.2f}</div>
          <div class="cb-gradient"></div>
          <div class="cb-val">{speed_min:.2f}</div>
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
        <div class="chart-box"><div id="chart-energy-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-components-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-temp-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-press-{sid}" class="chart"></div></div>
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
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             transition:all .15s; }}
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
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.7rem; color:#94a3b8; }}
.viewer-wrap {{ position:relative; background:#0f172a; border:1px solid #334155;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.particle-canvas {{ width:100%; height:500px; display:block; cursor:grab; }}
.particle-canvas:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(15,23,42,.85);
                border:1px solid #334155; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#94a3b8; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#e2e8f0; }}
.colorbar-box {{ position:absolute; top:.8rem; right:.8rem; background:rgba(15,23,42,.85);
                 border:1px solid #334155; border-radius:8px; padding:.6rem;
                 display:flex; flex-direction:column; align-items:center; gap:.2rem;
                 backdrop-filter:blur(4px); }}
.cb-title {{ font-size:.65rem; text-transform:uppercase; letter-spacing:.04em; color:#94a3b8; }}
.cb-gradient {{ width:16px; height:100px; border-radius:3px;
  background:linear-gradient(to bottom, #e61a0d, #e6c01a, #4dd94d, #12b5c9, #3112cc); }}
.cb-val {{ font-size:.65rem; color:#64748b; }}
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
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>LAMMPS Molecular Dynamics Report</h1>
  <p>Three molecular dynamics simulations wrapped as <strong>process-bigraph</strong>
  Processes using the LAMMPS engine. Each configuration demonstrates a distinct
  thermodynamic scenario with interactive 3D particle visualization.</p>
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
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
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

// ─── Three.js Particle Viewers ───
const viewers = {{}};
const playStates = {{}};

function speedToColor(t) {{
  // Turbo-like: blue -> cyan -> green -> yellow -> red
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

function initViewer(sid) {{
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 500;
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
  controls.autoRotateSpeed = 0.6;

  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dl1.position.set(30, 50, 40); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0x8b9cc7, 0.3);
  dl2.position.set(-20, -10, -30); scene.add(dl2);

  const snap0 = d.snapshots[0];
  const nAtoms = snap0.positions.length;

  // Compute bounding box center for camera target
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < nAtoms; i++) {{
    cx += snap0.positions[i][0];
    cy += snap0.positions[i][1];
    cz += snap0.positions[i][2];
  }}
  cx /= nAtoms; cy /= nAtoms; cz /= nAtoms;
  controls.target.set(cx, cy, cz);

  // Use InstancedMesh for efficient particle rendering
  const sphereGeo = new THREE.SphereGeometry(0.35, 12, 8);
  const sphereMat = new THREE.MeshPhongMaterial({{
    shininess: 60, specular: 0x444444,
  }});
  const instancedMesh = new THREE.InstancedMesh(sphereGeo, sphereMat, nAtoms);
  scene.add(instancedMesh);

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  function updateParticles(idx) {{
    const snap = d.snapshots[idx];
    const [smin, smax] = d.speed_range;
    for (let i = 0; i < nAtoms; i++) {{
      dummy.position.set(snap.positions[i][0], snap.positions[i][1], snap.positions[i][2]);
      dummy.updateMatrix();
      instancedMesh.setMatrixAt(i, dummy.matrix);

      const t = (snap.speeds[i] - smin) / (smax - smin + 1e-12);
      instancedMesh.setColorAt(i, speedToColor(t));
    }}
    instancedMesh.instanceMatrix.needsUpdate = true;
    instancedMesh.instanceColor.needsUpdate = true;
  }}

  updateParticles(0);

  // Bounding box wireframe
  let xmin=Infinity, ymin=Infinity, zmin=Infinity;
  let xmax=-Infinity, ymax=-Infinity, zmax=-Infinity;
  for (let i = 0; i < nAtoms; i++) {{
    const p = snap0.positions[i];
    xmin=Math.min(xmin,p[0]); ymin=Math.min(ymin,p[1]); zmin=Math.min(zmin,p[2]);
    xmax=Math.max(xmax,p[0]); ymax=Math.max(ymax,p[1]); zmax=Math.max(zmax,p[2]);
  }}
  const pad = 1.0;
  const boxGeo = new THREE.BoxGeometry(xmax-xmin+2*pad, ymax-ymin+2*pad, zmax-zmin+2*pad);
  const boxEdges = new THREE.EdgesGeometry(boxGeo);
  const boxLine = new THREE.LineSegments(boxEdges,
    new THREE.LineBasicMaterial({{color:0x475569, transparent:true, opacity:0.4}}));
  boxLine.position.set((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2);
  scene.add(boxLine);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    updateParticles(idx);
    tval.textContent = 't = ' + d.snapshots[idx].time;
  }});

  viewers[sid] = {{ renderer, scene, cam, controls, updateParticles, slider, tval }};
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
    v.controls.autoRotate = false;
    ps.interval = setInterval(() => {{
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= d.snapshots.length) idx = 0;
      v.slider.value = idx;
      v.updateParticles(idx);
      v.tval.textContent = 't = ' + d.snapshots[idx].time;
    }}, 250);
  }} else {{
    btn.textContent = 'Play';
    v.controls.autoRotate = true;
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

  Plotly.newPlot('chart-energy-'+sid, [{{
    x:c.times, y:c.etotal, type:'scatter', mode:'lines+markers',
    line:{{ color:'#6366f1', width:2 }}, marker:{{ size:4 }},
  }}], {{...pLayout, title:{{ text:'Total Energy', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }}
  }}, pCfg);

  Plotly.newPlot('chart-components-'+sid, [
    {{ x:c.times, y:c.pe, type:'scatter', mode:'lines+markers',
       line:{{ color:'#6366f1', width:1.5 }}, marker:{{ size:3 }}, name:'Potential' }},
    {{ x:c.times, y:c.ke, type:'scatter', mode:'lines+markers',
       line:{{ color:'#f43f5e', width:1.5 }}, marker:{{ size:3 }}, name:'Kinetic' }},
  ], {{...pLayout, title:{{ text:'Energy Components', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }},
    legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
  }}, pCfg);

  Plotly.newPlot('chart-temp-'+sid, [{{
    x:c.times, y:c.temperature, type:'scatter', mode:'lines+markers',
    line:{{ color:'#10b981', width:2 }}, marker:{{ size:4 }},
    fill:'tozeroy', fillcolor:'rgba(16,185,129,0.06)',
  }}], {{...pLayout, title:{{ text:'Temperature', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'T (LJ units)', font:{{ size:10 }} }} }}, showlegend:false
  }}, pCfg);

  Plotly.newPlot('chart-press-'+sid, [{{
    x:c.times, y:c.pressure, type:'scatter', mode:'lines+markers',
    line:{{ color:'#f59e0b', width:2 }}, marker:{{ size:4 }},
    fill:'tozeroy', fillcolor:'rgba(245,158,11,0.06)',
  }}], {{...pLayout, title:{{ text:'Pressure', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'P (LJ units)', font:{{ size:10 }} }} }}, showlegend:false
  }}, pCfg);
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
        print(f'  {len(snapshots)} snapshots collected')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    # Open in Safari
    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
