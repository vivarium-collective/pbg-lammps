"""Demo: visually rich LAMMPS physics with a benign cyclic control loop.

Three molecular dynamics scenarios:

  1. Spinodal decomposition — 50:50 binary Lennard-Jones mixture quenched
     below its consolute temperature. Same-species attractions are stronger
     than cross-species, so the system spontaneously demixes into A-rich
     and B-rich domains. Pure NVT, observed via the process-bigraph emitter.

  2. Kremer-Grest polymer melt — bead-spring chains with finitely
     extensible nonlinear elastic (FENE) bonds and WCA repulsion. The
     foundational coarse-grained polymer model. Chains start as straight
     rods and rapidly randomize. Pure NVT, observed via the emitter.

  3. Reversible compression cycle — a Lennard-Jones liquid is driven through
     a target_pressure schedule that goes 1.0 -> 4.5 -> 1.0 over the
     trajectory. A ScheduledSetpoint process publishes the schedule into
     controls/target_pressure; LAMMPS reads that store and re-applies its
     NPT barostat each step. The box visibly contracts and re-expands; the
     final pressure and volume return close to their starting values, so
     the loop *leaves the state unperturbed* in aggregate even as it
     traces out a clear orbit in P-V space.

The first two are pure LAMMPS — no control input perturbs the dynamics.
The third exercises the new target_pressure input port through a closed
thermodynamic cycle.
"""

import base64
import json
import os
import tempfile
import time as _time

import numpy as np
from process_bigraph import Composite, Process, allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_lammps import register_pbg_lammps
from pbg_lammps.composites import make_lammps_document


# ── Helper Process used in the cycle demo ────────────────────────────

class ScheduledSetpoint(Process):
    """Emit a piecewise-linear setpoint as a function of time."""

    config_schema = {
        'breakpoints': {'_type': 'list[float]', '_default': [0.0]},
        'values': {'_type': 'list[float]', '_default': [1.0]},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._t = 0.0

    def inputs(self):
        return {}

    def outputs(self):
        return {'value': 'overwrite[float]'}

    def initial_state(self):
        return {'value': self.config['values'][0]}

    def update(self, state, interval):
        self._t += interval
        bps = self.config['breakpoints']
        vs = self.config['values']
        if self._t <= bps[0]:
            return {'value': vs[0]}
        if self._t >= bps[-1]:
            return {'value': vs[-1]}
        for i in range(len(bps) - 1):
            if bps[i] <= self._t <= bps[i + 1]:
                span = bps[i + 1] - bps[i]
                if span <= 0:
                    return {'value': vs[i + 1]}
                f = (self._t - bps[i]) / span
                return {'value': vs[i] + f * (vs[i + 1] - vs[i])}
        return {'value': vs[-1]}


# ── Polymer data-file generator ──────────────────────────────────────

def _polymer_data_text(n_chains, chain_len, box_size, bond_len=0.97):
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


# ── LAMMPS input scripts ────────────────────────────────────────────

SPINODAL_IN = """\
# Spinodal decomposition: 50:50 binary LJ mixture quenched below T_c.
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

# Randomly relabel half the atoms as type 2 (50:50 mixture).
set             type 1 type/fraction 2 0.5 48392

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5     # A-A attraction
pair_coeff      2 2 1.0 1.0 2.5     # B-B attraction
pair_coeff      1 2 0.5 1.0 2.5     # A-B weaker -> demixing
pair_modify     shift yes

velocity        all create 2.0 87287 dist gaussian
timestep        0.005

# NVT well below the consolute temperature.
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

# Purely repulsive Weeks-Chandler-Andersen potential.
pair_style      lj/cut 1.122462
pair_coeff      1 1 1.0 1.0 1.122462
pair_modify     shift yes

# Finitely extensible nonlinear elastic bonds.
bond_style      fene
bond_coeff      1 30.0 1.5 1.0 1.0
special_bonds   fene

velocity        all create 1.0 87287 dist gaussian
timestep        0.005

fix             integ all nvt temp 1.0 1.0 0.5
"""

CYCLE_IN = """\
# Lennard-Jones liquid driven through a reversible pressure cycle.
# A ScheduledSetpoint process publishes target_pressure over time; the
# bridge re-issues the NPT fix every step so the box breathes
# coherently. The schedule returns to its starting value.

units           lj
atom_style      atomic
dimension       3
boundary        p p p

lattice         fcc 0.85
region          box block 0 6 0 6 0 6
create_box      1 box
create_atoms    1 box
mass            1 1.0

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0
pair_modify     shift yes

velocity        all create 1.0 87287 dist gaussian
timestep        0.005

# Placeholder — LAMMPSProcess strips this and re-issues each step.
fix             integ all npt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0
"""


# ── Simulation configurations ───────────────────────────────────────

CONFIGS = [
    {
        'id': 'spinodal',
        'title': 'Spinodal Decomposition',
        'subtitle': 'Binary Lennard-Jones mixture demixing into A-rich and B-rich domains',
        'description': (
            'A 50:50 binary Lennard-Jones mixture (2,048 atoms) is quenched '
            'below its consolute temperature at T = 0.7. Same-species attractions '
            '(epsilon_AA = epsilon_BB = 1.0) dominate cross-species interactions '
            '(epsilon_AB = 0.5), so composition fluctuations amplify and coarsen '
            'into macroscopic A-rich and B-rich domains. This is the same physical '
            'mechanism underlying biomolecular condensate formation in cells. The '
            'process-bigraph layer is a passive observer here: the LAMMPS output '
            'ports stream directly to the emitter. No control input perturbs the '
            'simulation.'
        ),
        'kind': 'pure',
        'input_filename': 'spinodal.in',
        'input_content': SPINODAL_IN,
        'interval': 1.0,
        'n_steps': 35,
        'color_scheme': 'indigo',
        'camera': [22, 16, 22],
        'color_mode': 'type',
    },
    {
        'id': 'polymer',
        'title': 'Kremer-Grest Polymer Melt',
        'subtitle': 'FENE bead-spring chains relaxing from straight rods into a disordered melt',
        'description': (
            'The Kremer-Grest model is the canonical coarse-grained polymer '
            'simulation: 36 chains of 20 beads each (720 atoms total) connected '
            'by finitely extensible nonlinear elastic (FENE) bonds and interacting '
            'via a purely repulsive Weeks-Chandler-Andersen potential. Starting '
            'from straight-rod configurations placed on a planar grid, the chains '
            'rapidly randomize through Rouse-like relaxation. This model captures '
            'universal polymer dynamics — entanglement, reptation, and viscoelastic '
            'response — and is the reference for studying polymer blends, gels, '
            'and biomolecular assemblies. As above, the bigraph layer is passive.'
        ),
        'kind': 'pure',
        'input_filename': 'polymer.in',
        'input_content': None,  # filled in at runtime (depends on data file)
        'interval': 1.5,
        'n_steps': 35,
        'color_scheme': 'emerald',
        'camera': [30, 22, 30],
        'color_mode': 'speed',
    },
    {
        'id': 'cycle',
        'title': 'Reversible Pressure Cycle (NPT)',
        'subtitle': (
            'A ScheduledSetpoint process drives target_pressure 1 -> 4.5 -> 1; '
            'the LAMMPS box breathes and returns to its starting state'
        ),
        'description': (
            'A LJ liquid is held in NPT while a ScheduledSetpoint process publishes '
            'a piecewise-linear target_pressure profile to controls/target_pressure. '
            'LAMMPSProcess reads that store every step and re-applies its NPT fix '
            '("unfix integ; fix integ all npt temp T T Tdamp iso P P Pdamp"). '
            'Pressure ramps from 1.0 up to 4.5 and back to 1.0; the simulation box '
            'visibly contracts at the apex of the cycle and re-expands. Because '
            'the schedule closes — both endpoints are the same — the bulk state '
            'returns close to its starting configuration: this is a *closed loop* '
            'in P-V space that leaves the macroscopic state approximately '
            'unperturbed at the end. The architecture exercises the new '
            'target_pressure input port end-to-end.'
        ),
        'kind': 'cycle',
        'input_filename': 'cycle.in',
        'input_content': CYCLE_IN,
        'breakpoints': [0.0, 12.0, 24.0],
        'values': [1.0, 4.5, 1.0],
        'thermostat_damping': 0.5,
        'barostat_damping': 1.0,
        'interval': 0.6,
        'n_steps': 40,
        'color_scheme': 'amber',
        'camera': [16, 12, 16],
        'color_mode': 'speed',
    },
]


# ── Composite builders ─────────────────────────────────────────────

def build_pure_composite(cfg):
    return make_lammps_document(
        input_script=cfg['input_content'],
        interval=cfg['interval'],
    )


def build_cycle_composite(cfg):
    doc = make_lammps_document(
        input_script=cfg['input_content'],
        interval=cfg['interval'],
        barostat_fix='integ',
        initial_target_pressure=cfg['values'][0],
    )
    doc['lammps']['config']['barostat_style'] = (
        f"npt temp 1.0 1.0 {cfg['thermostat_damping']} iso"
    )
    doc['scheduler'] = {
        '_type': 'process',
        'address': 'local:ScheduledSetpoint',
        'config': {
            'breakpoints': cfg['breakpoints'],
            'values': cfg['values'],
        },
        'interval': cfg['interval'],
        'inputs': {},
        'outputs': {'value': ['controls', 'target_pressure']},
    }
    return doc


def build_composite(cfg):
    if cfg['kind'] == 'pure':
        return build_pure_composite(cfg)
    if cfg['kind'] == 'cycle':
        return build_cycle_composite(cfg)
    raise ValueError(cfg['kind'])


# ── Run a single composite ─────────────────────────────────────────

def run_simulation(cfg):
    core = allocate_core()
    register_pbg_lammps(core)
    core.register_link('ScheduledSetpoint', ScheduledSetpoint)
    core.register_link('ram-emitter', RAMEmitter)

    doc = build_composite(cfg)
    if cfg['kind'] == 'cycle':
        doc['emitter']['config']['emit']['target_pressure'] = 'float'
        doc['emitter']['inputs']['target_pressure'] = ['controls', 'target_pressure']

    sim = Composite({'state': doc}, core=core)

    snapshots = []
    t0 = _time.perf_counter()

    def _snap(t):
        s = sim.state['stores']
        controls = sim.state.get('controls', {})
        return {
            'time': float(t),
            'positions': list(s.get('positions', [])),
            'velocities': list(s.get('velocities', [])),
            'atom_types': list(s.get('atom_types', [])),
            'box': list(s.get('box_dimensions', [10.0, 10.0, 10.0])),
            'temperature': float(s.get('temperature', 0.0)),
            'volume': float(s.get('volume', 0.0)),
            'target_pressure': float(controls.get('target_pressure', 0.0)),
        }

    sim.run(cfg['interval'])
    snapshots.append(_snap(cfg['interval']))
    for step in range(1, cfg['n_steps']):
        sim.run(cfg['interval'])
        snapshots.append(_snap(cfg['interval'] * (step + 1)))

    runtime = _time.perf_counter() - t0

    raw = []
    try:
        from process_bigraph import gather_emitter_results
        em = gather_emitter_results(sim)
        if ('emitter',) in em:
            raw = em[('emitter',)]
    except Exception:
        raw = []
    raw = [r for r in raw if r.get('time', 0.0) > 0.0]

    return snapshots, raw, runtime


# ── Bigraph diagram ────────────────────────────────────────────────

def _diagram_doc(cfg):
    if cfg['kind'] == 'cycle':
        return {
            'scheduler': {
                '_type': 'process',
                'address': 'local:ScheduledSetpoint',
                'outputs': {'value': ['controls', 'target_pressure']},
            },
            'lammps': {
                '_type': 'process',
                'address': 'local:LAMMPSProcess',
                'inputs': {'target_pressure': ['controls', 'target_pressure']},
                'outputs': {
                    'pressure': ['stores', 'pressure'],
                    'volume': ['stores', 'volume'],
                    'temperature': ['stores', 'temperature'],
                },
            },
            'controls': {'target_pressure': 1.0},
            'stores': {},
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'inputs': {
                    'pressure': ['stores', 'pressure'],
                    'volume': ['stores', 'volume'],
                    'target_pressure': ['controls', 'target_pressure'],
                    'time': ['global_time'],
                },
            },
        }
    # pure
    return {
        'lammps': {
            '_type': 'process',
            'address': 'local:LAMMPSProcess',
            'outputs': {
                'temperature': ['stores', 'temperature'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
                'volume': ['stores', 'volume'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'inputs': {
                'temperature': ['stores', 'temperature'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
                'volume': ['stores', 'volume'],
                'time': ['global_time'],
            },
        },
    }


def generate_bigraph_image(cfg):
    from bigraph_viz import plot_bigraph
    doc = _diagram_doc(cfg)
    if cfg['kind'] == 'cycle':
        node_colors = {
            ('scheduler',): '#f59e0b',
            ('lammps',): '#0ea5e9',
            ('emitter',): '#8b5cf6',
            ('stores',): '#fef3c7',
            ('controls',): '#fde68a',
        }
    else:
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


# ── HTML rendering ─────────────────────────────────────────────────

COLOR_SCHEMES = {
    'indigo':  {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'rose':    {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
    'amber':   {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#d97706'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
}


def _escape_html(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _build_pbg_doc_for_display(cfg):
    if cfg['kind'] == 'cycle':
        doc = build_cycle_composite(cfg)
    else:
        doc = build_pure_composite(cfg)
    if 'lammps' in doc:
        doc['lammps']['config']['input_script'] = (
            f"<see {cfg['input_filename']} above>"
        )
    return doc


def generate_html(sim_results, output_path):
    sections = []
    js_data = {}

    for idx, (cfg, snapshots, emitter_data, runtime) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        n_atoms = len(snapshots[0]['atom_types']) if snapshots[0]['atom_types'] else 0

        times = [r.get('time', 0.0) for r in emitter_data]
        temperature = [r.get('temperature', 0.0) for r in emitter_data]
        pe = [r.get('potential_energy', 0.0) for r in emitter_data]
        ke = [r.get('kinetic_energy', 0.0) for r in emitter_data]
        etotal = [r.get('total_energy', 0.0) for r in emitter_data]
        pressure = [r.get('pressure', 0.0) for r in emitter_data]
        volume = [r.get('volume', 0.0) for r in emitter_data]
        target_pressure = [r.get('target_pressure', None) for r in emitter_data]

        all_speeds = []
        for s in snapshots:
            if s['velocities']:
                vs = np.array(s['velocities'])
                all_speeds.extend(np.linalg.norm(vs, axis=1).tolist())
        if all_speeds:
            speed_min = float(np.percentile(all_speeds, 2))
            speed_max = float(np.percentile(all_speeds, 98))
        else:
            speed_min, speed_max = 0.0, 1.0

        type_counts = {}
        for t in (snapshots[0]['atom_types'] or []):
            type_counts[t] = type_counts.get(t, 0) + 1

        js_snaps = []
        for s in snapshots:
            speeds = []
            if s['velocities']:
                vs = np.array(s['velocities'])
                speeds = np.linalg.norm(vs, axis=1).tolist()
            js_snaps.append({
                'time': s['time'],
                'positions': s['positions'],
                'speeds': speeds,
                'types': s['atom_types'],
                'box': s['box'],
            })

        js_data[sid] = {
            'snapshots': js_snaps,
            'speed_range': [speed_min, speed_max],
            'camera': cfg['camera'],
            'kind': cfg['kind'],
            'color_mode': cfg['color_mode'],
            'charts': {
                'times': times, 'temperature': temperature,
                'pe': pe, 'ke': ke, 'etotal': etotal,
                'pressure': pressure, 'volume': volume,
                'target_pressure': target_pressure,
            },
        }

        bigraph_img = generate_bigraph_image(cfg)
        pbg_doc = _build_pbg_doc_for_display(cfg)

        lines = (cfg['input_content'] or '').rstrip('\n').split('\n')
        numbered = '\n'.join(
            f'<span class="in-line"><span class="in-ln">{i+1:>3}</span> {_escape_html(l)}</span>'
            for i, l in enumerate(lines))

        # Cycle deltas (start vs end) — for the "closes the loop" claim.
        loop_block = ''
        if cfg['kind'] == 'cycle' and pressure and volume:
            dP = pressure[-1] - pressure[0]
            dV = volume[-1] - volume[0]
            loop_block = (
                f'<div class="loop-banner">'
                f'<strong>Cycle closure:</strong> ΔP = {dP:+.3f} '
                f'(start {pressure[0]:.2f} → end {pressure[-1]:.2f}) · '
                f'ΔV = {dV:+.2f} (start {volume[0]:.2f} → end {volume[-1]:.2f})'
                f'</div>'
            )

        # Colorbar
        if cfg['color_mode'] == 'type':
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
        final_temp = temperature[-1] if temperature else 0.0
        final_press = pressure[-1] if pressure else 0.0

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
      {loop_block}

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Atoms</span><span class="metric-value">{n_atoms:,}</span></div>
        <div class="metric"><span class="metric-label">Steps</span><span class="metric-value">{cfg['n_steps']}</span></div>
        <div class="metric"><span class="metric-label">Δt store</span><span class="metric-value">{cfg['interval']:.2f}</span></div>
        <div class="metric"><span class="metric-label">T (final)</span><span class="metric-value">{final_temp:.3f}</span></div>
        <div class="metric"><span class="metric-label">P (final)</span><span class="metric-value">{final_press:.2f}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">LAMMPS Input File &middot; <code class="in-fname">{cfg['input_filename']}</code></h3>
      <div class="in-file-wrap"><pre class="in-file">{numbered}</pre></div>

      <h3 class="subsection-title">3D Particle Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="particle-canvas"></canvas>
        <div class="viewer-info">
          <strong>{n_atoms:,}</strong> atoms &middot; {type_str}<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="colorbar-box">{cb_html}</div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(js_snaps)-1}" value="0" step="1" style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Time Series</h3>
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
        sections.append(section)

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    pbg_docs_for_js = {
        cfg['id']: _build_pbg_doc_for_display(cfg)
        for cfg in [r[0] for r in sim_results]
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pbg-lammps · Bidirectional Bridge Demo</title>
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
.page-header p {{ color:#64748b; font-size:.95rem; max-width:760px; }}
.page-header code {{ font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;
                     font-size:.82rem; background:#eef2ff; color:#4338ca;
                     padding:.05rem .35rem; border-radius:4px; }}
.nav {{ display:flex; gap:.6rem; padding:.8rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; flex-wrap:wrap; }}
.nav-link {{ padding:.4rem .8rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.8rem; font-weight:600; color:#1e293b;
             transition:all .15s; white-space:nowrap; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem; padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.92rem; margin-bottom:1rem; max-width:830px; }}
.loop-banner {{ background:#fef3c7; border:1px solid #fde68a; color:#854d0e;
                padding:.5rem .8rem; border-radius:8px; font-size:.82rem;
                margin-bottom:1rem; max-width:830px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155; margin:1.5rem 0 .8rem; }}
.in-fname {{ font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;
             font-size:.85rem; background:#eef2ff; color:#4338ca; padding:.1rem .45rem;
             border-radius:5px; font-weight:500; }}
.in-file-wrap {{ background:#0f172a; border:1px solid #334155; border-radius:10px;
                 overflow:auto; max-height:380px; margin-bottom:1rem; }}
.in-file {{ font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;
            font-size:.78rem; line-height:1.55; color:#cbd5e1; padding:.9rem 1rem; white-space:pre; }}
.in-line {{ display:block; }}
.in-ln {{ display:inline-block; width:2.5em; color:#475569; user-select:none;
          margin-right:.7rem; text-align:right; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(115px,1fr)); gap:.8rem; margin-bottom:1.5rem; }}
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
.cb-gradient {{ width:16px; height:90px; border-radius:3px; }}
.cb-val {{ font-size:.65rem; color:#64748b; text-align:center; max-width:80px; }}
.slider-controls {{ position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(15,23,42,.95));
                    padding:1.4rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }}
.slider-controls label {{ font-size:.8rem; color:#94a3b8; }}
.time-slider {{ flex:1; height:5px; }}
.time-val {{ font-size:.92rem; font-weight:600; color:#e2e8f0; min-width:100px; text-align:right; }}
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
              padding:1rem; max-height:500px; overflow-y:auto;
              font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;
              font-size:.78rem; line-height:1.5; }}
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
  <h1>pbg-lammps · Bidirectional Bridge Demo</h1>
  <p>Three molecular-dynamics scenarios. The first two are pure LAMMPS physics
  observed through process-bigraph emitters — no control input perturbs the
  dynamics. The third drives the new <code>target_pressure</code> input port
  with a closed schedule (1.0 → 4.5 → 1.0): the simulation traces a clear orbit
  in P-V space and returns close to its starting state, demonstrating the
  bidirectional wiring without permanently disturbing equilibrium.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections)}

<div class="footer">
  Generated by <strong>pbg-lammps</strong> · LAMMPS + process-bigraph
</div>

<script>
const DATA = {json.dumps(js_data)};
const DOCS = {json.dumps(pbg_docs_for_js)};

function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;').replace(/\\n/g,'\\\\n') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 6 && obj.every(x => typeof x !== 'object' || x === null)) {{
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
  }} else {{
    el.classList.add('jt-collapsed');
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

const TYPE_COLORS = [
  new THREE.Color(0.39, 0.40, 0.95),
  new THREE.Color(0.95, 0.25, 0.37),
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
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.autoRotate = true; controls.autoRotateSpeed = 0.5;

  const snap0 = d.snapshots[0];
  const nAtoms = snap0.positions.length;
  let cx=0, cy=0, cz=0;
  for (let i = 0; i < nAtoms; i++) {{
    cx += snap0.positions[i][0]; cy += snap0.positions[i][1]; cz += snap0.positions[i][2];
  }}
  if (nAtoms > 0) {{ cx /= nAtoms; cy /= nAtoms; cz /= nAtoms; }}
  controls.target.set(cx, cy, cz);

  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dl1.position.set(30, 50, 40); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0x8b9cc7, 0.3);
  dl2.position.set(-20, -10, -30); scene.add(dl2);

  const sphereGeo = new THREE.SphereGeometry(0.32, 10, 6);
  const sphereMat = new THREE.MeshPhongMaterial({{shininess: 60, specular: 0x444444}});
  const mesh = new THREE.InstancedMesh(sphereGeo, sphereMat, Math.max(nAtoms, 1));
  scene.add(mesh);
  const dummy = new THREE.Object3D();
  function updateParticles(idx) {{
    const snap = d.snapshots[idx];
    const [smin, smax] = d.speed_range;
    for (let i = 0; i < snap.positions.length; i++) {{
      dummy.position.set(snap.positions[i][0], snap.positions[i][1], snap.positions[i][2]);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      let col;
      if (d.color_mode === 'type') {{
        const t = (snap.types[i] || 1) - 1;
        col = TYPE_COLORS[Math.min(t, TYPE_COLORS.length - 1)];
      }} else {{
        const sp = snap.speeds[i] !== undefined ? snap.speeds[i] : 0.0;
        col = speedToColor((sp - smin) / (smax - smin + 1e-12));
      }}
      mesh.setColorAt(i, col);
    }}
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }}
  updateParticles(0);

  const boxGeo = new THREE.BoxGeometry(1,1,1);
  const boxEdges = new THREE.EdgesGeometry(boxGeo);
  const boxLine = new THREE.LineSegments(boxEdges,
    new THREE.LineBasicMaterial({{color:0x475569, transparent:true, opacity:0.4}}));
  scene.add(boxLine);
  function updateBox(idx) {{
    const b = d.snapshots[idx].box;
    boxLine.scale.set(b[0], b[1], b[2]);
    boxLine.position.set(b[0]/2, b[1]/2, b[2]/2);
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

// ─── Plotly charts ────────────────────────────────────────────
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
  const c = DATA[sid].charts;
  const kind = DATA[sid].kind;

  if (kind === 'cycle') {{
    Plotly.newPlot('chart-a-'+sid, [
      {{ x:c.times, y:c.target_pressure, type:'scatter', mode:'lines', name:'target P (scheduler)',
         line:{{ color:'#f59e0b', width:2.5, dash:'dot' }} }},
      {{ x:c.times, y:c.pressure, type:'scatter', mode:'lines', name:'measured P (LAMMPS)',
         line:{{ color:'#0ea5e9', width:2 }} }},
    ], {{...pL, title:{{ text:'Pressure cycle: setpoint vs measured', font:{{ size:12, color:'#334155' }} }},
        legend:{{ font:{{ size:10 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true}}, pC);
    Plotly.newPlot('chart-b-'+sid, [
      {{ x:c.times, y:c.volume, type:'scatter', mode:'lines',
         line:{{ color:'#8b5cf6', width:2 }}, fill:'tozeroy', fillcolor:'rgba(139,92,246,0.06)' }},
    ], {{...pL, title:{{ text:'Volume (response to barostat)', font:{{ size:12, color:'#334155' }} }}}}, pC);
    Plotly.newPlot('chart-c-'+sid, [
      {{ x:c.target_pressure, y:c.volume, type:'scatter', mode:'lines+markers',
         line:{{ color:'#f59e0b', width:2 }},
         marker:{{ size:5, color:c.times,
                   colorscale:[[0,'#fde68a'],[1,'#92400e']], showscale:false }} }},
    ], {{...pL, title:{{ text:'P-V trajectory (closes)', font:{{ size:12, color:'#334155' }} }},
         xaxis:{{...pL.xaxis, title:{{ text:'target P', font:{{ size:10 }} }} }},
         yaxis:{{...pL.yaxis, title:{{ text:'V', font:{{ size:10 }} }} }}}}, pC);
    Plotly.newPlot('chart-d-'+sid, [
      {{ x:c.times, y:c.temperature, type:'scatter', mode:'lines',
         line:{{ color:'#10b981', width:2 }} }},
    ], {{...pL, title:{{ text:'Temperature (NPT)', font:{{ size:12, color:'#334155' }} }}}}, pC);

  }} else {{
    Plotly.newPlot('chart-a-'+sid, [
      {{ x:c.times, y:c.etotal, type:'scatter', mode:'lines', name:'Total',
         line:{{ color:'#1e293b', width:2 }} }},
      {{ x:c.times, y:c.pe, type:'scatter', mode:'lines', name:'PE',
         line:{{ color:'#6366f1', width:1.5 }} }},
      {{ x:c.times, y:c.ke, type:'scatter', mode:'lines', name:'KE',
         line:{{ color:'#f43f5e', width:1.5 }} }},
    ], {{...pL, title:{{ text:'Energy components', font:{{ size:12, color:'#334155' }} }},
        legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true}}, pC);
    Plotly.newPlot('chart-b-'+sid, [
      {{ x:c.times, y:c.temperature, type:'scatter', mode:'lines',
         line:{{ color:'#10b981', width:2 }},
         fill:'tozeroy', fillcolor:'rgba(16,185,129,0.06)' }},
    ], {{...pL, title:{{ text:'Temperature', font:{{ size:12, color:'#334155' }} }}}}, pC);
    Plotly.newPlot('chart-c-'+sid, [
      {{ x:c.times, y:c.pressure, type:'scatter', mode:'lines',
         line:{{ color:'#f59e0b', width:2 }} }},
    ], {{...pL, title:{{ text:'Pressure', font:{{ size:12, color:'#334155' }} }}}}, pC);
    Plotly.newPlot('chart-d-'+sid, [
      {{ x:c.times, y:c.pe, type:'scatter', mode:'lines',
         line:{{ color:'#6366f1', width:2 }},
         fill:'tozeroy', fillcolor:'rgba(99,102,241,0.06)' }},
    ], {{...pL, title:{{ text:'Potential energy', font:{{ size:12, color:'#334155' }} }}}}, pC);
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


# ── Driver ─────────────────────────────────────────────────────────

def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    workdir = tempfile.mkdtemp(prefix='pbg_lammps_demo_')

    # Materialize the polymer data file and fill in the polymer config.
    poly_text, _ = _polymer_data_text(n_chains=36, chain_len=20, box_size=20.0)
    poly_data_path = os.path.join(workdir, 'polymer_melt.data')
    with open(poly_data_path, 'w') as f:
        f.write(poly_text)
    polymer_in = POLYMER_IN_TEMPLATE.format(data_file=poly_data_path)
    for cfg in CONFIGS:
        if cfg['id'] == 'polymer':
            cfg['input_content'] = polymer_in

    sim_results = []
    for cfg in CONFIGS:
        print(f"Running: {cfg['title']}...")
        snapshots, emitter_data, runtime = run_simulation(cfg)
        sim_results.append((cfg, snapshots, emitter_data, runtime))
        print(f"  Runtime: {runtime:.2f}s, {len(snapshots)} snapshots, "
              f"{len(emitter_data)} emitter records")

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    import shutil
    shutil.rmtree(workdir, ignore_errors=True)

    import webbrowser
    webbrowser.open('file://' + os.path.abspath(output_path))


if __name__ == '__main__':
    run_demo()
