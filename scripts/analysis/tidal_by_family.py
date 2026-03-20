"""
Per-family tidal sensitivity for top 20 families.
Ranks by Rayleigh R statistic, plots individual histograms ordered by sensitivity.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime, timezone
from scipy.signal import hilbert

import redpy

matplotlib.use('agg')

CONFIG   = 'axial_settings.cfg'
OUT_DIR  = 'runs/axial/eruption_analysis'
TIDE_DIR = os.path.expanduser('~/Documents/TidesArchive/Tides_Central_caldera')
YEARS    = list(range(2015, 2022))
DOWNSAMPLE = 240   # 15 s × 240 = 1 hour

# ── 1. Load tide data & compute analytic signal ───────────────────────────────
print('Loading tide data...')
chunks = []
for yr in YEARS:
    fpath = os.path.join(TIDE_DIR, f'pred_F_{yr}.txt')
    df = pd.read_csv(fpath, sep=r'\s+', header=None,
                     names=['yr','mo','dy','hr','mn','sc','tide'],
                     dtype=float)
    df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)
    chunks.append(df)
tide_df = pd.concat(chunks, ignore_index=True)

tide_times = np.array([
    datetime(int(r.yr), int(r.mo), int(r.dy),
             int(r.hr), int(r.mn), int(r.sc),
             tzinfo=timezone.utc).timestamp()
    for r in tide_df.itertuples()
])
tide_levels = tide_df['tide'].values.astype(float)

tide_dm  = tide_levels - tide_levels.mean()
analytic = hilbert(tide_dm)
print(f'Tide samples: {len(tide_times):,}')

def get_eq_phase(t_sec_array):
    """Return tidal phase (degrees, 0°=low tide) for each Unix timestamp."""
    ar = np.interp(t_sec_array, tide_times, np.real(analytic))
    ai = np.interp(t_sec_array, tide_times, np.imag(analytic))
    ph = np.angle(ar + 1j*ai) + np.pi           # shift: 0° = low tide
    ph = (ph + np.pi) % (2*np.pi) - np.pi        # wrap to (-π, π]
    return np.degrees(ph)

def rayleigh(phases_deg):
    phi = np.radians(phases_deg)
    n   = len(phi)
    if n == 0:
        return np.nan, np.nan, np.nan
    C = np.sum(np.cos(phi)); S = np.sum(np.sin(phi))
    R = np.sqrt(C**2 + S**2) / n
    z = n * R**2
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*n)
                      - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    p = float(np.clip(p, 0, 1))
    mean_dir = float(np.degrees(np.arctan2(S/n, C/n)) % 360)
    return float(R), mean_dir, p

# ── 2. Load REDPy & get top 20 ────────────────────────────────────────────────
print('Loading REDPy...')
d = redpy.Detector(CONFIG)
d.open()
rtimes_mpl = d.get('rtable', 'startTimeMPL')
nfam = len(d)

rtimes_sec = np.array([
    mdates.num2date(t).replace(tzinfo=timezone.utc).timestamp()
    for t in rtimes_mpl
])

sizes = sorted([(f, len(d.get_members(f))) for f in range(nfam)],
               key=lambda x: -x[1])
top20 = [f for f, _ in sizes[:20]]

# Time filter: within tide data range
t_min = tide_times[0]; t_max = tide_times[-1]

# ── 3. Per-family tidal analysis ──────────────────────────────────────────────
print('Computing per-family tidal phase...')
results = []
N_BINS  = 18
bin_edges   = np.linspace(-180, 180, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

for fnum in top20:
    mems = d.get_members(fnum)
    t    = rtimes_sec[mems]
    mask = (t >= t_min) & (t <= t_max)
    t_filt = t[mask]
    n_total = len(mems)
    n_tide  = mask.sum()

    if n_tide < 10:
        R, mdir, p = np.nan, np.nan, np.nan
        phases = np.array([])
        counts = np.zeros(N_BINS)
    else:
        phases = get_eq_phase(t_filt)
        R, mdir, p = rayleigh(phases)
        counts, _ = np.histogram(phases, bins=bin_edges)

    results.append(dict(fnum=fnum, n_total=n_total, n_tide=n_tide,
                        R=R, mean_dir=mdir, p=p,
                        phases=phases, counts=counts))
    print(f'  F{fnum:4d}: n={n_tide:5d}, R={R:.3f}, mean={mdir:.0f}°, p={p:.2e}')

d.close()

# ── 4. Rank by R ──────────────────────────────────────────────────────────────
results.sort(key=lambda x: -(x['R'] if not np.isnan(x['R']) else -1))

print('\nRanking by Rayleigh R:')
print(f"{'Rank':>4} {'Fnum':>5} {'n_tide':>7} {'R':>6} {'MeanDir':>8} {'p':>10}")
for i, r in enumerate(results):
    print(f"  {i+1:2d}  F{r['fnum']:4d}  {r['n_tide']:6d}  "
          f"{r['R']:.3f}  {r['mean_dir']:6.1f}°  {r['p']:.2e}")

# ── 5. Summary bar chart ──────────────────────────────────────────────────────
fig_sum, ax_sum = plt.subplots(figsize=(12, 5))
fnums_ranked = [r['fnum'] for r in results]
R_vals       = [r['R'] if not np.isnan(r['R']) else 0 for r in results]
colors_bar   = plt.get_cmap('RdYlGn')(np.linspace(0.85, 0.15, len(results)))
bars = ax_sum.bar(range(len(results)), R_vals, color=colors_bar,
                  edgecolor='white', lw=0.5)
ax_sum.set_xticks(range(len(results)))
ax_sum.set_xticklabels([f'F{r["fnum"]}\n(n={r["n_tide"]})' for r in results],
                        fontsize=7)
ax_sum.set_ylabel('Rayleigh R', fontsize=10)
ax_sum.set_title('Tidal Sensitivity (Rayleigh R) — Top 20 Families, Ranked\n'
                 '(higher R = stronger tidal triggering; 0° = low tide = peak stress)',
                 fontsize=10)
ax_sum.axhline(0, color='k', lw=0.5)
# Annotate p-values
for i, r in enumerate(results):
    if not np.isnan(r['p']):
        p_str = 'p<0.001' if r['p'] < 0.001 else f"p={r['p']:.3f}"
        ax_sum.text(i, R_vals[i] + 0.004, p_str, ha='center',
                    va='bottom', fontsize=5.5, rotation=90)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/14_tidal_sensitivity_ranking.png', dpi=150)
plt.close()
print(f'\nSaved: {OUT_DIR}/14_tidal_sensitivity_ranking.png')

# ── 6. Grid of per-family histograms (4 rows × 5 cols, ranked order) ──────────
COLS = 5; ROWS = 4
fig, axes = plt.subplots(ROWS, COLS, figsize=(18, 13), sharey=False)
axes = axes.flatten()

for rank, (r, ax) in enumerate(zip(results, axes)):
    counts  = r['counts']
    n_tide  = r['n_tide']
    fnum    = r['fnum']
    R_val   = r['R']
    mdir    = r['mean_dir']
    p_val   = r['p']

    if n_tide < 10:
        ax.text(0.5, 0.5, 'n < 10\n(no data)', ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
        ax.set_title(f'Rank {rank+1}  F{fnum}', fontsize=8)
        continue

    expected = n_tide / N_BINS
    excess   = (counts - expected) / expected * 100

    bar_colors = np.where(excess >= 0, 'tomato', 'steelblue')
    ax.bar(bin_centers, excess, width=360/N_BINS,
           color=bar_colors, edgecolor='white', lw=0.4, alpha=0.85)
    ax.axhline(0, color='k', lw=0.7)

    # Mean direction arrow
    mdir_rad = np.radians(mdir - 180)   # shift back to original convention for display
    # Just annotate the mean direction as text
    p_str = f'p={p_val:.2e}' if p_val >= 1e-99 else 'p≈0'
    ax.set_title(f'Rank {rank+1}  F{fnum}  (n={n_tide})\nR={R_val:.3f}  {p_str}',
                 fontsize=7)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(['-180°', '-90°', '0°', '+90°', '+180°'], fontsize=5)
    ax.tick_params(labelsize=6)
    ax.set_xlabel('Phase (°)', fontsize=6)
    if rank % COLS == 0:
        ax.set_ylabel('Excess (%)', fontsize=6)

    # Color-coded background: more sensitive = warmer title box
    intensity = np.clip(R_val / 0.5, 0, 1)
    title_color = plt.cm.RdYlGn(0.85 - 0.7 * intensity)
    ax.set_facecolor((*title_color[:3], 0.08))

fig.suptitle('Per-Family Tidal Phase Histogram — Top 20 Families (ranked by Rayleigh R)\n'
             'Bar color: red = excess above uniform, blue = deficit  |  '
             '0° = low tide / peak extensional stress',
             fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/14_tidal_per_family.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/14_tidal_per_family.png')
print('\nDone.')
