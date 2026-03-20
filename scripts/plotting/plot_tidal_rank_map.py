"""
Top 20 families (by event count), re-ranked by tidal sensitivity (Rayleigh R).
Four 3-panel subplots (map / right depth / bottom depth), 5 families each.
Groups: tidal rank 1-5, 6-10, 11-15, 16-20.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from datetime import timezone, datetime
from scipy.signal import hilbert

import redpy
from redpy.outputs.mapping import _CALDERA_RIM_LON, _CALDERA_RIM_LAT, _STATIONS

matplotlib.use('agg')

CONFIG    = 'axial_settings.cfg'
CAT_DD    = 'axial_catalog_dd.csv'
CAT_FELIX = 'axial_catalog_felix_2022_2025.csv'
OUT_DIR   = 'runs/axial/eruption_analysis'
TIDE_DIR  = os.path.expanduser('~/Documents/TidesArchive/Tides_Central_caldera')
MARKER_SIZE = 4
DEPTH_MAX   = 2.5
DOWNSAMPLE  = 240   # 15 s × 240 = 1 h

# ── Tide: load & build analytic signal ────────────────────────────────────────
print('Loading tide data...')
chunks = []
for yr in range(2015, 2022):
    fpath = os.path.join(TIDE_DIR, f'pred_F_{yr}.txt')
    df = pd.read_csv(fpath, sep=r'\s+', header=None,
                     names=['yr','mo','dy','hr','mn','sc','tide'], dtype=float)
    chunks.append(df.iloc[::DOWNSAMPLE].reset_index(drop=True))
tide_df = pd.concat(chunks, ignore_index=True)

tide_times = np.array([
    datetime(int(r.yr), int(r.mo), int(r.dy),
             int(r.hr), int(r.mn), int(r.sc),
             tzinfo=timezone.utc).timestamp()
    for r in tide_df.itertuples()
])
tide_levels = tide_df['tide'].values.astype(float)
analytic    = hilbert(tide_levels - tide_levels.mean())
t_min, t_max = tide_times[0], tide_times[-1]

def get_eq_phase_deg(t_sec_array):
    ar = np.interp(t_sec_array, tide_times, np.real(analytic))
    ai = np.interp(t_sec_array, tide_times, np.imag(analytic))
    ph = np.angle(ar + 1j*ai) + np.pi
    ph = (ph + np.pi) % (2*np.pi) - np.pi
    return np.degrees(ph)

def rayleigh_R(phases_deg):
    if len(phases_deg) < 5:
        return np.nan
    phi = np.radians(phases_deg)
    C = np.sum(np.cos(phi)); S = np.sum(np.sin(phi))
    return float(np.sqrt(C**2 + S**2) / len(phi))

# ── Catalog ────────────────────────────────────────────────────────────────────
cat = pd.concat([pd.read_csv(CAT_DD), pd.read_csv(CAT_FELIX)], ignore_index=True)
cat['Time'] = pd.to_datetime(cat['Time'], utc=True)
cat_times_sec = cat['Time'].values.astype('datetime64[ns]').astype(float) / 1e9

# ── REDPy ──────────────────────────────────────────────────────────────────────
print('Loading REDPy...')
d = redpy.Detector(CONFIG)
d.open()

rtimes_mpl = d.get('rtable', 'startTimeMPL')
rtimes_sec = np.array([
    mdates.num2date(t).replace(tzinfo=timezone.utc).timestamp()
    for t in rtimes_mpl
])

sizes = sorted([(f, len(d.get_members(f))) for f in range(len(d))],
               key=lambda x: -x[1])
top20_fnums = [f for f, _ in sizes[:20]]
size_dict   = dict(sizes)

# ── Compute tidal R for each top-20 family ────────────────────────────────────
print('Computing tidal sensitivity per family...')
R_dict = {}
for fnum in top20_fnums:
    mems = d.get_members(fnum)
    t    = rtimes_sec[mems]
    mask = (t >= t_min) & (t <= t_max)
    t_f  = t[mask]
    R_dict[fnum] = rayleigh_R(get_eq_phase_deg(t_f)) if mask.sum() >= 5 else np.nan

# Re-rank top 20 by R (descending), NaN last
top20_by_R = sorted(top20_fnums,
                    key=lambda f: -(R_dict[f] if not np.isnan(R_dict.get(f, np.nan)) else -1))

print('\nTidal rank order:')
for rank, fnum in enumerate(top20_by_R, 1):
    R = R_dict[fnum]
    print(f'  Tidal rank {rank:2d}: F{fnum:4d}  n={size_dict[fnum]:6d}  R={R:.3f}')

groups       = [top20_by_R[0:5], top20_by_R[5:10],
                top20_by_R[10:15], top20_by_R[15:20]]
group_labels = ['Tidal rank 1–5 (most sensitive)',
                'Tidal rank 6–10',
                'Tidal rank 11–15',
                'Tidal rank 16–20 (least sensitive)']

COLORS = plt.get_cmap('tab10').colors

# ── Helper: match catalog locations ───────────────────────────────────────────
def get_family_locs(fnum):
    members = d.get_members(fnum)
    lons, lats, deps = [], [], []
    for m in members:
        t_s = rtimes_sec[m]
        diffs = np.abs(cat_times_sec - t_s)
        idx   = np.argmin(diffs)
        if diffs[idx] < 60:
            row = cat.iloc[idx]
            lons.append(float(row['Longitude']))
            lats.append(float(row['Latitude']))
            deps.append(float(np.clip(row['Depth'], 0, DEPTH_MAX)))
    return np.array(lons), np.array(lats), np.array(deps)

def _setup_map(ax):
    ax.set_xlim(-130.05, -129.94)
    ax.set_ylim(45.91, 46.00)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(labelsize=5)
    ax.set_ylabel('Latitude', fontsize=6)
    ax.plot(_CALDERA_RIM_LON, _CALDERA_RIM_LAT, 'k-', linewidth=0.6, zorder=2)
    for name, (slon, slat) in _STATIONS.items():
        ax.scatter(slon, slat, marker='^', s=12, color='k',
                   facecolors='none', linewidths=0.6, zorder=5)
        ax.annotate(name, (slon, slat), fontsize=3,
                    xytext=(2, 2), textcoords='offset points')

# ── Build figure ───────────────────────────────────────────────────────────────
print('Plotting...')
fig = plt.figure(figsize=(16, 14))
outer_gs = GridSpec(2, 2, figure=fig, hspace=0.22, wspace=0.18)

for gi, (group, label) in enumerate(zip(groups, group_labels)):
    row, col = gi // 2, gi % 2
    inner_gs = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_gs[row, col],
        width_ratios=[3, 1], height_ratios=[3, 1],
        hspace=0.04, wspace=0.04)

    ax_map   = fig.add_subplot(inner_gs[0, 0])
    ax_right = fig.add_subplot(inner_gs[0, 1], sharey=ax_map)
    ax_bot   = fig.add_subplot(inner_gs[1, 0], sharex=ax_map)
    ax_corner = fig.add_subplot(inner_gs[1, 1])
    ax_corner.set_axis_off()

    _setup_map(ax_map)
    plt.setp(ax_map.get_xticklabels(), visible=False)
    ax_map.set_title(label, fontsize=8, fontweight='bold')

    ax_right.set_xlim(0, DEPTH_MAX)
    ax_right.tick_params(labelsize=5)
    ax_right.set_xlabel('Depth (km)', fontsize=6)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    plt.setp(ax_right.get_xticklabels(), visible=False)

    ax_bot.set_ylim(DEPTH_MAX, 0)
    ax_bot.tick_params(labelsize=5)
    ax_bot.set_xlabel('Longitude', fontsize=6)
    ax_bot.set_ylabel('Depth (km)', fontsize=6)
    ax_bot.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    # tidal rank offset within this group
    rank_offset = gi * 5
    for ci, fnum in enumerate(group):
        tidal_rank = rank_offset + ci + 1
        n_total    = size_dict[fnum]
        R_val      = R_dict.get(fnum, np.nan)
        lons, lats, deps = get_family_locs(fnum)
        color = COLORS[ci % len(COLORS)]
        R_str = f'R={R_val:.3f}' if not np.isnan(R_val) else 'R=n/a'
        lbl   = f'F{fnum} #{tidal_rank} {R_str} (n={len(lons)}/{n_total})'
        kw    = dict(s=MARKER_SIZE, color=color, zorder=4,
                     edgecolors='none', alpha=0.75)
        if len(lons) > 0:
            ax_map.scatter(lons, lats,   **kw, label=lbl)
            ax_right.scatter(deps, lats, **kw)
            ax_bot.scatter(lons, deps,   **kw)
        else:
            ax_map.scatter([], [], s=MARKER_SIZE, color=color, label=lbl)

    ax_map.legend(fontsize=4.5, loc='lower left', framealpha=0.85,
                  markerscale=2, handlelength=1)

fig.suptitle(
    'Axial Seamount — Top 20 Families (by size), Re-ranked by Tidal Sensitivity (Rayleigh R)\n'
    '(#N = tidal rank, R = Rayleigh mean resultant length; higher R = more tidally triggered)',
    fontsize=10)

out_path = f'{OUT_DIR}/15_tidal_rank_map.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

d.close()
print(f'Saved: {out_path}')
