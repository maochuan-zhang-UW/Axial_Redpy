"""
Eruption analysis figures for Axial Seamount repeating earthquakes.
Outputs all figures to runs/axial/eruption_analysis/

Figures:
  01  Event rate timeline (events/hour per top family)
  02  Cumulative event count curves
  03  Interevent spacing vs time
  04  Hypocenter migration (cross-section, top 5 families, colored by time)
  05  Depth histogram  pre / co / post eruption
  06  All-event cross-section colored by time
  07  Distance from caldera center vs time
  08  CC matrix heatmap (top 5 families)
  09  Amplitude timeline
  10  Family longevity vs size scatter
  11  New family formation rate histogram
  12  Family birth/death timeline
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from datetime import timezone, datetime, timedelta
from obspy import UTCDateTime

import redpy
import redpy.correlation
from redpy.outputs.mapping import _CALDERA_RIM_LON, _CALDERA_RIM_LAT, _STATIONS

matplotlib.use('agg')

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG   = 'axial_settings.cfg'
CAT_DD   = 'axial_catalog_dd.csv'
CAT_FELIX = 'axial_catalog_felix_2022_2025.csv'
OUT_DIR  = 'runs/axial/eruption_analysis'

# Eruption boundaries (UTC Unix seconds)
def utc(*args):
    return datetime(*args, tzinfo=timezone.utc).timestamp()

T_PRE_START = utc(2015, 1, 22)
T_ERU_START = utc(2015, 4, 24, 8, 0)   # eruption onset
T_ERU_END   = utc(2015, 5, 17)          # end of co-eruption window
# post = T_ERU_END → ∞

ERU_START_MPL = mdates.date2num(datetime(2015, 4, 24, 8, 0, tzinfo=timezone.utc))
ERU_END_MPL   = mdates.date2num(datetime(2015, 5, 17,      tzinfo=timezone.utc))

CALDERA_CENTER = (-130.009, 45.955)   # approximate center lon, lat
TOP_N  = 20
TOP_5  = 5
COLORS = plt.get_cmap('tab10').colors

# ── Load data ─────────────────────────────────────────────────────────────────
cat = pd.concat([pd.read_csv(CAT_DD), pd.read_csv(CAT_FELIX)], ignore_index=True)
cat['Time'] = pd.to_datetime(cat['Time'], utc=True)
cat_times_sec = cat['Time'].values.astype('datetime64[ns]').astype(float) / 1e9

d = redpy.Detector(CONFIG)
d.open()

rtimes_mpl = d.get('rtable', 'startTimeMPL')
rtimes_sec = np.array([mdates.num2date(t).timestamp() for t in rtimes_mpl])
nfam = len(d)

# Rank by size
sizes = sorted([(fnum, len(d.get_members(fnum))) for fnum in range(nfam)],
               key=lambda x: -x[1])
top20_fnums = [f for f, _ in sizes[:TOP_N]]
top5_fnums  = top20_fnums[:TOP_5]

def members_of(fnum):
    return d.get_members(fnum)

def match_catalog(t_sec_array):
    """For each event time, return matched row index or -1."""
    idxs = []
    for t in t_sec_array:
        diffs = np.abs(cat_times_sec - t)
        i = np.argmin(diffs)
        idxs.append(i if diffs[i] < 60 else -1)
    return np.array(idxs)

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def _setup_map(ax, fontsize=6):
    ax.set_xlim(-130.05, -129.94)
    ax.set_ylim(45.91, 46.00)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(labelsize=fontsize-1)
    ax.plot(_CALDERA_RIM_LON, _CALDERA_RIM_LAT, 'k-', lw=0.6, zorder=2)
    for name, (slon, slat) in _STATIONS.items():
        ax.scatter(slon, slat, marker='^', s=12, color='k',
                   facecolors='none', lw=0.6, zorder=5)

def _eruption_lines(ax, vertical=True):
    kw = dict(color='red', lw=1.2, ls='--', alpha=0.8, zorder=10)
    if vertical:
        ax.axvline(ERU_START_MPL, label='Eruption onset', **kw)
        ax.axvline(ERU_END_MPL,   color='orange', lw=1.2, ls='--',
                   alpha=0.8, zorder=10, label='Co-erup. end')
    else:
        ax.axhline(ERU_START_MPL, **kw)
        ax.axhline(ERU_END_MPL, color='orange', lw=1.2, ls='--', alpha=0.8, zorder=10)

print('Starting eruption analysis figures...')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 01 — Event rate timeline (events/hour binned, top 5 families)
# ─────────────────────────────────────────────────────────────────────────────
print('01 Event rate timeline...')
fig, ax = plt.subplots(figsize=(14, 5))
BIN_HOURS = 24
all_t = rtimes_mpl
t0 = all_t.min();  t1 = all_t.max()
bins = np.arange(t0, t1 + BIN_HOURS/24, BIN_HOURS/24)

for ci, fnum in enumerate(top5_fnums):
    mems = members_of(fnum)
    t_mpl = rtimes_mpl[mems]
    counts, edges = np.histogram(t_mpl, bins=bins)
    rate = counts / BIN_HOURS   # events per hour
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot_date(centers, rate, '-', lw=0.8, color=COLORS[ci],
                 label=f'F{fnum} (n={len(mems)})', alpha=0.85)

_eruption_lines(ax)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
ax.set_ylabel('Events / hour', fontsize=9)
ax.set_title(f'Event Rate (top 5 families, {BIN_HOURS}-h bins)', fontsize=10)
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(t0, t1)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_event_rate_timeline.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 02 — Cumulative event count curves (top 10 families)
# ─────────────────────────────────────────────────────────────────────────────
print('02 Cumulative event count...')
fig, ax = plt.subplots(figsize=(14, 6))
for ci, fnum in enumerate(top20_fnums[:10]):
    mems = members_of(fnum)
    t_sorted = np.sort(rtimes_mpl[mems])
    ax.plot_date(t_sorted, np.arange(1, len(t_sorted)+1), '-',
                 lw=0.9, color=COLORS[ci % 10],
                 label=f'F{fnum} (n={len(mems)})', alpha=0.85)
_eruption_lines(ax)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
ax.set_ylabel('Cumulative events', fontsize=9)
ax.set_title('Cumulative Event Count (top 10 families)', fontsize=10)
ax.legend(fontsize=6, loc='upper left', ncol=2)
ax.set_xlim(t0, t1)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_cumulative_events.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 03 — Interevent spacing vs time (top 5 families)
# ─────────────────────────────────────────────────────────────────────────────
print('03 Interevent spacing...')
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
for ci, fnum in enumerate(top5_fnums):
    mems = members_of(fnum)
    t_sorted = np.sort(rtimes_mpl[mems])
    spacing_h = np.diff(t_sorted) * 24   # days → hours
    ax = axes[ci]
    ax.plot_date(t_sorted[1:], spacing_h, '.', ms=2,
                 color=COLORS[ci], alpha=0.6)
    ax.set_yscale('log')
    ax.set_ylabel('Spacing (h)', fontsize=7)
    ax.set_title(f'Family {fnum}  (n={len(mems)})', fontsize=8)
    _eruption_lines(ax)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
plt.setp(axes[-1].get_xticklabels(), rotation=30, ha='right', fontsize=7)
fig.suptitle('Interevent Spacing vs Time (top 5 families)', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_interevent_spacing.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 04 — Hypocenter migration (top 5 families, cross-section colored by time)
# ─────────────────────────────────────────────────────────────────────────────
print('04 Hypocenter migration...')
fig = plt.figure(figsize=(18, 14))
outer = GridSpec(3, 2, figure=fig, hspace=0.2, wspace=0.15)

for ci, fnum in enumerate(top5_fnums):
    row, col = ci // 2, ci % 2
    if ci == 4:  # last one: span bottom center
        inner = GridSpecFromSubplotSpec(2, 2,
            subplot_spec=outer[2, :],
            width_ratios=[3,1], height_ratios=[3,1], hspace=0.04, wspace=0.04)
    else:
        inner = GridSpecFromSubplotSpec(2, 2,
            subplot_spec=outer[row, col],
            width_ratios=[3,1], height_ratios=[3,1], hspace=0.04, wspace=0.04)

    ax_map   = fig.add_subplot(inner[0, 0])
    ax_right = fig.add_subplot(inner[0, 1], sharey=ax_map)
    ax_bot   = fig.add_subplot(inner[1, 0], sharex=ax_map)
    ax_cb    = fig.add_subplot(inner[1, 1])
    ax_cb.set_axis_off()

    mems = members_of(fnum)
    t_sec_arr = rtimes_sec[mems]
    cat_idxs  = match_catalog(t_sec_arr)
    mask = cat_idxs >= 0
    if mask.sum() < 1:
        continue
    lons  = cat.iloc[cat_idxs[mask]]['Longitude'].values.astype(float)
    lats  = cat.iloc[cat_idxs[mask]]['Latitude'].values.astype(float)
    deps  = np.clip(cat.iloc[cat_idxs[mask]]['Depth'].values.astype(float), 0, 2.5)
    times = t_sec_arr[mask]

    norm = mcolors.Normalize(vmin=times.min(), vmax=times.max())
    cmap_t = cm.plasma

    _setup_map(ax_map)
    plt.setp(ax_map.get_xticklabels(), visible=False)
    ax_map.set_ylabel('Latitude', fontsize=6)
    ax_map.set_title(f'Family {fnum}  (n={mask.sum()})', fontsize=8)

    for ax_panel, xs, ys in [(ax_map, lons, lats),
                              (ax_right, deps, lats),
                              (ax_bot, lons, deps)]:
        ax_panel.scatter(xs, ys, s=5, c=times, cmap=cmap_t, norm=norm,
                         zorder=4, edgecolors='none', alpha=0.8)

    ax_right.set_xlim(0, 2.5); plt.setp(ax_right.get_yticklabels(), visible=False)
    plt.setp(ax_right.get_xticklabels(), visible=False)
    ax_bot.set_ylim(2.5, 0)
    ax_bot.set_ylabel('Depth (km)', fontsize=6)
    ax_bot.set_xlabel('Longitude', fontsize=6)
    ax_bot.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax_bot.tick_params(labelsize=5)

    sm = cm.ScalarMappable(cmap=cmap_t, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_cb, fraction=0.6, pad=0.05, aspect=12)
    cb.set_label('Date', fontsize=6)
    ticks = np.linspace(times.min(), times.max(), 4)
    cb.set_ticks(ticks)
    cb.set_ticklabels([datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m')
                       for t in ticks], fontsize=5)

fig.suptitle('Hypocenter Migration — Top 5 Families (colored by time)', fontsize=11)
plt.savefig(f'{OUT_DIR}/04_hypocenter_migration.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 05 — Depth histogram  pre / co / post eruption (all top 20 families)
# ─────────────────────────────────────────────────────────────────────────────
print('05 Depth histogram...')
bins_d = np.linspace(0, 2.5, 26)
labels_p = ['Pre-eruption\n(< Apr 24 2015)', 'Co-eruption\n(Apr 24–May 16 2015)',
            'Post-eruption\n(> May 16 2015)']
colors_p = ['steelblue', 'tomato', 'seagreen']

all_deps = {'pre': [], 'co': [], 'post': []}

for fnum in top20_fnums:
    mems = members_of(fnum)
    t_sec_arr = rtimes_sec[mems]
    cat_idxs  = match_catalog(t_sec_arr)
    mask = cat_idxs >= 0
    if not mask.any():
        continue
    deps  = np.clip(cat.iloc[cat_idxs[mask]]['Depth'].values.astype(float), 0, 2.5)
    times = t_sec_arr[mask]
    all_deps['pre'].extend( deps[times < T_ERU_START])
    all_deps['co'].extend(  deps[(times >= T_ERU_START) & (times < T_ERU_END)])
    all_deps['post'].extend(deps[times >= T_ERU_END])

fig, ax = plt.subplots(figsize=(8, 6))
for key, label, color in zip(['pre','co','post'], labels_p, colors_p):
    ax.hist(all_deps[key], bins=bins_d, orientation='horizontal',
            alpha=0.6, color=color, label=f'{label}  (n={len(all_deps[key])})',
            density=True)
ax.set_xlabel('Density', fontsize=9)
ax.set_ylabel('Depth (km)', fontsize=9)
ax.invert_yaxis()
ax.set_title('Depth Distribution — Pre / Co / Post Eruption\n(top 20 families)', fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_depth_histogram.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 06 — All-event cross-section colored by time (top 20 families combined)
# ─────────────────────────────────────────────────────────────────────────────
print('06 Cross-section colored by time...')
all_lons, all_lats, all_deps_t, all_times = [], [], [], []
for fnum in top20_fnums:
    mems = members_of(fnum)
    t_sec_arr = rtimes_sec[mems]
    cat_idxs  = match_catalog(t_sec_arr)
    mask = cat_idxs >= 0
    if not mask.any():
        continue
    rows = cat.iloc[cat_idxs[mask]]
    all_lons.extend(rows['Longitude'].values.astype(float))
    all_lats.extend(rows['Latitude'].values.astype(float))
    all_deps_t.extend(np.clip(rows['Depth'].values.astype(float), 0, 2.5))
    all_times.extend(t_sec_arr[mask])

all_lons = np.array(all_lons); all_lats = np.array(all_lats)
all_deps_t = np.array(all_deps_t); all_times = np.array(all_times)
sort_idx = np.argsort(all_times)
all_lons = all_lons[sort_idx]; all_lats = all_lats[sort_idx]
all_deps_t = all_deps_t[sort_idx]; all_times = all_times[sort_idx]

norm_t = mcolors.Normalize(vmin=all_times.min(), vmax=all_times.max())
cmap_t = cm.plasma

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig, width_ratios=[3,1], height_ratios=[3,1],
              hspace=0.04, wspace=0.04)
ax_map   = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1], sharey=ax_map)
ax_bot   = fig.add_subplot(gs[1, 0], sharex=ax_map)
ax_cb    = fig.add_subplot(gs[1, 1])
ax_cb.set_axis_off()

_setup_map(ax_map)
plt.setp(ax_map.get_xticklabels(), visible=False)
ax_map.set_ylabel('Latitude', fontsize=8)

for ax_p, xs, ys in [(ax_map, all_lons, all_lats),
                     (ax_right, all_deps_t, all_lats),
                     (ax_bot, all_lons, all_deps_t)]:
    ax_p.scatter(xs, ys, s=3, c=all_times, cmap=cmap_t, norm=norm_t,
                 zorder=4, edgecolors='none', alpha=0.6)

ax_right.set_xlim(0, 2.5); plt.setp(ax_right.get_yticklabels(), visible=False)
plt.setp(ax_right.get_xticklabels(), visible=False); ax_right.set_xlabel('Depth (km)', fontsize=7)
ax_bot.set_ylim(2.5, 0); ax_bot.set_ylabel('Depth (km)', fontsize=8)
ax_bot.set_xlabel('Longitude', fontsize=8)
ax_bot.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
ax_bot.tick_params(labelsize=6)

sm = cm.ScalarMappable(cmap=cmap_t, norm=norm_t)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax_cb, fraction=0.6, pad=0.05, aspect=12)
cb.set_label('Date', fontsize=8)
ticks = np.linspace(all_times.min(), all_times.max(), 6)
cb.set_ticks(ticks)
cb.set_ticklabels([datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m')
                   for t in ticks], fontsize=6)

fig.suptitle('All Events (top 20 families) — Cross-section Colored by Time', fontsize=10)
plt.savefig(f'{OUT_DIR}/06_crosssection_colored_by_time.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 07 — Distance from caldera center vs time (top 10 families)
# ─────────────────────────────────────────────────────────────────────────────
print('07 Distance from caldera center...')
fig, ax = plt.subplots(figsize=(14, 6))
for ci, fnum in enumerate(top20_fnums[:10]):
    mems = members_of(fnum)
    t_sec_arr = rtimes_sec[mems]
    cat_idxs  = match_catalog(t_sec_arr)
    mask = cat_idxs >= 0
    if not mask.any():
        continue
    rows = cat.iloc[cat_idxs[mask]]
    dists = haversine_km(rows['Longitude'].values.astype(float),
                         rows['Latitude'].values.astype(float),
                         CALDERA_CENTER[0], CALDERA_CENTER[1])
    t_mpl = rtimes_mpl[mems][mask]
    ax.plot_date(t_mpl, dists, '.', ms=2, color=COLORS[ci % 10],
                 alpha=0.5, label=f'F{fnum}')

_eruption_lines(ax)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
ax.set_ylabel('Distance from caldera center (km)', fontsize=9)
ax.set_title('Distance from Caldera Center vs Time (top 10 families)', fontsize=10)
ax.legend(fontsize=6, loc='upper right', ncol=2)
ax.set_xlim(t0, t1)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/07_distance_from_center.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 08 — CC matrix heatmap (top 5 families)
# ─────────────────────────────────────────────────────────────────────────────
print('08 CC matrix heatmap...')
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
plotvars = d.get('plotvars')
ids_all   = plotvars['ids']
ccc_sparse = plotvars['ccc_sparse']

for ci, fnum in enumerate(top5_fnums):
    mems = members_of(fnum)
    # subsample if too large
    MAX_SHOW = 300
    if len(mems) > MAX_SHOW:
        step = len(mems) // MAX_SHOW
        mems_sub = mems[::step][:MAX_SHOW]
    else:
        mems_sub = mems

    member_ids = ids_all[mems_sub]
    ccc_dense  = redpy.correlation.subset_matrix(member_ids, ccc_sparse,
                                                 return_type='matrix')
    ax = axes[ci]
    cmap_cc = plt.get_cmap('inferno_r').copy()
    cmap_cc.set_extremes(under='w')
    im = ax.imshow(ccc_dense, vmin=0.7, vmax=1.0, cmap=cmap_cc, aspect='auto')
    ax.set_title(f'F{fnum}  (n={len(mems_sub)})', fontsize=8)
    ax.set_xlabel('Event index', fontsize=7)
    if ci == 0:
        ax.set_ylabel('Event index', fontsize=7)
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)

fig.suptitle('CC Matrix Heatmap — Top 5 Families', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/08_cc_matrix.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 09 — Amplitude timeline (top 5 families)
# ─────────────────────────────────────────────────────────────────────────────
print('09 Amplitude timeline...')
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
printsta = d.get('printsta')
for ci, fnum in enumerate(top5_fnums):
    mems = members_of(fnum)
    amps = d.get('rtable', 'windowAmp')[mems, printsta]
    t_mpl = rtimes_mpl[mems]
    order = np.argsort(t_mpl)
    ax = axes[ci]
    ax.plot_date(t_mpl[order], amps[order], '.', ms=2,
                 color=COLORS[ci], alpha=0.6)
    ax.set_yscale('log')
    ax.set_ylabel('Amplitude\n(counts)', fontsize=7)
    ax.set_title(f'Family {fnum}  (n={len(mems)})', fontsize=8)
    _eruption_lines(ax)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
plt.setp(axes[-1].get_xticklabels(), rotation=30, ha='right', fontsize=7)
fig.suptitle(f'Amplitude Timeline — Top 5 Families '
             f'(station {d.get("station")[printsta]})', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/09_amplitude_timeline.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 — Family longevity vs size scatter
# ─────────────────────────────────────────────────────────────────────────────
print('10 Longevity vs size scatter...')
longs, nevents, mean_deps = [], [], []
for fnum in range(nfam):
    mems = members_of(fnum)
    if len(mems) < 2:
        continue
    longevity = d.get('ftable', 'longevity', fnum)
    longs.append(float(longevity))
    nevents.append(len(mems))

    # mean depth from catalog
    t_sec_arr = rtimes_sec[mems]
    cat_idxs  = match_catalog(t_sec_arr[:min(50, len(mems))])
    mask = cat_idxs >= 0
    if mask.any():
        dep_vals = np.clip(cat.iloc[cat_idxs[mask]]['Depth'].values.astype(float), 0, 2.5)
        mean_deps.append(float(dep_vals.mean()))
    else:
        mean_deps.append(np.nan)

longs = np.array(longs); nevents = np.array(nevents); mean_deps = np.array(mean_deps)

fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(longs, nevents, c=mean_deps, cmap='viridis_r',
                vmin=0, vmax=2.5, s=20, alpha=0.7, edgecolors='k', lw=0.3)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label('Mean depth (km)', fontsize=9)
ax.set_xlabel('Longevity (days)', fontsize=9)
ax.set_ylabel('Number of events', fontsize=9)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title(f'Family Longevity vs Size — All {nfam} Families\n'
             '(color = mean depth)', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/10_longevity_vs_size.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 11 — New family formation rate (when did each family's first event occur)
# ─────────────────────────────────────────────────────────────────────────────
print('11 Family formation rate...')
first_times_mpl = []
for fnum in range(nfam):
    mems = members_of(fnum)
    if len(mems) > 0:
        first_times_mpl.append(rtimes_mpl[mems].min())

first_times_mpl = np.array(first_times_mpl)
BIN_DAYS = 30
bins_t = np.arange(first_times_mpl.min(), first_times_mpl.max() + BIN_DAYS, BIN_DAYS)

fig, ax = plt.subplots(figsize=(14, 5))
counts, edges, _ = ax.hist(first_times_mpl, bins=bins_t,
                           color='steelblue', alpha=0.8, edgecolor='white', lw=0.3)
_eruption_lines(ax)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
ax.set_ylabel('New families formed', fontsize=9)
ax.set_title(f'New Family Formation Rate ({BIN_DAYS}-day bins)', fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(first_times_mpl.min(), first_times_mpl.max())
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/11_family_formation_rate.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 12 — Family birth/death timeline (top 50 families by size)
# ─────────────────────────────────────────────────────────────────────────────
print('12 Family birth/death timeline...')
top50 = [f for f, _ in sizes[:50]]
bar_data = []
for fnum in top50:
    mems = members_of(fnum)
    t_mpl = rtimes_mpl[mems]
    bar_data.append((fnum, t_mpl.min(), t_mpl.max(), len(mems)))
bar_data.sort(key=lambda x: x[1])  # sort by first event

fig, ax = plt.subplots(figsize=(14, 12))
for yi, (fnum, t_start_b, t_end_b, n) in enumerate(bar_data):
    ax.barh(yi, t_end_b - t_start_b, left=t_start_b, height=0.7,
            color=COLORS[top50.index(fnum) % 10], alpha=0.75, edgecolor='none')
    ax.text(t_end_b + 5, yi, f'F{fnum} (n={n})', va='center', fontsize=5)

_eruption_lines(ax)
ax.set_yticks(range(len(bar_data)))
ax.set_yticklabels([f'F{f}' for f, *_ in bar_data], fontsize=5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
ax.set_xlabel('Date', fontsize=9)
ax.set_title('Family Activity Lifespan — Top 50 Families\n'
             '(red dashed = eruption onset, orange = co-eruption end)', fontsize=10)
ax.set_xlim(t0 - 30, t1 + 200)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/12_family_birth_death.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
d.close()
print(f'\nAll 12 figures saved to {OUT_DIR}/')
