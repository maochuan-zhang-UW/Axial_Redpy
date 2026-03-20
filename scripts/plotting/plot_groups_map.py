"""
Plot top 20 families (by event count) in 4 groups, each as a 3-panel
cross-section: map (lon vs lat), right (depth vs lat), bottom (lon vs depth).

Generates three figures:
  1. Full catalog (all time)
  2. 2015-01-22 to 2015-04-24 08:00 UTC  (pre-eruption)
  3. 2015-04-24 08:00 to 2015-05-16 UTC  (co-eruption)
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from datetime import timezone, datetime

import redpy
from redpy.outputs.mapping import _CALDERA_RIM_LON, _CALDERA_RIM_LAT, _STATIONS

matplotlib.use('agg')

CONFIG    = 'axial_settings.cfg'
CAT_DD    = 'axial_catalog_dd.csv'
CAT_FELIX = 'axial_catalog_felix_2022_2025.csv'
MARKER_SIZE = 4
DEPTH_MAX   = 2.5

# ── Load combined catalog ─────────────────────────────────────────────────────
cat = pd.concat([pd.read_csv(CAT_DD), pd.read_csv(CAT_FELIX)],
                ignore_index=True)
cat['Time'] = pd.to_datetime(cat['Time'], utc=True)
cat_times = cat['Time'].values.astype('datetime64[ns]').astype(float) / 1e9

# ── Rank families by event count ──────────────────────────────────────────────
d = redpy.Detector(CONFIG)
d.open()

sizes = [(fnum, len(d.get_members(fnum))) for fnum in range(len(d))]
sizes.sort(key=lambda x: -x[1])
top20       = [fnum for fnum, _ in sizes[:20]]
groups      = [top20[0:5], top20[5:10], top20[10:15], top20[15:20]]
group_labels = ['Rank 1–5', 'Rank 6–10', 'Rank 11–15', 'Rank 16–20']
COLORS = plt.get_cmap('tab10').colors


def get_family_locs(fnum, t_start=None, t_end=None):
    """
    Return (lons, lats, depths) matched to catalog within 60 s.
    Optionally filter to events between t_start and t_end (Unix seconds).
    """
    members    = d.get_members(fnum)
    rtimes_mpl = d.get('rtable', 'startTimeMPL')
    lons, lats, deps = [], [], []
    for m in members:
        t_sec = mdates.num2date(rtimes_mpl[m]).timestamp()
        if t_start is not None and t_sec < t_start:
            continue
        if t_end is not None and t_sec >= t_end:
            continue
        diffs = np.abs(cat_times - t_sec)
        idx = np.argmin(diffs)
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


def make_figure(title, out_png, t_start=None, t_end=None):
    fig = plt.figure(figsize=(16, 14))
    outer_gs = GridSpec(2, 2, figure=fig, hspace=0.18, wspace=0.18)

    for gi, (group, label) in enumerate(zip(groups, group_labels)):
        row, col = gi // 2, gi % 2
        inner_gs = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer_gs[row, col],
            width_ratios=[3, 1], height_ratios=[3, 1],
            hspace=0.04, wspace=0.04)

        ax_map    = fig.add_subplot(inner_gs[0, 0])
        ax_right  = fig.add_subplot(inner_gs[0, 1], sharey=ax_map)
        ax_bot    = fig.add_subplot(inner_gs[1, 0], sharex=ax_map)
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
        ax_bot.xaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%.2f'))

        rank_start = gi * 5
        for ci, fnum in enumerate(group):
            n_total = sizes[rank_start + ci][1]
            lons, lats, deps = get_family_locs(fnum, t_start, t_end)
            color = COLORS[ci % len(COLORS)]
            lbl = f'F{fnum} (n={len(lons)}/{n_total})'
            if len(lons) > 0:
                kw = dict(s=MARKER_SIZE, color=color, zorder=4,
                          edgecolors='none', alpha=0.75)
                ax_map.scatter(lons, lats, **kw, label=lbl)
                ax_right.scatter(deps, lats, **kw)
                ax_bot.scatter(lons, deps,  **kw)
            else:
                ax_map.scatter([], [], s=MARKER_SIZE, color=color,
                               label=lbl)  # keep legend entry

        ax_map.legend(fontsize=4.5, loc='lower left', framealpha=0.8,
                      markerscale=2, handlelength=1)

    fig.suptitle(title, fontsize=11)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_png}')


# ── UTC timestamps for time windows ──────────────────────────────────────────
def utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute,
                    tzinfo=timezone.utc).timestamp()

T_START_ALL   = None
T_END_ALL     = None

T_START_PRE   = utc(2015, 1, 22)
T_END_PRE     = utc(2015, 4, 24, 8, 0)   # Apr 24 08:00 UTC

T_START_CO    = utc(2015, 4, 24, 8, 0)   # Apr 24 08:00 UTC
T_END_CO      = utc(2015, 5, 17)          # through May 16

T_START_POST  = utc(2015, 5, 17)          # May 16 end → catalog end
T_END_POST    = None

# ── Generate figures ──────────────────────────────────────────────────────────
print('Generating full-catalog figure...')
make_figure(
    'Axial Seamount — Top 20 Families  |  All time\n'
    '(ranked by event count, colored by family)',
    'runs/axial/top20_groups_map.png',
    T_START_ALL, T_END_ALL)

print('Generating pre-eruption figure (Jan 22 – Apr 24 08:00 UTC 2015)...')
make_figure(
    'Axial Seamount — Top 20 Families  |  Pre-eruption: 2015-01-22 → 2015-04-24 08:00 UTC\n'
    '(ranked by event count, n = events in window / total)',
    'runs/axial/top20_groups_map_pre_eruption.png',
    T_START_PRE, T_END_PRE)

print('Generating co-eruption figure (Apr 24 08:00 – May 16 2015)...')
make_figure(
    'Axial Seamount — Top 20 Families  |  Co-eruption: 2015-04-24 08:00 UTC → 2015-05-16\n'
    '(ranked by event count, n = events in window / total)',
    'runs/axial/top20_groups_map_co_eruption.png',
    T_START_CO, T_END_CO)

print('Generating post-eruption figure (May 16 2015 → end)...')
make_figure(
    'Axial Seamount — Top 20 Families  |  Post-eruption: 2015-05-16 → 2025-03-07\n'
    '(ranked by event count, n = events in window / total)',
    'runs/axial/top20_groups_map_post_eruption.png',
    T_START_POST, T_END_POST)

d.close()
print('All done.')
