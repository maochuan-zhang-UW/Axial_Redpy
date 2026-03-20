"""Figs 08-12 only (re-run after fixing subset_matrix return_type)."""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from datetime import timezone, datetime

import redpy
import redpy.correlation

matplotlib.use('agg')

CONFIG    = 'axial_settings.cfg'
CAT_DD    = 'axial_catalog_dd.csv'
CAT_FELIX = 'axial_catalog_felix_2022_2025.csv'
OUT_DIR   = 'runs/axial/eruption_analysis'

ERU_START_MPL = mdates.date2num(datetime(2015, 4, 24, 8, 0, tzinfo=timezone.utc))
ERU_END_MPL   = mdates.date2num(datetime(2015, 5, 17,      tzinfo=timezone.utc))

def _eruption_lines(ax):
    ax.axvline(ERU_START_MPL, color='red',    lw=1.2, ls='--', alpha=0.8,
               zorder=10, label='Eruption onset')
    ax.axvline(ERU_END_MPL,   color='orange', lw=1.2, ls='--', alpha=0.8,
               zorder=10, label='Co-erup. end')

cat = pd.concat([pd.read_csv(CAT_DD), pd.read_csv(CAT_FELIX)], ignore_index=True)
cat['Time'] = pd.to_datetime(cat['Time'], utc=True)
cat_times_sec = cat['Time'].values.astype('datetime64[ns]').astype(float) / 1e9

d = redpy.Detector(CONFIG)
d.open()

rtimes_mpl = d.get('rtable', 'startTimeMPL')
rtimes_sec = np.array([mdates.num2date(t).timestamp() for t in rtimes_mpl])
nfam = len(d)
t0 = rtimes_mpl.min(); t1 = rtimes_mpl.max()

sizes = sorted([(fnum, len(d.get_members(fnum))) for fnum in range(nfam)],
               key=lambda x: -x[1])
top20_fnums = [f for f, _ in sizes[:20]]
top5_fnums  = top20_fnums[:5]
COLORS = plt.get_cmap('tab10').colors

def members_of(fnum):
    return d.get_members(fnum)

def match_catalog(t_sec_array):
    idxs = []
    for t in t_sec_array:
        diffs = np.abs(cat_times_sec - t)
        i = np.argmin(diffs)
        idxs.append(i if diffs[i] < 60 else -1)
    return np.array(idxs)

# ── Fig 08 — CC matrix heatmap ────────────────────────────────────────────────
print('08 CC matrix heatmap...')
ids_all, ccc_sparse = d.get_matrix()

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ci, fnum in enumerate(top5_fnums):
    mems = members_of(fnum)
    MAX_SHOW = 300
    if len(mems) > MAX_SHOW:
        step = max(1, len(mems) // MAX_SHOW)
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

# ── Fig 09 — Amplitude timeline ───────────────────────────────────────────────
print('09 Amplitude timeline...')
printsta = d.get('printsta')
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
for ci, fnum in enumerate(top5_fnums):
    mems = members_of(fnum)
    amps  = d.get('rtable', 'windowAmp')[mems, printsta]
    t_mpl = rtimes_mpl[mems]
    order = np.argsort(t_mpl)
    ax = axes[ci]
    ax.plot(t_mpl[order], amps[order], '.', ms=2, color=COLORS[ci], alpha=0.6)
    ax.set_yscale('log')
    ax.set_ylabel('Amplitude\n(counts)', fontsize=7)
    ax.set_title(f'Family {fnum}  (n={len(mems)})', fontsize=8)
    _eruption_lines(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(axes[-1].get_xticklabels(), rotation=30, ha='right', fontsize=7)
fig.suptitle(f'Amplitude Timeline — Top 5 Families '
             f'(station {d.get("station")[printsta]})', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/09_amplitude_timeline.png', dpi=150)
plt.close()

# ── Fig 10 — Longevity vs size scatter ───────────────────────────────────────
print('10 Longevity vs size scatter...')
longs, nevents, mean_deps = [], [], []
for fnum in range(nfam):
    mems = members_of(fnum)
    if len(mems) < 2:
        continue
    longs.append(float(d.get('ftable', 'longevity', fnum)))
    nevents.append(len(mems))
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
ax.set_yscale('log'); ax.set_xscale('log')
ax.set_title(f'Family Longevity vs Size — All {nfam} Families\n(color = mean depth)', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/10_longevity_vs_size.png', dpi=150)
plt.close()

# ── Fig 11 — New family formation rate ───────────────────────────────────────
print('11 Family formation rate...')
first_times_mpl = []
for fnum in range(nfam):
    mems = members_of(fnum)
    if len(mems) > 0:
        first_times_mpl.append(rtimes_mpl[mems].min())
first_times_mpl = np.array(first_times_mpl)

BIN_DAYS = 30
bins_t = np.arange(first_times_mpl.min(),
                   first_times_mpl.max() + BIN_DAYS, BIN_DAYS)
fig, ax = plt.subplots(figsize=(14, 5))
ax.hist(first_times_mpl, bins=bins_t,
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

# ── Fig 12 — Family birth/death timeline ─────────────────────────────────────
print('12 Family birth/death timeline...')
top50 = [f for f, _ in sizes[:50]]
bar_data = []
for fnum in top50:
    mems = members_of(fnum)
    t_mpl = rtimes_mpl[mems]
    bar_data.append((fnum, t_mpl.min(), t_mpl.max(), len(mems)))
bar_data.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(14, 12))
for yi, (fnum, t_s, t_e, n) in enumerate(bar_data):
    ax.barh(yi, t_e - t_s, left=t_s, height=0.7,
            color=COLORS[top50.index(fnum) % 10], alpha=0.75, edgecolor='none')
    ax.text(t_e + 5, yi, f'F{fnum} (n={n})', va='center', fontsize=5)
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
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/12_family_birth_death.png', dpi=150)
plt.close()

d.close()
print(f'\nFigs 08–12 saved to {OUT_DIR}/')
