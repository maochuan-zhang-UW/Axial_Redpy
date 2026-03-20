"""
Fig 17 — Post-eruption R recovery.
From May 17, 2015 to end of tide data (Dec 2021).
Narrower 30-day windows, 7-day step to resolve the recovery curve clearly.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime, timezone
from scipy.signal import hilbert
import redpy

matplotlib.use('agg')

CONFIG   = 'axial_settings.cfg'
OUT_DIR  = 'runs/axial/eruption_analysis'
TIDE_DIR = os.path.expanduser('~/Documents/TidesArchive/Tides_Central_caldera')
YEARS    = list(range(2015, 2022))
DOWNSAMPLE = 240

ERU_END  = datetime(2015, 5, 19, tzinfo=timezone.utc).timestamp()
WIN_DAYS_LIST = [30, 60, 90]   # three curves for comparison
STEP_DAYS = 7
MIN_N     = 20

# ── Tide ─────────────────────────────────────────────────────────────────────
print('Loading tide...')
chunks = []
for yr in YEARS:
    df = pd.read_csv(os.path.join(TIDE_DIR, f'pred_F_{yr}.txt'),
                     sep=r'\s+', header=None,
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
analytic = hilbert(tide_levels - tide_levels.mean())
t_min, t_max = tide_times[0], tide_times[-1]

def get_phase(t_arr):
    ar = np.interp(t_arr, tide_times, np.real(analytic))
    ai = np.interp(t_arr, tide_times, np.imag(analytic))
    ph = np.angle(ar + 1j*ai) + np.pi
    return np.degrees((ph + np.pi) % (2*np.pi) - np.pi)

def rayleigh_R(phases_deg):
    if len(phases_deg) < MIN_N:
        return np.nan
    phi = np.radians(phases_deg)
    n   = len(phi)
    C   = np.sum(np.cos(phi)); S = np.sum(np.sin(phi))
    return float(np.sqrt(C**2 + S**2) / n)

# ── Full earthquake catalog (DD + Felix) ─────────────────────────────────────
print('Loading earthquake catalog...')
cat = pd.concat([pd.read_csv('axial_catalog_dd.csv'),
                 pd.read_csv('axial_catalog_felix_2022_2025.csv')],
                ignore_index=True)
cat['Time'] = pd.to_datetime(cat['Time'], utc=True)
eq_sec = cat['Time'].values.astype('datetime64[ns]').astype(float) / 1e9

# Keep only post-eruption + within tide window
mask = (eq_sec >= ERU_END) & (eq_sec <= t_max)
eq_post = eq_sec[mask]
print(f'All catalog events after May 19 2015 (within tide window): {len(eq_post):,}')

# ── Sliding window for each window size ──────────────────────────────────────
step = STEP_DAYS * 86400
colors_w = ['#1565C0', '#2E7D32', '#E65100']
labels_w  = ['30-day window', '60-day window', '90-day window']

all_centers = {}
all_R       = {}

for win_days in WIN_DAYS_LIST:
    win = win_days * 86400
    centers, R_vals, n_vals = [], [], []
    t = ERU_END + win / 2
    while t <= t_max - win / 2:
        idx = np.where((eq_post >= t - win/2) & (eq_post < t + win/2))[0]
        ph  = get_phase(eq_post[idx]) if len(idx) >= MIN_N else np.array([])
        centers.append(t)
        R_vals.append(rayleigh_R(ph))
        n_vals.append(len(idx))
        t += step
    all_centers[win_days] = np.array(centers)
    all_R[win_days]       = np.array(R_vals, dtype=float)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06})

ax_R = axes[0]
ax_n = axes[1]

# Pre-eruption R reference line
ax_R.axhline(0.390, color='#1A3A5C', lw=1.2, ls=':', alpha=0.7,
             label='Pre-eruption R = 0.390 (reference)')
ax_R.axhline(0.103, color='red', lw=1.0, ls=':', alpha=0.5,
             label='Co-eruption R = 0.103')

for win_days, color, label in zip(WIN_DAYS_LIST, colors_w, labels_w):
    centers_mpl = np.array([
        mdates.date2num(datetime.fromtimestamp(t, tz=timezone.utc))
        for t in all_centers[win_days]
    ])
    R_vals = all_R[win_days]
    valid  = ~np.isnan(R_vals)
    ax_R.plot(centers_mpl[valid], R_vals[valid],
              'o-', ms=3, lw=1.4, color=color, label=label, alpha=0.85)

# Shaded "recovery zone" between co-eruption R and pre-eruption R
ax_R.axhspan(0.103, 0.390, color='#FFF9C4', alpha=0.5, zorder=0,
             label='Recovery zone')

ax_R.set_ylabel('Rayleigh R', fontsize=11)
ax_R.set_ylim(0, 0.65)
ax_R.legend(fontsize=8, loc='upper right', ncol=2)
ax_R.set_title(
    'Post-Eruption Recovery of Tidal Sensitivity (Rayleigh R)\n'
    'Axial Seamount  |  May 17, 2015 → December 2021  '
    '(0° = low tide = peak extensional stress)',
    fontsize=11)

# Event count (60-day for reference)
centers_mpl_60 = np.array([
    mdates.date2num(datetime.fromtimestamp(t, tz=timezone.utc))
    for t in all_centers[60]
])
# raw count per 7-day step bin from post-eruption catalog
bin_edges = np.arange(ERU_END, t_max + step, step)
counts, _ = np.histogram(eq_post, bins=bin_edges)
bin_mpl   = np.array([
    mdates.date2num(datetime.fromtimestamp(0.5*(bin_edges[i]+bin_edges[i+1]),
                                           tz=timezone.utc))
    for i in range(len(counts))
])
ax_n.bar(bin_mpl, counts, width=STEP_DAYS * 0.6,
         color='#78909C', alpha=0.7, label='Events per 7-day bin')
ax_n.set_ylabel('Events / 7 days', fontsize=9)
ax_n.legend(fontsize=8, loc='upper right')

# Year formatter
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
plt.setp(axes[-1].get_xticklabels(), rotation=30, ha='right', fontsize=9)
axes[-1].set_xlabel('Date', fontsize=11)

plt.tight_layout()
out_path = f'{OUT_DIR}/17_R_recovery.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')
