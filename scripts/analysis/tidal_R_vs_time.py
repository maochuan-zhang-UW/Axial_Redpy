"""
Rayleigh R through time — sliding window tidal sensitivity.
Shows how tidal triggering strength evolves before, during, and after
the 2015 Axial Seamount eruption.
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
from scipy.stats import chi2

import redpy

matplotlib.use('agg')

CONFIG   = 'axial_settings.cfg'
OUT_DIR  = 'runs/axial/eruption_analysis'
TIDE_DIR = os.path.expanduser('~/Documents/TidesArchive/Tides_Central_caldera')
YEARS    = list(range(2015, 2022))
DOWNSAMPLE = 240          # 15 s × 240 = 1 h

ERU_START = datetime(2015, 4, 24, 8, 0, tzinfo=timezone.utc).timestamp()
ERU_END   = datetime(2015, 5, 17,    tzinfo=timezone.utc).timestamp()

WIN_DAYS  = 60            # sliding window length
STEP_DAYS = 15            # step size
MIN_N     = 30            # minimum events to compute R

# ── 1. Tide ───────────────────────────────────────────────────────────────────
print('Loading tide data...')
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
analytic    = hilbert(tide_levels - tide_levels.mean())
t_min, t_max = tide_times[0], tide_times[-1]

def get_phase(t_arr):
    ar = np.interp(t_arr, tide_times, np.real(analytic))
    ai = np.interp(t_arr, tide_times, np.imag(analytic))
    ph = np.angle(ar + 1j*ai) + np.pi
    return np.degrees((ph + np.pi) % (2*np.pi) - np.pi)

def rayleigh(phases_deg):
    if len(phases_deg) < MIN_N:
        return np.nan, np.nan, np.nan
    phi = np.radians(phases_deg)
    n   = len(phi)
    C   = np.sum(np.cos(phi)); S = np.sum(np.sin(phi))
    R   = np.sqrt(C**2 + S**2) / n
    z   = n * R**2
    # p-value via chi2 approximation (more stable for large n)
    p   = float(np.exp(-z))
    p   = min(p, 1.0)
    mdir = float(np.degrees(np.arctan2(S/n, C/n)) % 360)
    return float(R), mdir, p

# ── 2. REDPy events ───────────────────────────────────────────────────────────
print('Loading REDPy...')
d = redpy.Detector(CONFIG)
d.open()
rtimes_mpl = d.get('rtable', 'startTimeMPL')
d.close()

eq_sec = np.array([
    mdates.num2date(t).replace(tzinfo=timezone.utc).timestamp()
    for t in rtimes_mpl
])
mask = (eq_sec >= t_min) & (eq_sec <= t_max)
eq_sec = eq_sec[mask]
print(f'Events in tide window: {len(eq_sec):,}')

# ── 3. Sliding window ─────────────────────────────────────────────────────────
print('Sliding window R...')
win  = WIN_DAYS  * 86400
step = STEP_DAYS * 86400

centers, R_vals, p_vals, n_vals, mdir_vals = [], [], [], [], []
t = t_min + win / 2
while t <= t_max - win / 2:
    idx = np.where((eq_sec >= t - win/2) & (eq_sec < t + win/2))[0]
    ph  = get_phase(eq_sec[idx]) if len(idx) >= MIN_N else np.array([])
    R, mdir, p = rayleigh(ph)
    centers.append(t)
    R_vals.append(R)
    p_vals.append(p)
    n_vals.append(len(idx))
    mdir_vals.append(mdir)
    t += step

centers   = np.array(centers)
R_vals    = np.array(R_vals,    dtype=float)
p_vals    = np.array(p_vals,    dtype=float)
n_vals    = np.array(n_vals,    dtype=int)
mdir_vals = np.array(mdir_vals, dtype=float)

# Convert to matplotlib dates
centers_mpl = np.array([
    mdates.date2num(datetime.fromtimestamp(t, tz=timezone.utc))
    for t in centers
])
eru_start_mpl = mdates.date2num(datetime.fromtimestamp(ERU_START, tz=timezone.utc))
eru_end_mpl   = mdates.date2num(datetime.fromtimestamp(ERU_END,   tz=timezone.utc))

# ── 4. Period-averaged R (pre / co / post eruption) ──────────────────────────
def period_R(t_lo, t_hi):
    idx = np.where((eq_sec >= t_lo) & (eq_sec < t_hi))[0]
    ph  = get_phase(eq_sec[idx])
    R, mdir, p = rayleigh(ph)
    return R, mdir, p, len(idx)

R_pre,  mdir_pre,  p_pre,  n_pre  = period_R(t_min,     ERU_START)
R_co,   mdir_co,   p_co,   n_co   = period_R(ERU_START,  ERU_END)
R_post, mdir_post, p_post, n_post = period_R(ERU_END,    t_max)
print(f'Pre-eruption  : n={n_pre:5d}  R={R_pre:.3f}  mdir={mdir_pre:.0f}°  p={p_pre:.2e}')
print(f'Co-eruption   : n={n_co:5d}   R={R_co:.3f}   mdir={mdir_co:.0f}°   p={p_co:.2e}')
print(f'Post-eruption : n={n_post:5d} R={R_post:.3f} mdir={mdir_post:.0f}° p={p_post:.2e}')

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = fig.add_gridspec(3, 1, hspace=0.08,
                        height_ratios=[2.5, 1.5, 1])

ax_R   = fig.add_subplot(gs[0])
ax_n   = fig.add_subplot(gs[1], sharex=ax_R)
ax_dir = fig.add_subplot(gs[2], sharex=ax_R)

# ── colour R by significance ──────────────────────────────────────────────────
sig   = p_vals < 0.05
insig = ~sig & ~np.isnan(R_vals)

# eruption shading
for ax in [ax_R, ax_n, ax_dir]:
    ax.axvspan(eru_start_mpl, eru_end_mpl,
               color='lightsalmon', alpha=0.45, zorder=0, label='Co-eruption')

# ── Panel 1: R(t) ─────────────────────────────────────────────────────────────
ax_R.plot(centers_mpl, R_vals, color='#AAAAAA', lw=0.8, zorder=1)
ax_R.scatter(centers_mpl[sig],   R_vals[sig],   c='#1A6FBF', s=18,
             zorder=3, label='p < 0.05 (significant)')
ax_R.scatter(centers_mpl[insig], R_vals[insig], c='#CCCCCC', s=14,
             zorder=2, label='p ≥ 0.05 (not significant)')

# Period-mean horizontal bars
def hbar(ax, t_lo, t_hi, val, color, lw=2.5, ls='-'):
    lo = mdates.date2num(datetime.fromtimestamp(t_lo, tz=timezone.utc))
    hi = mdates.date2num(datetime.fromtimestamp(t_hi, tz=timezone.utc))
    ax.plot([lo, hi], [val, val], color=color, lw=lw, ls=ls, zorder=4)

hbar(ax_R, t_min,     ERU_START, R_pre,  '#2196F3', lw=3)
hbar(ax_R, ERU_START, ERU_END,   R_co,   '#E53935', lw=3)
hbar(ax_R, ERU_END,   t_max,     R_post, '#43A047', lw=3)

# Annotate period means
def annotate_R(ax, t_mid, R, color, label):
    mpl = mdates.date2num(datetime.fromtimestamp(t_mid, tz=timezone.utc))
    ax.annotate(f'{label}\nR={R:.3f}',
                xy=(mpl, R), xytext=(0, 14), textcoords='offset points',
                ha='center', fontsize=8, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=color, lw=0.8))

annotate_R(ax_R, (t_min + ERU_START)/2,    R_pre,  '#2196F3', 'Pre-eruption')
annotate_R(ax_R, (ERU_START + ERU_END)/2,  R_co,   '#E53935', 'Co-eruption')
annotate_R(ax_R, (ERU_END + t_max)/2,      R_post, '#43A047', 'Post-eruption')

ax_R.axhline(0, color='k', lw=0.5, ls=':')
ax_R.set_ylabel(f'Rayleigh R\n({WIN_DAYS}-day window)', fontsize=10)
ax_R.set_ylim(-0.02, min(1.0, np.nanmax(R_vals) * 1.35))
ax_R.legend(fontsize=8, loc='upper right')
ax_R.set_title(
    f'Tidal Sensitivity (Rayleigh R) Through Time — Axial Seamount 2015–2021\n'
    f'Sliding {WIN_DAYS}-day window, {STEP_DAYS}-day step  |  '
    f'0° = low tide = peak extensional stress',
    fontsize=10)
plt.setp(ax_R.get_xticklabels(), visible=False)

# Inset: eruption zoom (2015 only)
ax_inset = ax_R.inset_axes([0.01, 0.55, 0.28, 0.42])
mask_2015 = (centers >= datetime(2015,1,1,tzinfo=timezone.utc).timestamp()) & \
            (centers <= datetime(2016,1,1,tzinfo=timezone.utc).timestamp())
ax_inset.axvspan(eru_start_mpl, eru_end_mpl, color='lightsalmon', alpha=0.5)
ax_inset.plot(centers_mpl[mask_2015], R_vals[mask_2015],
              'o-', ms=4, lw=1.0, color='#555555')
sig_2015 = sig & mask_2015
ax_inset.scatter(centers_mpl[sig_2015], R_vals[sig_2015],
                 c='#1A6FBF', s=20, zorder=3)
ax_inset.set_title('2015 zoom', fontsize=7)
ax_inset.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax_inset.xaxis.set_major_locator(mdates.MonthLocator())
ax_inset.tick_params(labelsize=6)
ax_inset.set_ylim(-0.02, None)
ax_inset.axvline(eru_start_mpl, color='red',    lw=1, ls='--')
ax_inset.axvline(eru_end_mpl,   color='orange', lw=1, ls='--')

# ── Panel 2: event count ──────────────────────────────────────────────────────
ax_n.bar(centers_mpl, n_vals, width=STEP_DAYS * 0.6,
         color=np.where(sig, '#1A6FBF', '#AAAAAA'), alpha=0.75, zorder=1)
ax_n.set_ylabel(f'Events in\n{WIN_DAYS}-day window', fontsize=9)
ax_n.axhline(MIN_N, color='k', lw=0.8, ls=':', label=f'Min n={MIN_N}')
ax_n.legend(fontsize=7, loc='upper right')
plt.setp(ax_n.get_xticklabels(), visible=False)

# ── Panel 3: mean direction ───────────────────────────────────────────────────
valid = ~np.isnan(mdir_vals) & sig
ax_dir.scatter(centers_mpl[valid], mdir_vals[valid],
               c='#1A6FBF', s=14, zorder=2, label='Mean direction (sig.)')
ax_dir.axhline(0,   color='k',      lw=0.8, ls='-',  alpha=0.3)
ax_dir.axhline(180, color='gray',   lw=0.8, ls='--', alpha=0.4)
ax_dir.axhline(360, color='gray',   lw=0.8, ls='--', alpha=0.4)
ax_dir.set_yticks([0, 90, 180, 270, 360])
ax_dir.set_yticklabels(['0°\n(low tide)', '90°', '180°\n(high tide)', '270°', '360°'],
                        fontsize=7)
ax_dir.set_ylabel('Mean direction', fontsize=9)
ax_dir.set_ylim(-10, 370)
ax_dir.legend(fontsize=7, loc='upper right')

# ── shared x-axis formatting ──────────────────────────────────────────────────
for ax in [ax_R, ax_n, ax_dir]:
    ax.axvline(eru_start_mpl, color='red',    lw=1.2, ls='--', alpha=0.8, zorder=5)
    ax.axvline(eru_end_mpl,   color='orange', lw=1.2, ls='--', alpha=0.8, zorder=5)

ax_dir.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_dir.xaxis.set_major_locator(mdates.YearLocator())
ax_dir.set_xlabel('Date', fontsize=10)
plt.setp(ax_dir.get_xticklabels(), rotation=30, ha='right', fontsize=8)

# Legend patches for eruption lines
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color='red',    lw=1.2, ls='--', label='Eruption onset (Apr 24)'),
    Line2D([0],[0], color='orange', lw=1.2, ls='--', label='Co-erup. end (May 17)'),
    mpatches.Patch(color='lightsalmon', alpha=0.6, label='Co-eruption period'),
]
ax_R.legend(handles=ax_R.get_legend().legend_handles + legend_elements,
            fontsize=7.5, loc='upper right', ncol=2)

out_path = f'{OUT_DIR}/16_R_vs_time.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')
