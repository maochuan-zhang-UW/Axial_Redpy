"""
Compare tidal sensitivity between REDPy repeating earthquakes
and the general earthquake population (DD + Felix catalog).
Three panels:
  1. Phase histograms side-by-side (overall)
  2. R(t) sliding window — both catalogs on same axes
  3. Summary stats bar chart
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone
from scipy.signal import hilbert
import redpy

matplotlib.use('agg')

CONFIG   = 'axial_settings.cfg'
OUT_DIR  = 'runs/axial/eruption_analysis'
TIDE_DIR = os.path.expanduser('~/Documents/TidesArchive/Tides_Central_caldera')
YEARS    = list(range(2015, 2022))
DOWNSAMPLE = 240

ERU_START = datetime(2015, 4, 24, 8, 0, tzinfo=timezone.utc).timestamp()
ERU_END   = datetime(2015, 5, 19,       tzinfo=timezone.utc).timestamp()

WIN_DAYS  = 60
STEP_DAYS = 15
MIN_N     = 30
N_BINS    = 18

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
analytic    = hilbert(tide_levels - tide_levels.mean())
t_min, t_max = tide_times[0], tide_times[-1]

def get_phase(t_arr):
    ar = np.interp(t_arr, tide_times, np.real(analytic))
    ai = np.interp(t_arr, tide_times, np.imag(analytic))
    ph = np.angle(ar + 1j * ai) + np.pi
    return np.degrees((ph + np.pi) % (2*np.pi) - np.pi)

def rayleigh(phases_deg):
    n = len(phases_deg)
    if n < MIN_N:
        return np.nan, np.nan, np.nan
    phi = np.radians(phases_deg)
    C = np.sum(np.cos(phi)); S = np.sum(np.sin(phi))
    R = np.sqrt(C**2 + S**2) / n
    z = n * R**2
    p = float(np.clip(np.exp(-z), 0, 1))
    mdir = float(np.degrees(np.arctan2(S/n, C/n)) % 360)
    return float(R), mdir, p

# ── Load catalogs ─────────────────────────────────────────────────────────────
print('Loading REDPy...')
d = redpy.Detector(CONFIG)
d.open()
rtimes_mpl = d.get('rtable', 'startTimeMPL')
d.close()
redpy_sec = np.array([
    mdates.num2date(t).replace(tzinfo=timezone.utc).timestamp()
    for t in rtimes_mpl
])
mask_rp = (redpy_sec >= t_min) & (redpy_sec <= t_max)
redpy_sec = redpy_sec[mask_rp]

print('Loading general catalog...')
cat = pd.concat([pd.read_csv('axial_catalog_dd.csv'),
                 pd.read_csv('axial_catalog_felix_2022_2025.csv')],
                ignore_index=True)
cat['Time'] = pd.to_datetime(cat['Time'], utc=True)
all_sec = cat['Time'].values.astype('datetime64[ns]').astype(float) / 1e9
mask_cat = (all_sec >= t_min) & (all_sec <= t_max)
all_sec = all_sec[mask_cat]

# Non-repeating = general catalog minus REDPy events (match within 2 s)
# Simpler: just use full catalog vs repeating as two separate populations
print(f'REDPy events in tide window:   {len(redpy_sec):,}')
print(f'General catalog (tide window): {len(all_sec):,}')

# ── Overall phase & Rayleigh for both ─────────────────────────────────────────
print('Computing overall tidal phase...')
phase_rp  = get_phase(redpy_sec)
phase_all = get_phase(all_sec)

R_rp,  mdir_rp,  p_rp  = rayleigh(phase_rp)
R_all, mdir_all, p_all = rayleigh(phase_all)

bin_edges   = np.linspace(-180, 180, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

cnt_rp,  _ = np.histogram(phase_rp,  bins=bin_edges)
cnt_all, _ = np.histogram(phase_all, bins=bin_edges)

exc_rp  = (cnt_rp  - len(phase_rp)  / N_BINS) / (len(phase_rp)  / N_BINS) * 100
exc_all = (cnt_all - len(phase_all) / N_BINS) / (len(phase_all) / N_BINS) * 100

print(f'REDPy:   R={R_rp:.3f}  mdir={mdir_rp:.1f}°  p={p_rp:.2e}')
print(f'General: R={R_all:.3f}  mdir={mdir_all:.1f}°  p={p_all:.2e}')

# ── Sliding window R for both ─────────────────────────────────────────────────
print('Sliding window...')
win  = WIN_DAYS  * 86400
step = STEP_DAYS * 86400

def sliding_R(eq_times):
    centers, R_vals, n_vals = [], [], []
    t = t_min + win / 2
    while t <= t_max - win / 2:
        idx = np.where((eq_times >= t - win/2) & (eq_times < t + win/2))[0]
        ph  = get_phase(eq_times[idx]) if len(idx) >= MIN_N else np.array([])
        R, _, _ = rayleigh(ph)
        centers.append(t); R_vals.append(R); n_vals.append(len(idx))
        t += step
    return np.array(centers), np.array(R_vals, dtype=float), np.array(n_vals)

c_rp,  R_rp_t,  n_rp_t  = sliding_R(redpy_sec)
c_all, R_all_t, n_all_t = sliding_R(all_sec)

to_mpl = lambda arr: np.array([
    mdates.date2num(datetime.fromtimestamp(t, tz=timezone.utc)) for t in arr])
c_rp_mpl  = to_mpl(c_rp)
c_all_mpl = to_mpl(c_all)
eru_s_mpl = mdates.date2num(datetime.fromtimestamp(ERU_START, tz=timezone.utc))
eru_e_mpl = mdates.date2num(datetime.fromtimestamp(ERU_END,   tz=timezone.utc))

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig,
                         height_ratios=[1.6, 2.2, 1.0],
                         hspace=0.38, wspace=0.28)

C_RP  = '#1565C0'   # REDPy blue
C_ALL = '#E65100'   # General orange

# ── Row 0: phase histograms ───────────────────────────────────────────────────
ax_h_rp  = fig.add_subplot(gs[0, 0])
ax_h_all = fig.add_subplot(gs[0, 1])

for ax, exc, n, R, p, mdir, title, color in [
    (ax_h_rp,  exc_rp,  len(phase_rp),  R_rp,  p_rp,  mdir_rp,
     'REDPy Repeating Earthquakes', C_RP),
    (ax_h_all, exc_all, len(phase_all), R_all, p_all, mdir_all,
     'General Earthquake Population (DD + Felix)', C_ALL),
]:
    bar_c = [color if v >= 0 else '#AAAAAA' for v in exc]
    ax.bar(bin_centers, exc, width=360/N_BINS,
           color=bar_c, edgecolor='white', lw=0.5, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(['-180°\n(high tide)', '-90°', '0°\n(low tide)',
                        '+90°', '+180°\n(high tide)'], fontsize=7)
    ax.set_ylabel('Excess above uniform (%)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', color=color)
    ax.set_xlim(-180, 180)
    p_str = f'{p:.1e}' if p > 1e-300 else '≈ 0'
    ax.text(0.97, 0.97,
            f'n = {n:,}\nR = {R:.3f}\np = {p_str}\nMean = {mdir:.0f}°',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

# ── Row 1: R(t) comparison ────────────────────────────────────────────────────
ax_R = fig.add_subplot(gs[1, :])

ax_R.axvspan(eru_s_mpl, eru_e_mpl, color='lightsalmon', alpha=0.4, zorder=0)
ax_R.axvline(eru_s_mpl, color='red',    lw=1.2, ls='--', alpha=0.8, zorder=5)
ax_R.axvline(eru_e_mpl, color='orange', lw=1.2, ls='--', alpha=0.8, zorder=5)

# REDPy
valid_rp = ~np.isnan(R_rp_t)
ax_R.plot(c_rp_mpl[valid_rp], R_rp_t[valid_rp],
          'o-', ms=3, lw=1.4, color=C_RP, alpha=0.9,
          label=f'REDPy repeating  (overall R={R_rp:.3f})')

# General
valid_all = ~np.isnan(R_all_t)
ax_R.plot(c_all_mpl[valid_all], R_all_t[valid_all],
          's-', ms=3, lw=1.4, color=C_ALL, alpha=0.9,
          label=f'General catalog  (overall R={R_all:.3f})')

# Period-mean bars
def period_R_both(eq_arr, t_lo, t_hi):
    idx = np.where((eq_arr >= t_lo) & (eq_arr < t_hi))[0]
    R, _, _ = rayleigh(get_phase(eq_arr[idx]))
    return R

periods = [
    (t_min,     ERU_START, 'Pre'),
    (ERU_START, ERU_END,   'Co'),
    (ERU_END,   t_max,     'Post'),
]
bar_colors_p = ['#2196F3', '#E53935', '#43A047']
for (t_lo, t_hi, label), bc in zip(periods, bar_colors_p):
    lo = mdates.date2num(datetime.fromtimestamp(t_lo, tz=timezone.utc))
    hi = mdates.date2num(datetime.fromtimestamp(t_hi, tz=timezone.utc))
    for eq_arr, ls, lw in [(redpy_sec, '-', 2.5), (all_sec, '--', 2.0)]:
        R_p = period_R_both(eq_arr, t_lo, t_hi)
        if not np.isnan(R_p):
            ax_R.plot([lo, hi], [R_p, R_p], color=bc, lw=lw, ls=ls,
                      alpha=0.75, zorder=4)

ax_R.set_ylabel(f'Rayleigh R  ({WIN_DAYS}-day window)', fontsize=10)
ax_R.set_ylim(0, None)
ax_R.legend(fontsize=9, loc='upper right')
ax_R.set_title(
    f'Tidal Sensitivity Through Time — REDPy vs General Catalog\n'
    f'({WIN_DAYS}-day sliding window, {STEP_DAYS}-day step  |  '
    f'solid bars = REDPy period means, dashed = general catalog)',
    fontsize=10)
ax_R.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_R.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax_R.get_xticklabels(), rotation=30, ha='right', fontsize=8)

# Eruption labels
ax_R.text(eru_s_mpl, ax_R.get_ylim()[1]*0.97, 'Eruption\nonset',
          color='red', fontsize=7, ha='right', va='top')
ax_R.text(eru_e_mpl, ax_R.get_ylim()[1]*0.97, 'Erup.\nend',
          color='darkorange', fontsize=7, ha='left', va='top')

# ── Row 2: summary bar chart ─────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[2, :])

period_labels = ['All time\n2015–2021',
                 'Pre-eruption\n(Jan–Apr 24, 2015)',
                 'Co-eruption\n(Apr 24–May 19, 2015)',
                 'Post-eruption\n(May 19, 2015–Dec 2021)']
bounds = [(t_min, t_max), (t_min, ERU_START), (ERU_START, ERU_END), (ERU_END, t_max)]
x = np.arange(len(period_labels))
w = 0.35

R_rp_periods  = [period_R_both(redpy_sec, lo, hi) for lo, hi in bounds]
R_all_periods = [period_R_both(all_sec,   lo, hi) for lo, hi in bounds]

bars1 = ax_bar.bar(x - w/2, R_rp_periods,  w, label='REDPy repeating',
                    color=C_RP,  alpha=0.85, edgecolor='white')
bars2 = ax_bar.bar(x + w/2, R_all_periods, w, label='General catalog',
                    color=C_ALL, alpha=0.85, edgecolor='white')

for bars, R_list in [(bars1, R_rp_periods), (bars2, R_all_periods)]:
    for bar, R in zip(bars, R_list):
        if not np.isnan(R):
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f'{R:.3f}', ha='center', va='bottom', fontsize=8)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(period_labels, fontsize=9)
ax_bar.set_ylabel('Rayleigh R', fontsize=10)
ax_bar.set_ylim(0, max(max(R_rp_periods), max(R_all_periods)) * 1.25)
ax_bar.legend(fontsize=9)
ax_bar.set_title('Period-Averaged Tidal Sensitivity Comparison', fontsize=10)
ax_bar.axhline(0, color='k', lw=0.5)

fig.suptitle(
    'Tidal Triggering Sensitivity: REDPy Repeating Earthquakes vs General Earthquake Population\n'
    'Axial Seamount 2015–2021  |  0° = low tide = peak extensional stress',
    fontsize=12, fontweight='bold', y=0.99)

out_path = f'{OUT_DIR}/18_tidal_compare_catalogs.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')
