"""
Fig 13 — Tidal triggering sensitivity.

Phase convention (matching Tan et al. 2019 Fig. 2):
  0° = peak extensional (tensile) stress = lowest ocean height (tide minimum)
  ±180° = highest ocean height (tide maximum)

Steps:
  1. Load hourly-downsampled tide predictions 2015-2021
  2. Apply Hilbert transform to get instantaneous phase; shift so 0° = tide minimum
  3. Load REDPy earthquake times (rtable), keep only 2015-2021
  4. Interpolate phase at each earthquake time
  5. Plot histogram + Rayleigh test
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from scipy.signal import hilbert

import redpy

matplotlib.use('agg')

CONFIG  = 'axial_settings.cfg'
OUT_DIR = 'runs/axial/eruption_analysis'
TIDE_DIR = os.path.expanduser(
    '~/Documents/TidesArchive/Tides_Central_caldera')
YEARS   = list(range(2015, 2022))   # tide data covers 2015-2021

DOWNSAMPLE = 240   # keep every 240th row (15 s × 240 = 1 hour)

# ── 1. Load tide data ─────────────────────────────────────────────────────────
print('Loading tide data...')
chunks = []
for yr in YEARS:
    fpath = os.path.join(TIDE_DIR, f'pred_F_{yr}.txt')
    if not os.path.exists(fpath):
        print(f'  WARNING: {fpath} not found, skipping')
        continue
    df = pd.read_csv(fpath, sep=r'\s+', header=None,
                     names=['yr','mo','dy','hr','mn','sc','tide'],
                     dtype={'yr':int,'mo':int,'dy':int,
                            'hr':int,'mn':int,'sc':int,'tide':float})
    df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)
    chunks.append(df)
    print(f'  {yr}: {len(df)} hourly samples')

tide_df = pd.concat(chunks, ignore_index=True)

# Build Unix timestamps for the tide samples
tide_times = np.array([
    datetime(int(r.yr), int(r.mo), int(r.dy),
             int(r.hr), int(r.mn), int(r.sc),
             tzinfo=timezone.utc).timestamp()
    for r in tide_df.itertuples()
])
tide_levels = tide_df['tide'].values.astype(float)

print(f'Total hourly tide samples: {len(tide_times):,}')
print(f'Tide range: {tide_levels.min():.3f} to {tide_levels.max():.3f} m')

# ── 2. Compute instantaneous tidal phase via Hilbert transform ─────────────────
print('Computing tidal phase via Hilbert transform...')
# Demean so Hilbert works cleanly
tide_dm = tide_levels - tide_levels.mean()
analytic = hilbert(tide_dm)
inst_phase = np.angle(analytic)   # radians, range (-π, π], 0 = rising zero-crossing

# Shift so that 0° = tide MINIMUM (peak extensional stress):
#   Standard Hilbert: phase=0 → cos(0)=1 → signal at maximum (high tide)
#   We want: phase=0 → signal at minimum (low tide)
#   So add π and wrap back to (-π, π]
phase_shifted = inst_phase + np.pi                # shift
phase_shifted = (phase_shifted + np.pi) % (2*np.pi) - np.pi  # wrap to (-π, π]
# Now: phase=0 → tide minimum, phase=±π → tide maximum

tide_phase_deg = np.degrees(phase_shifted)        # degrees, range (-180, 180]

# ── 3. Load REDPy event times (filter to 2015-2021) ───────────────────────────
print('Loading REDPy event times...')
d = redpy.Detector(CONFIG)
d.open()
rtimes_mpl = d.get('rtable', 'startTimeMPL')
d.close()

# Convert to Unix seconds
rtimes_sec = np.array([
    mdates.num2date(t).replace(tzinfo=timezone.utc).timestamp()
    for t in rtimes_mpl
])

# Filter to tide data time range
t_min = tide_times[0]
t_max = tide_times[-1]
mask  = (rtimes_sec >= t_min) & (rtimes_sec <= t_max)
eq_times = rtimes_sec[mask]
print(f'REDPy events in 2015-2021: {mask.sum():,} / {len(rtimes_sec):,}')

# ── 4. Interpolate tidal phase at each earthquake time ─────────────────────────
print('Interpolating tidal phase at earthquake times...')
# Linear interpolation of the complex analytic signal, then re-extract angle
# (avoids phase-wrap artefacts)
analytic_real_interp = np.interp(eq_times, tide_times, np.real(analytic))
analytic_imag_interp = np.interp(eq_times, tide_times, np.imag(analytic))
eq_inst_phase = np.angle(analytic_real_interp + 1j*analytic_imag_interp) + np.pi
eq_inst_phase = (eq_inst_phase + np.pi) % (2*np.pi) - np.pi
eq_phase_deg  = np.degrees(eq_inst_phase)

# ── 5. Rayleigh test for non-uniform circular distribution ────────────────────
def rayleigh_test(phases_deg):
    """Return (p_value, mean_direction_deg, mean_resultant_length R)."""
    phi = np.radians(phases_deg)
    n   = len(phi)
    C   = np.sum(np.cos(phi))
    S   = np.sum(np.sin(phi))
    R   = np.sqrt(C**2 + S**2) / n
    z   = n * R**2
    # Approximation for p-value (Zar 1999)
    p   = np.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    mean_dir = np.degrees(np.arctan2(S/n, C/n)) % 360
    return p, mean_dir, R

p_val, mean_dir, R = rayleigh_test(eq_phase_deg)
print(f'Rayleigh test: R={R:.4f}, mean direction={mean_dir:.1f}°, p={p_val:.4e}')

# ── 6. Plot ───────────────────────────────────────────────────────────────────
print('Plotting...')
N_BINS = 18   # 20° bins
bin_edges = np.linspace(-180, 180, N_BINS + 1)
counts, _ = np.histogram(eq_phase_deg, bins=bin_edges)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# — Left panel: histogram with tide schematic inset ——————————————————————————
ax = axes[0]
bar_colors = np.where(bin_centers < 0, 'steelblue', 'tomato')
bars = ax.bar(bin_centers, counts, width=360/N_BINS,
              color=bar_colors, edgecolor='white', linewidth=0.5, alpha=0.85)

# Expected uniform line
n_eq = len(eq_phase_deg)
expected = n_eq / N_BINS
ax.axhline(expected, color='k', lw=1.0, ls='--', label=f'Uniform ({expected:.0f})')

# Tidal phase labels
ax.set_xlabel('Tidal phase (°)', fontsize=11)
ax.set_ylabel('Number of repeating earthquakes', fontsize=11)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_xticklabels(['-180°\n(high tide)', '-90°\n(ebb)', '0°\n(low tide / peak stress)',
                    '+90°\n(flood)', '+180°\n(high tide)'], fontsize=8)
ax.set_xlim(-180, 180)

# Annotation box
textstr = (f'n = {n_eq:,}\n'
           f'Rayleigh R = {R:.3f}\n'
           f'p = {p_val:.2e}\n'
           f'Mean dir = {mean_dir:.1f}°')
ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
        fontsize=8, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))
ax.legend(fontsize=8, loc='upper left')
ax.set_title('Tidal Phase Distribution — REDPy Repeating Earthquakes\n'
             'Axial Seamount 2015–2021  (0° = low tide = peak extensional stress)',
             fontsize=10)

# — Right panel: normalized excess (fraction above uniform) ——————————————————
ax2 = axes[1]
excess = (counts - expected) / expected * 100   # % above/below uniform
bar_colors2 = np.where(excess >= 0, 'tomato', 'steelblue')
ax2.bar(bin_centers, excess, width=360/N_BINS,
        color=bar_colors2, edgecolor='white', linewidth=0.5, alpha=0.85)
ax2.axhline(0, color='k', lw=0.8)
ax2.set_xlabel('Tidal phase (°)', fontsize=11)
ax2.set_ylabel('Excess above uniform (%)', fontsize=11)
ax2.set_xticks([-180, -90, 0, 90, 180])
ax2.set_xticklabels(['-180°\n(high tide)', '-90°\n(ebb)', '0°\n(low tide)',
                     '+90°\n(flood)', '+180°\n(high tide)'], fontsize=8)
ax2.set_xlim(-180, 180)
ax2.set_title('Excess Relative to Uniform Distribution\n'
              '(positive = more earthquakes than expected by chance)', fontsize=10)

plt.tight_layout()
out_path = f'{OUT_DIR}/13_tidal_triggering.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')

# ── 7. Also plot tide-level time series with earthquake occurrences for sanity check ──
print('Sanity-check plot: tide level + quake phases...')
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 10))

# Tide level (first 30 days of 2015)
mask_30d = tide_times < (tide_times[0] + 30*86400)
ax = axes2[0]
ax.plot(tide_times[mask_30d], tide_levels[mask_30d], 'b-', lw=0.8)
ax.set_ylabel('Tide level (m)', fontsize=9)
ax.set_title('Tide level — Jan 2015 (first 30 days)', fontsize=9)
# Convert tide_times x-axis to dates for readability
xt = tide_times[mask_30d]
ax.set_xlabel('Day of January 2015', fontsize=8)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: datetime.fromtimestamp(x, tz=timezone.utc).strftime('%b %d')))
fig2.canvas.draw()

# Phase time series (first 30 days)
ax2 = axes2[1]
ax2.plot(tide_times[mask_30d], tide_phase_deg[mask_30d], 'g-', lw=0.6, alpha=0.8)
ax2.axhline(0, color='r', lw=0.8, ls='--', alpha=0.7)
ax2.set_ylabel('Tidal phase (°)', fontsize=9)
ax2.set_ylim(-185, 185)
ax2.set_yticks([-180, -90, 0, 90, 180])
ax2.set_title('Instantaneous tidal phase (0° = low tide)', fontsize=9)

# Full histogram (polar)
ax3 = axes2[2]
ax3.bar(bin_centers, counts, width=360/N_BINS,
        color='steelblue', edgecolor='white', alpha=0.7)
ax3.axhline(expected, color='k', lw=1.0, ls='--')
ax3.set_xlabel('Tidal phase (°)', fontsize=9)
ax3.set_ylabel('Count', fontsize=9)
ax3.set_title(f'Earthquake tidal phase histogram (n={n_eq:,})', fontsize=9)
ax3.set_xticks([-180, -90, 0, 90, 180])

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/13_tidal_sanity_check.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR}/13_tidal_sanity_check.png')
print('\nDone.')
