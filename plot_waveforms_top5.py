"""
Plot stacked waveforms for the top 5 largest repeating earthquake families,
one subplot per station (7 stations), waveforms overlaid per family.
P-wave alignment uses per-station Dt_* picks from the Felix DD catalog.
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import tables
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio

H5FILE   = './h5/axial_redpytable.h5'
FELIX    = '/Users/mczhang/Documents/GitHub/FM3/02-data/A_All/Felix_DD_PSt.mat'
NSTA     = 7
STATIONS = ['AXCC1', 'AXEC1', 'AXEC2', 'AXEC3', 'AXAS1', 'AXAS2', 'AXID1']
# Felix short codes in same order as STATIONS
FELIX_STA = ['CC1', 'EC1', 'EC2', 'EC3', 'AS1', 'AS2', 'ID1']
SAMPRATE = 100.0   # Hz
N_TOP    = 5
MAX_WAVEFORMS = 50  # max waveforms to overlay per family
MATLAB_EPOCH  = 719529.0   # days from MATLAB epoch to Unix epoch

# Window around P: 1 sec before to 2.5 sec after P arrival
PRE_P  = 1.0   # seconds before P
POST_P = 2.5   # seconds after P
WIN_SAMPLES = int((PRE_P + POST_P) * SAMPRATE)  # 350 samples
t_plot = np.linspace(-PRE_P, POST_P, WIN_SAMPLES, endpoint=False)

# Fallback: fixed ptrig position (used when Dt=0 / no pick)
ptrig = 1.5 * 256 / SAMPRATE  # 3.84 sec from window start

# --- Load Felix catalog P travel times ---
print('Loading Felix catalog...')
mat = sio.loadmat(FELIX)
F   = mat['Felix'][0]  # (147897,) structured array

def _col(field):
    out = np.full(len(F), np.nan)
    for i, ev in enumerate(F):
        v = ev[field]
        if v.size > 0:
            out[i] = float(v.flat[0])
    return out

on_unix = _col('on') - MATLAB_EPOCH     # event origin times in Unix days
Dt      = {s: _col('Dt_' + s) for s in FELIX_STA}
print(f'  Loaded {len(on_unix)} Felix events')

# --- Load HDF5 ---
print('Loading HDF5...')
h5     = tables.open_file(H5FILE, 'r')
rtable = h5.root.axial.repeaters
ftable = h5.root.axial.families

rep_waveforms = rtable.col('waveform')     # (N_events, NSTA*wlen)
rep_start_mpl = rtable.col('startTimeMPL') # Unix days (window start)

families = []
for row in ftable.iterrows():
    raw     = row['members'].decode('utf-8').strip()
    members = [int(x) for x in raw.split() if x.strip()]
    families.append(members)
h5.close()

wlen = rep_waveforms.shape[1] // NSTA  # samples per station

# Build Felix time lookup: for each REDPy event, find matching Felix index
print('Building Felix time lookup...')
valid_on = ~np.isnan(on_unix)

def find_felix(start_mpl):
    """Return Felix index best matching this REDPy window start time."""
    on_approx = start_mpl + ptrig / 86400   # estimated origin time (Unix days)
    diffs = np.where(valid_on, np.abs(on_unix - on_approx) * 86400, 1e9)
    best_i = int(np.argmin(diffs))
    if diffs[best_i] < 5.0:   # within 5 seconds
        return best_i
    return None

def get_p_sample(m, sta_idx):
    """Return the P-wave sample index within the per-station waveform (length wlen).
    Falls back to ptrig if no Felix pick available."""
    felix_i = find_felix(rep_start_mpl[m])
    if felix_i is not None:
        dt_val = Dt[FELIX_STA[sta_idx]][felix_i]
        if not np.isnan(dt_val) and 0 < dt_val < 4.0:
            # P sample in window: (origin + dt - window_start) * samprate
            on_ev = on_unix[felix_i]
            p_sec = (on_ev + dt_val / 86400 - rep_start_mpl[m]) * 86400
            p_samp = int(round(p_sec * SAMPRATE))
            if 0 < p_samp < wlen:
                return p_samp
    # fallback: fixed ptrig
    return int(ptrig * SAMPRATE)

# Top 5 by family size
top5   = sorted(enumerate(families), key=lambda x: -len(x[1]))[:N_TOP]
colors = plt.cm.tab10(np.linspace(0, 1, N_TOP))

# --- Plot ---
print('Plotting...')
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Top 5 Repeating Earthquake Families — Per-Station P-wave Alignment\n'
             'Axial Seamount (7 stations) | Red dashed = P arrival (Felix Dt_* picks)',
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(NSTA, N_TOP, figure=fig, hspace=0.15, wspace=0.08)

for col_idx, (rank, (fam_idx, members)) in enumerate(enumerate(top5)):
    n = len(members)
    plot_members = members if n <= MAX_WAVEFORMS else \
        list(np.random.choice(members, MAX_WAVEFORMS, replace=False))

    for sta_idx in range(NSTA):
        ax = fig.add_subplot(gs[sta_idx, col_idx])

        waves = []
        for m in plot_members:
            if m >= len(rep_waveforms):
                continue
            raw = rep_waveforms[m, sta_idx*wlen:(sta_idx+1)*wlen].astype(float)
            p_samp = get_p_sample(m, sta_idx)

            # Extract window around P: [p_samp - PRE_P*sr : p_samp + POST_P*sr]
            i0 = p_samp - int(PRE_P  * SAMPRATE)
            i1 = p_samp + int(POST_P * SAMPRATE)

            # Pad if window extends outside waveform
            if i0 < 0 or i1 > wlen:
                w = np.zeros(WIN_SAMPLES)
                src_start = max(i0, 0)
                src_end   = min(i1, wlen)
                dst_start = src_start - i0
                dst_end   = dst_start + (src_end - src_start)
                w[dst_start:dst_end] = raw[src_start:src_end]
            else:
                w = raw[i0:i1]

            peak = np.max(np.abs(w))
            if peak > 0:
                w = w / peak
            waves.append(w)

        for w in waves:
            ax.plot(t_plot, w, color=colors[rank], alpha=0.3, lw=0.5)

        if waves:
            stack = np.mean(waves, axis=0)
            ax.plot(t_plot, stack, color='black', lw=1.2, zorder=5)

        ax.axvline(0, color='red', lw=0.8, ls='--', zorder=6)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

        if col_idx == 0:
            ax.set_ylabel(STATIONS[sta_idx], fontsize=8,
                          rotation=0, labelpad=35, va='center')
        if sta_idx == 0:
            ax.set_title(f'#{rank+1} Fam {fam_idx+1}\n(n={n})',
                         fontsize=9, color=colors[rank], fontweight='bold')
        if sta_idx == NSTA - 1:
            ax.set_xlabel('Time rel. P (s)', fontsize=8)
        else:
            ax.set_xticklabels([])

plt.tight_layout(rect=[0.04, 0, 1, 0.95])

outfile = './runs/axial/top5_waveforms_p_aligned.png'
plt.savefig(outfile, dpi=150)
print(f'Saved: {outfile}')
