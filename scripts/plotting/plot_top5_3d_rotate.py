"""
3D rotating animation of top 5 repeating earthquake families (lat, lon, depth).
Event locations from Felix DD catalog. Saved as MP4.
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import tables
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import scipy.io as sio

H5FILE  = './h5/axial_redpytable.h5'
FELIX   = '/Users/mczhang/Documents/GitHub/FM3/02-data/A_All/Felix_DD_PSt.mat'
OUTFILE = './runs/axial/top5_families_3d.mp4'
MATLAB_EPOCH = 719529.0   # days from MATLAB epoch to Unix epoch
N_TOP   = 5
ptrig   = 1.5 * 256 / 100.0   # 3.84 sec

# ── Load Felix catalog ────────────────────────────────────────────────────────
print('Loading Felix catalog...')
mat = sio.loadmat(FELIX)
F   = mat['Felix'][0]   # (147897,) structured array

def _col(field):
    out = np.full(len(F), np.nan)
    for i, ev in enumerate(F):
        v = ev[field]
        if v.size > 0:
            out[i] = float(v.flat[0])
    return out

on_unix = _col('on') - MATLAB_EPOCH
lat_all = _col('lat')
lon_all = _col('lon')
dep_all = _col('depth')   # km, positive down
print(f'  {len(on_unix)} events, depth range: {np.nanmin(dep_all):.1f}–{np.nanmax(dep_all):.1f} km')

# ── Load HDF5 families ────────────────────────────────────────────────────────
print('Loading HDF5...')
h5     = tables.open_file(H5FILE, 'r')
rtable = h5.root.axial.repeaters
ftable = h5.root.axial.families

rep_start_mpl = rtable.col('startTimeMPL')   # Unix days

families = []
for row in ftable.iterrows():
    raw     = row['members'].decode('utf-8').strip()
    members = [int(x) for x in raw.split() if x.strip()]
    families.append(members)
h5.close()

top5 = sorted(enumerate(families), key=lambda x: -len(x[1]))[:N_TOP]

# ── Match each REDPy member to a Felix event ──────────────────────────────────
print('Matching events to Felix catalog...')
valid_on = ~np.isnan(on_unix)

def felix_idx(start_mpl):
    on_approx = start_mpl + ptrig / 86400
    diffs = np.where(valid_on, np.abs(on_unix - on_approx) * 86400, 1e9)
    best_i = int(np.argmin(diffs))
    return best_i if diffs[best_i] < 5.0 else None

colors = plt.cm.tab10(np.linspace(0, 1, N_TOP))

# Build per-family location arrays
fam_data = []
for rank, (fam_idx, members) in enumerate(top5):
    lats, lons, deps = [], [], []
    for m in members:
        fi = felix_idx(rep_start_mpl[m])
        if fi is not None and not np.isnan(lat_all[fi]):
            lats.append(lat_all[fi])
            lons.append(lon_all[fi])
            deps.append(dep_all[fi])
    fam_data.append({
        'rank':    rank,
        'fam_idx': fam_idx,
        'n':       len(members),
        'lat':     np.array(lats),
        'lon':     np.array(lons),
        'dep':     np.array(deps),
        'color':   colors[rank],
    })
    print(f'  Family #{rank+1} (Fam {fam_idx+1}): {len(members)} members, '
          f'{len(lats)} located, depth {np.nanmean(deps):.1f} km avg')

# ── Build 3D figure ───────────────────────────────────────────────────────────
print('Building animation...')
fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection='3d')

# Use local offsets (km-ish) for better axis scaling
# lon/lat → rough km from center
ctr_lat = 45.950
ctr_lon = -130.000
km_per_deg_lat = 111.0
km_per_deg_lon = 111.0 * np.cos(np.radians(ctr_lat))

for fd in fam_data:
    x = (fd['lon'] - ctr_lon) * km_per_deg_lon
    y = (fd['lat'] - ctr_lat) * km_per_deg_lat
    z = -fd['dep']   # negative depth = elevation style (up = shallow)

    ax.scatter(x, y, z,
               c=[fd['color']], s=8, alpha=0.5, linewidths=0,
               label=f"#{fd['rank']+1} Fam {fd['fam_idx']+1} (n={fd['n']})")

ax.set_xlabel('E–W (km)', fontsize=9, labelpad=6)
ax.set_ylabel('N–S (km)', fontsize=9, labelpad=6)
ax.set_zlabel('Depth (km)', fontsize=9, labelpad=6)

# Depth tick labels: convert back to positive depth
def fmt_depth(val, pos=None):
    return f'{-val:.0f}'

from matplotlib.ticker import FuncFormatter
ax.zaxis.set_major_formatter(FuncFormatter(fmt_depth))

ax.set_title('Top 5 Repeating Earthquake Families — Axial Seamount\n'
             'Felix DD Catalog | Depth positive downward',
             fontsize=11, fontweight='bold', pad=12)

legend = ax.legend(loc='upper left', fontsize=8, framealpha=0.7,
                   markerscale=2, title='Family', title_fontsize=8)

# ── Animation: full 360° azimuth sweep ───────────────────────────────────────
N_FRAMES = 360   # one frame per degree
ELEV     = 20    # fixed elevation angle (degrees)

def init():
    ax.view_init(elev=ELEV, azim=0)
    return []

def update(frame):
    # First half: rotate flat (elev=20)
    # Second half: tilt down for a top-down look, then back
    if frame < 180:
        elev = ELEV
    else:
        # smoothly change elev: 20 → 60 → 20 over frames 180-360
        t = (frame - 180) / 180.0
        elev = ELEV + 40 * np.sin(np.pi * t)
    ax.view_init(elev=elev, azim=frame)
    return []

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                               init_func=init, interval=40, blit=False)

writer = animation.FFMpegWriter(fps=25, bitrate=1800,
                                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
ani.save(OUTFILE, writer=writer, dpi=130)
print(f'Saved: {OUTFILE}')
plt.close(fig)
