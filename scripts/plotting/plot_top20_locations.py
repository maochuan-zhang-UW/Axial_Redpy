"""
Plot locations of the 10 largest repeating earthquake families.
Map extent matches Figure04_FM.m: lon [-130.031, -129.97], lat [45.92, 45.970]
"""
import tables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from obspy import UTCDateTime

H5FILE  = './h5/axial_redpytable.h5'
CATALOG = './axial_catalog_mar_apr_2015.csv'

# --- Map bounds from Figure04_FM.m ---
LON_LIM = [-130.031, -129.970]
LAT_LIM = [45.920,   45.970]

# --- Load HDF5 ---
h5     = tables.open_file(H5FILE, 'r')
rtable = h5.root.axial.repeaters
ftable = h5.root.axial.families

rep_times = rtable.col('startTimeMPL')   # matplotlib datenums

families = []
for row in ftable.iterrows():
    raw     = row['members'].decode('utf-8').strip()
    members = [int(x) for x in raw.split() if x.strip()]
    families.append(members)
h5.close()

# Top 20 by size
top20 = sorted(enumerate(families), key=lambda x: -len(x[1]))[:20]

# --- Load catalog ---
cat = pd.read_csv(CATALOG, parse_dates=['Time'])
cat['mpl'] = cat['Time'].apply(
    lambda t: (t.timestamp() / 86400.0) + 719163.0)  # Unix→matplotlib datenum

# --- Match repeater times to catalog lat/lon (nearest within 1 sec) ---
import matplotlib.dates as mdates

colors = plt.cm.tab20(np.linspace(0, 1, 20))

fig, ax = plt.subplots(figsize=(9, 7))

# Background scatter of ALL catalog events (light grey)
mask = ((cat['Longitude'] >= LON_LIM[0]) & (cat['Longitude'] <= LON_LIM[1]) &
        (cat['Latitude']  >= LAT_LIM[0]) & (cat['Latitude']  <= LAT_LIM[1]))
ax.scatter(cat.loc[mask, 'Longitude'], cat.loc[mask, 'Latitude'],
           s=2, c='lightgrey', zorder=1, label='All catalog events')

# Station locations
stations = {
    'AXCC1': (-130.0089, 45.9540),
    'AXEC1': (-129.9778, 45.9549),
    'AXEC2': (-130.0123, 45.9675),
    'AXEC3': (-129.9887, 45.9665),
    'AXAS1': (-130.0143, 45.9313),
    'AXAS2': (-130.0027, 45.9272),
    'AXID1': (-129.9752, 45.9255),
}
for sname, (slon, slat) in stations.items():
    ax.plot(slon, slat, '^', color='black', ms=7, zorder=5)
    ax.text(slon+0.0005, slat+0.0005, sname, fontsize=7, zorder=6)

# Plot top 10 families
legend_handles = []
for rank, (fam_idx, members) in enumerate(top20):
    times_mpl = [rep_times[i] for i in members if i < len(rep_times)]
    lats, lons = [], []
    for t in times_mpl:
        dt = mdates.num2date(t).replace(tzinfo=None)
        diff = (cat['Time'].dt.tz_localize(None) - dt).abs()
        idx  = diff.idxmin()
        if diff[idx].total_seconds() < 5:
            lons.append(cat.loc[idx, 'Longitude'])
            lats.append(cat.loc[idx, 'Latitude'])

    if lons:
        ax.scatter(lons, lats, s=18, color=colors[rank],
                   zorder=4, alpha=0.8, edgecolors='none')
        h = mpatches.Patch(color=colors[rank],
                           label=f'#{rank+1} Family {fam_idx+1} (n={len(members)})')
        legend_handles.append(h)

# Regions from Figure04_FM.m
regions = {
    'West': dict(lat=[45.930, 45.950], lon=[-130.029, -130.008]),
    'East': dict(lat=[45.970, 45.930], lon=[-130.0015, -129.975]),
    'ID':   dict(lat=[45.921, 45.929], lon=[-130.004, -129.975]),
}
for rname, r in regions.items():
    rx = [r['lon'][0], r['lon'][1], r['lon'][1], r['lon'][0], r['lon'][0]]
    ry = [r['lat'][0], r['lat'][0], r['lat'][1], r['lat'][1], r['lat'][0]]
    ax.plot(rx, ry, 'k--', lw=1.5, zorder=3)
    ax.text(np.mean(r['lon']), max(r['lat'])+0.001, rname,
            fontsize=9, ha='center', fontweight='bold')

ax.set_xlim(LON_LIM)
ax.set_ylim(LAT_LIM)
# Correct aspect ratio for latitude
ax.set_aspect(1.0 / np.cos(np.radians(np.mean(LAT_LIM))))
ax.set_xlabel('Longitude (°)', fontsize=12)
ax.set_ylabel('Latitude (°)', fontsize=12)
ax.set_title('Top 20 Repeating Earthquake Family Locations\nAxial Seamount — Mar 2015',
             fontsize=13)
ax.grid(True, linestyle='-', linewidth=0.5, color='grey', alpha=0.5)
ax.legend(handles=legend_handles, fontsize=8, loc='lower right',
          framealpha=0.9, ncol=2)
plt.tight_layout()

outfile = './runs/axial/top20_locations.png'
plt.savefig(outfile, dpi=150)
print(f'Saved: {outfile}')
