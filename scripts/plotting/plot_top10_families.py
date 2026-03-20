"""
Plot the 10 largest repeating earthquake families over time.
"""
import tables
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone

H5FILE = './h5/axial_redpytable.h5'

h5 = tables.open_file(H5FILE, 'r')
rtable = h5.root.axial.repeaters
ftable = h5.root.axial.families

# Load repeater data
rep_ids   = rtable.col('id')
rep_times = rtable.col('startTimeMPL')  # matplotlib datenum

# Load family data: members stored as space-separated bytes string of row indices
families = []
for row in ftable.iterrows():
    raw = row['members'].decode('utf-8').strip()
    members = [int(x) for x in raw.split() if x.strip()]
    families.append(members)

# Sort families by size, take top 10
fam_sizes = [(i, len(m)) for i, m in enumerate(families)]
top10 = sorted(fam_sizes, key=lambda x: -x[1])[:10]

# Convert matplotlib datenums to datetime
def mpl_to_dt(mpl_time):
    # matplotlib datenum: days since 0001-01-01 (proleptic Gregorian)
    import matplotlib.dates as mdates
    return mdates.num2date(mpl_time)

fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.tab10(np.linspace(0, 1, 10))

for rank, (fam_idx, size) in enumerate(top10):
    member_rows = families[fam_idx]
    # Get times for each member
    times = []
    for row_idx in member_rows:
        if row_idx < len(rep_times):
            times.append(rep_times[row_idx])
    if not times:
        continue
    times = sorted(times)
    dts = [mpl_to_dt(t) for t in times]
    ys = [rank + 1] * len(dts)
    ax.scatter(dts, ys, s=15, color=colors[rank],
               label=f'Family {fam_idx+1} (n={size})', zorder=3)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Family rank (1 = largest)', fontsize=12)
ax.set_title('Top 10 Repeating Earthquake Families — Axial Seamount Mar 2015', fontsize=13)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.xticks(rotation=45)
ax.set_yticks(range(1, 11))
ax.set_yticklabels([f'#{r} (n={s})' for r, (i, s) in enumerate(top10, 1)])
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=8, ncol=2)
plt.tight_layout()

outfile = './runs/axial/top10_families.png'
plt.savefig(outfile, dpi=150)
print(f'Saved: {outfile}')

h5.close()
