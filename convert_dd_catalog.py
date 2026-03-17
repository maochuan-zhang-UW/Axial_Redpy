"""
Convert Felix DD catalog (Felix_DD_PSt.mat) to REDPy CSV format.

Filters:
  - Keep only events where >= 5 of 7 stations have DDPt_* != 0
  - Join magnitude from evMw.txt by event ID

Output: axial_catalog_dd.csv
  Columns: EventID, Time, Latitude, Longitude, Depth, Magnitude

Usage:
    python convert_dd_catalog.py
"""
import sys
import csv
import numpy as np
import scipy.io as sio
from datetime import datetime, timezone, timedelta

MAT_FILE = '/Users/mczhang/Documents/GitHub/FM3/02-data/A_All/Felix_DD_PSt.mat'
MW_FILE  = '/Users/mczhang/Documents/GitHub/FM/02-data/rawdata/evMw.txt'
OUT_FILE = 'axial_catalog_dd.csv'

STATIONS   = ['AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1']
MIN_STA    = 5          # minimum stations with DDPt != 0
MATLAB_EPOCH_DAYS = 719529.0  # days from MATLAB datenum(0) to Unix epoch


def matlab_datenum_to_iso(datenum):
    """Convert MATLAB serial datenum to ISO 8601 string (UTC)."""
    unix_sec = (datenum - MATLAB_EPOCH_DAYS) * 86400.0
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=unix_sec)
    us = dt.microsecond
    return (f'{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T'
            f'{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}'
            f'.{us:06d}')


def load_magnitudes(mw_file):
    """Load evMw.txt → dict {ID: magnitude}."""
    mw = {}
    with open(mw_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                mw[int(parts[0])] = float(parts[1])
    return mw


def convert(mat_file, mw_file, out_file):
    print(f'Loading {mat_file} ...')
    d = sio.loadmat(mat_file, squeeze_me=True)
    F = d['Felix']
    print(f'Total events: {len(F)}')

    print(f'Loading magnitudes from {mw_file} ...')
    mw_dict = load_magnitudes(mw_file)

    events = []
    n_low_sta = 0

    for e in F:
        # Count stations with non-zero DDPt
        n_obs = sum(1 for s in STATIONS if float(e[f'DDPt_{s}']) != 0)
        if n_obs < MIN_STA:
            n_low_sta += 1
            continue

        event_id = int(e['ID'])
        time_iso = matlab_datenum_to_iso(float(e['on']))
        lat   = float(e['lat'])
        lon   = float(e['lon'])
        depth = float(e['depth'])
        mag   = mw_dict.get(event_id, float('nan'))

        events.append({
            'EventID':   event_id,
            'Time':      time_iso,
            'Latitude':  round(lat, 5),
            'Longitude': round(lon, 5),
            'Depth':     round(depth, 3),
            'Magnitude': round(mag, 3) if not np.isnan(mag) else '',
        })

    print(f'Removed (< {MIN_STA} stations): {n_low_sta}')
    print(f'Kept: {len(events)}')

    # Sort by time
    events.sort(key=lambda x: x['Time'])

    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['EventID', 'Time', 'Latitude', 'Longitude',
                           'Depth', 'Magnitude'])
        writer.writeheader()
        writer.writerows(events)

    print(f'Saved: {out_file}')
    t0 = events[0]['Time']
    t1 = events[-1]['Time']
    print(f'Date range: {t0[:10]} → {t1[:10]}')


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else OUT_FILE
    convert(MAT_FILE, MW_FILE, out)
