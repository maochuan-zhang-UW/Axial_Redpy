"""
Convert A_wavelarge5.mat (Felix struct) to REDPy catalog CSV format.
Outputs axial_catalog_felix_2022_2025.csv with columns:
    EventID, Time, Latitude, Longitude, Depth, Magnitude
"""
import numpy as np
import scipy.io
from datetime import datetime, timezone

MAT_PATH = ('/Users/mczhang/Documents/GitHub/FM4/02-data/Before22OBSs/A_All/'
            'A_wavelarge5.mat')
OUT_CSV = '/Users/mczhang/Documents/GitHub/Axial_Redpy/axial_catalog_felix_2022_2025.csv'

# MATLAB datenum epoch offset (days to Unix epoch Jan 1 1970)
_MATLAB_EPOCH_DAYS = 719529.0


def matlab2iso(dn):
    """Convert MATLAB datenum (float, days) to ISO8601 UTC string."""
    unix_sec = (float(dn) - _MATLAB_EPOCH_DAYS) * 86400.0
    dt = datetime.fromtimestamp(unix_sec, tz=timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')


def main():
    print(f'Loading {MAT_PATH}...')
    m = scipy.io.loadmat(MAT_PATH, squeeze_me=True)
    F = m['Felix']
    n = len(F)
    print(f'  {n} events found')

    with open(OUT_CSV, 'w') as f:
        f.write('EventID,Time,Latitude,Longitude,Depth,Magnitude\n')
        for i in range(n):
            eid = int(F['ID'][i])
            t = matlab2iso(float(F['on'][i]))
            lat = float(F['lat'][i])
            lon = float(F['lon'][i])
            dep = float(F['depth'][i])
            f.write(f'{eid},{t},{lat:.5f},{lon:.5f},{dep:.3f},0.0\n')

    print(f'Saved {n} events to {OUT_CSV}')


if __name__ == '__main__':
    main()
