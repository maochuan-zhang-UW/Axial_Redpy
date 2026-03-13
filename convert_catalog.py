"""
Convert Axial Seamount hypo71 catalog to REDPy CSV format.

Input format (hypo71):
  yyyymmdd HHMM SS.SS  LatD LatM  LonD LonM  Depth  MW  NWR GAP DMIN RMS ERH ERZ  ID  PMom SMom

  Example:
  20260220 0147 24.77 45 55.41 129 59.16   0.90   0.30 11 230  0.7 0.01  1.2  1.4  514937  4.8e+18 2.4e+18

Output format (REDPy):
  EventID,Time,Latitude,Longitude,Depth,Magnitude
"""
import csv
import sys
from datetime import datetime, timedelta


def dm_to_dd(degrees, minutes):
    """Convert degrees + decimal minutes to decimal degrees."""
    return degrees + minutes / 60.0


def parse_hypo71(infile, outfile, start_date=None, end_date=None):
    """
    start_date / end_date: optional 'YYYYMM' strings to filter by month.
    """
    events = []

    with open(infile, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header line
            if not line or line.startswith('yyyy') or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 16:
                continue

            date_str = parts[0]          # yyyymmdd
            time_str = parts[1]          # HHMM
            sec_str  = parts[2]          # SS.SS
            lat_d    = int(parts[3])     # latitude degrees
            lat_m    = float(parts[4])   # latitude decimal minutes
            lon_d    = int(parts[5])     # longitude degrees (positive, W implied)
            lon_m    = float(parts[6])   # longitude decimal minutes
            depth    = float(parts[7])
            mag      = float(parts[8])
            event_id = parts[15]

            # Optional date filter
            yyyymm = date_str[:6]
            if start_date and yyyymm < start_date:
                continue
            if end_date and yyyymm > end_date:
                continue

            # Parse date/time
            year   = int(date_str[0:4])
            month  = int(date_str[4:6])
            day    = int(date_str[6:8])
            hour   = int(time_str[0:2])
            minute = int(time_str[2:4])
            sec    = float(sec_str)

            # Handle seconds >= 60 (roll over to next minute)
            extra_minutes = int(sec) // 60
            sec = sec - extra_minutes * 60
            dt = datetime(year, month, day, hour, minute) + timedelta(minutes=extra_minutes)
            year, month, day, hour, minute = dt.year, dt.month, dt.day, dt.hour, dt.minute

            sec_int  = int(sec)
            microsec = int(round((sec - sec_int) * 1e6))
            iso_time = (f'{year:04d}-{month:02d}-{day:02d}T'
                        f'{hour:02d}:{minute:02d}:{sec_int:02d}'
                        f'.{microsec:06d}')

            # Convert lat/lon from degrees+decimal minutes to decimal degrees
            # Longitude is West (negative)
            lat = dm_to_dd(lat_d, lat_m)
            lon = -dm_to_dd(lon_d, lon_m)

            events.append({
                'EventID':   event_id,
                'Time':      iso_time,
                'Latitude':  round(lat, 5),
                'Longitude': round(lon, 5),
                'Depth':     depth,
                'Magnitude': mag,
            })

    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['EventID', 'Time', 'Latitude', 'Longitude',
                           'Depth', 'Magnitude'])
        writer.writeheader()
        writer.writerows(events)

    print(f'Converted {len(events)} events -> {outfile}')


if __name__ == '__main__':
    infile     = sys.argv[1] if len(sys.argv) > 1 else 'hypo71.dat'
    outfile    = sys.argv[2] if len(sys.argv) > 2 else 'axial_catalog.csv'
    start_date = sys.argv[3] if len(sys.argv) > 3 else None  # e.g. '201503'
    end_date   = sys.argv[4] if len(sys.argv) > 4 else None  # e.g. '201504'
    parse_hypo71(infile, outfile, start_date, end_date)
