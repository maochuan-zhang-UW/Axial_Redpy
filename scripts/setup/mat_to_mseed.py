"""
Convert Axial Seamount .mat waveform files to MiniSEED format for REDPy.

Each .mat file contains ~1 hour of multi-station data in MATLAB trace structs.
Output: one .mseed file per input .mat file in the output directory.

Usage:
    python mat_to_mseed.py <input_year_dir> <output_dir>

Example (2015 only):
    python mat_to_mseed.py /path/to/Axial-AutoLocate/2015 ./mseed/2015
"""

import sys
import os
import glob
import numpy as np
import scipy.io as sio
from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats

# Stations and their vertical channel names (as stored in the .mat files)
STATION_CHANNEL = {
    'AXCC1': 'HHZ',
    'AXEC1': 'EHZ',
    'AXEC2': 'HHZ',
    'AXEC3': 'EHZ',
    'AXAS1': 'EHZ',
    'AXAS2': 'EHZ',
    'AXID1': 'EHZ',
}

# MATLAB datenum to Unix timestamp conversion
# MATLAB epoch (datenum=0) is January 0, 0000; datenum(1970,1,1)=719529
MATLAB_EPOCH_OFFSET = 719529.0  # days


def matlab_datenum_to_utc(datenum):
    """Convert MATLAB serial datenum to ObsPy UTCDateTime."""
    unix_seconds = (datenum - MATLAB_EPOCH_OFFSET) * 86400.0
    return UTCDateTime(unix_seconds)


def convert_mat_to_mseed(mat_file, out_dir):
    """Convert a single .mat file to a MiniSEED file."""
    try:
        data = sio.loadmat(mat_file, squeeze_me=True)
    except Exception as e:
        print(f'  ERROR loading {mat_file}: {e}')
        return

    if 'trace' not in data:
        print(f'  SKIP (no trace key): {mat_file}')
        return

    traces = data['trace']
    if traces.ndim == 0:
        traces = traces.reshape(1)

    st = Stream()
    for t in traces:
        try:
            station = str(t['station']).strip()
            channel = str(t['channel']).strip()
        except Exception:
            continue

        # Only keep the Z-component traces for our target stations
        if station not in STATION_CHANNEL:
            continue
        if channel != STATION_CHANNEL[station]:
            continue

        try:
            raw_data = np.array(t['data'], dtype=np.float64)
            sample_rate = float(t['sampleRate'])
            start_datenum = float(t['startTime'])
            network = str(t['network']).strip()

            # Location: empty array → empty string
            loc_raw = t['location']
            if hasattr(loc_raw, '__len__') and len(loc_raw) == 0:
                location = ''
            else:
                location = str(loc_raw).strip()

        except Exception as e:
            print(f'  ERROR reading trace {station}.{channel}: {e}')
            continue

        starttime = matlab_datenum_to_utc(start_datenum)

        header = Stats()
        header.network = network
        header.station = station
        header.location = location
        header.channel = channel
        header.sampling_rate = sample_rate
        header.starttime = starttime
        header.npts = len(raw_data)

        tr = Trace(data=raw_data, header=header)
        st.append(tr)

    if len(st) == 0:
        return

    # Name output file after the input file
    basename = os.path.splitext(os.path.basename(mat_file))[0]
    out_file = os.path.join(out_dir, basename + '.mseed')
    st.write(out_file, format='MSEED')
    print(f'  Wrote {len(st)} traces -> {out_file}')


def convert_year(input_dir, output_dir):
    """Recursively convert all .mat files under input_dir."""
    mat_files = sorted(glob.glob(os.path.join(input_dir, '**', '*.mat'),
                                  recursive=True))
    if not mat_files:
        print(f'No .mat files found in {input_dir}')
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f'Converting {len(mat_files)} files from {input_dir} -> {output_dir}')

    for i, mat_file in enumerate(mat_files, 1):
        print(f'[{i}/{len(mat_files)}] {os.path.basename(mat_file)}')
        convert_mat_to_mseed(mat_file, output_dir)

    print('Done.')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    convert_year(input_dir, output_dir)
