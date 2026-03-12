# REDPy — Axial Seamount Setup

## 1. Install

```bash
conda env create -f environment.yml
conda activate redpy
pip install .
```

## 2. Convert catalog

```bash
python convert_catalog.py hypo71_20260220.dat axial_catalog.csv
```

This converts the hypo71 format to REDPy's CSV format.

## 3. Configure

Open `axial_settings.cfg` and update two things:

**a) Waveform directory:**
```ini
searchdir=/path/to/your/waveforms/
```

**b) Station metadata — check your actual file headers first:**
```bash
python -c "
import obspy
st = obspy.read('/path/to/any/waveform.mseed', headonly=True)
for tr in st:
    print(tr.stats.network, tr.stats.station, tr.stats.channel, repr(tr.stats.location), tr.stats.sampling_rate)
"
```
Then update `network`, `channel`, `location` in `axial_settings.cfg` to match exactly.

## 4. Run

```bash
# Initialize the HDF5 database
redpy-initialize -c axial_settings.cfg -v

# Process events from catalog (fast — skips STA/LTA scanning)
redpy-catfill -f -v -c axial_settings.cfg axial_catalog.csv

# Check junk to tune QC parameters if needed
redpy-plot-junk -c axial_settings.cfg

# Generate output plots
redpy-force-plot -c axial_settings.cfg
```

## 5. Outputs

Results are written to `./runs/axial/`:
- `overview.html` — interactive timeline of all families
- `families/<N>.html` — per-family pages
- `families/<N>.png` — waveform/spectrum plots
- `catalog.txt` — text catalog of repeaters

## Troubleshooting

| Problem | Fix |
|---|---|
| "No data found" for a station | Location code mismatch — check `repr(tr.stats.location)` vs config |
| ValueError at startup | `nstac`/`ncor`/`teleok` > `nsta`, or FI params outside `fmin`/`fmax` |
| New files not found | Delete `./runs/axial/filelist.csv` to force re-indexing |
| Real events going to junk | Raise `kurtmax` to 130 or set `teleok=7` |
