"""
Filter REDPy families where all 7 stations have mean CC > threshold
between the core event and family members. Generates a filtered HTML page.
"""
import os
import numpy as np
from scipy.fft import ifft
import redpy
from obspy import UTCDateTime

CC_THRESH = 0.95
CONFIG = 'axial_settings.cfg'
OUT_HTML = './runs/axial/filtered.html'


def per_station_cc(fft1, coeff1, fft2, coeff2, nsta, winlen):
    """Return per-station CC array (length nsta) between two events."""
    ccs = np.zeros(nsta)
    for s in range(nsta):
        win1 = fft1[s*winlen:(s+1)*winlen]
        win2 = fft2[s*winlen:(s+1)*winlen]
        corr = np.real(ifft(win1 * np.conj(win2)))
        ccs[s] = corr.max() * coeff1[s] * coeff2[s]
    return ccs


def family_mean_station_cc(fft_core, coeff_core, members, fft_col, coeff_col,
                            nsta, winlen):
    """Mean per-station CC across all members vs core."""
    all_ccs = []
    for m in members:
        fft_m = fft_col[m]
        coeff_m = coeff_col[m]
        ccs = per_station_cc(fft_core, coeff_core, fft_m, coeff_m, nsta, winlen)
        all_ccs.append(ccs)
    return np.mean(all_ccs, axis=0)  # shape (nsta,)


def main():
    d = redpy.Detector(CONFIG)
    d.open()

    nfam = len(d)
    nsta = d.get('nsta')
    winlen = d.get('winlen')
    sta_names = list(d.get('station'))

    fft_col = d.h5file.root.axial.repeaters.col('windowFFT')
    coeff_col = d.h5file.root.axial.repeaters.col('windowCoeff')
    rtimes_mpl = d.get('rtable', 'startTimeMPL')

    import matplotlib.dates as mdates

    qualifying = []
    print(f'Checking {nfam} families for per-station CC > {CC_THRESH}...')
    for fnum in range(nfam):
        members = d.get_members(fnum)
        if len(members) < 2:
            continue
        core = d.get('ftable', 'core', fnum)
        fft_core = fft_col[core]
        coeff_core = coeff_col[core]
        mean_cc = family_mean_station_cc(
            fft_core, coeff_core, members, fft_col, coeff_col, nsta, winlen)
        if np.all(mean_cc >= CC_THRESH):
            n_events = len(members)
            t_first = mdates.num2date(rtimes_mpl[members].min())
            t_last  = mdates.num2date(rtimes_mpl[members].max())
            qualifying.append({
                'fnum': fnum,
                'n': n_events,
                'mean_cc': mean_cc,
                'first': t_first.strftime('%Y-%m-%d'),
                'last':  t_last.strftime('%Y-%m-%d'),
            })
            print(f'  Family {fnum}: {n_events} events, '
                  f'min_sta_cc={mean_cc.min():.3f}')

    d.close()

    print(f'\n{len(qualifying)} families pass (all stations mean CC >= {CC_THRESH})')

    # --- Write filtered HTML page ---
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html><head>
<title>REDPy Filtered Families (per-station CC ≥ {CC_THRESH})</title>
<style>
  body {{ font-family: Helvetica; font-size: 12px; margin: 20px; }}
  h1 {{ font-size: 18px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  tr:hover {{ background: #f9f9f9; }}
  a {{ color: #c00; }}
</style>
</head><body>
<h1>Filtered Families — all {nsta} stations mean CC ≥ {CC_THRESH}</h1>
<p>{len(qualifying)} of {nfam} families pass &nbsp;|&nbsp;
<a href="overview.html">← Back to overview</a></p>
<table>
<tr><th>#</th><th>Family</th><th>Events</th>
<th>First</th><th>Last</th>
""")
        for sname in sta_names:
            f.write(f'<th>{sname}<br>mean CC</th>')
        f.write('<th>Waveform</th><th>Map</th></tr>\n')

        for q in qualifying:
            fnum = q['fnum']
            fam_url = f'families/{fnum}.html'
            f.write(f"""<tr>
<td>{qualifying.index(q)+1}</td>
<td><a href="{fam_url}">Family {fnum}</a></td>
<td>{q['n']}</td>
<td>{q['first']}</td>
<td>{q['last']}</td>
""")
            for cc in q['mean_cc']:
                color = '#2a2' if cc >= CC_THRESH else '#c00'
                f.write(f'<td style="color:{color}">{cc:.3f}</td>')
            f.write(f"""<td><img src="families/{fnum}.png" width=300 height=60></td>
<td><img src="families/map{fnum}.png" width=200></td>
</tr>\n""")

        f.write('</table></body></html>\n')

    print(f'Saved: {OUT_HTML}')


if __name__ == '__main__':
    main()
