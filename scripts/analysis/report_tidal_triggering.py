"""
Generate a multi-page PDF report on tidal triggering of repeating earthquakes
at Axial Seamount, using matplotlib PdfPages.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import textwrap, datetime

matplotlib.use('agg')

OUT_PDF  = 'runs/axial/eruption_analysis/tidal_triggering_report.pdf'
FIG_DIR  = 'runs/axial/eruption_analysis'
TODAY    = datetime.date.today().strftime('%B %d, %Y')

# ── colour / style constants ───────────────────────────────────────────────────
BG   = '#FAFAFA'
HEAD = '#1A3A5C'
SUB  = '#2E6DA4'
TXT  = '#222222'
RULE = '#AAAAAA'

def page_style(fig):
    fig.patch.set_facecolor(BG)

def hline(ax, y, lw=0.8, color=RULE):
    ax.plot([0, 1], [y, y], color=color, lw=lw,
            transform=ax.transAxes, clip_on=False)

def text_ax(fig, rect=[0, 0, 1, 1]):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    return ax

# ── helper: embed a PNG ────────────────────────────────────────────────────────
def embed_png(pdf, png_path, title=None, caption=None, title_y=0.97):
    fig = plt.figure(figsize=(11, 8.5))
    page_style(fig)

    if title:
        fig.text(0.5, title_y, title, ha='center', va='top',
                 fontsize=13, fontweight='bold', color=HEAD)

    top    = title_y - 0.06 if title else 0.97
    bottom = 0.10 if caption else 0.03
    img    = mpimg.imread(png_path)
    ax_img = fig.add_axes([0.03, bottom, 0.94, top - bottom])
    ax_img.imshow(img, aspect='equal')
    ax_img.set_axis_off()

    if caption:
        fig.text(0.5, 0.04, caption, ha='center', va='bottom',
                 fontsize=8, color=TXT, style='italic',
                 wrap=True, multialignment='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
with PdfPages(OUT_PDF) as pdf:

    # ── PAGE 1 — Title ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page_style(fig)
    ax = text_ax(fig)

    ax.add_patch(plt.Rectangle((0.05, 0.62), 0.90, 0.30,
                                transform=ax.transAxes,
                                color=HEAD, zorder=0, clip_on=False))
    ax.text(0.50, 0.80, 'Tidal Triggering of Repeating Earthquakes',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=20, fontweight='bold', color='white', zorder=1)
    ax.text(0.50, 0.71, 'at Axial Seamount, Juan de Fuca Ridge (2015–2021)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=14, color='#CCDDEE', zorder=1)

    body = [
        ('Data',    'REDPy repeating earthquake catalog  |  29,536 events in 2015–2021'),
        ('Tide',    'Predicted tidal loading: Tides_Central_caldera/pred_F_YYYY.txt'),
        ('Method',  'Hilbert-transform instantaneous phase  +  Rayleigh circular statistics'),
        ('Output',  'Overall sensitivity  |  Per-family ranking  |  Spatial maps  |  Temporal evolution'),
        ('Date',    TODAY),
    ]
    y0 = 0.56
    for label, val in body:
        ax.text(0.10, y0, label + ':', ha='left', va='top',
                transform=ax.transAxes, fontsize=10,
                fontweight='bold', color=SUB)
        ax.text(0.26, y0, val, ha='left', va='top',
                transform=ax.transAxes, fontsize=10, color=TXT)
        y0 -= 0.07

    ax.plot([0.05, 0.95], [0.08, 0.08], color=RULE, lw=1.0, transform=ax.transAxes, clip_on=False)
    ax.text(0.50, 0.04,
            'M. Zhang  ·  Axial Seamount Repeating Earthquake Project',
            ha='center', va='bottom', transform=ax.transAxes,
            fontsize=9, color='#666666')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── PAGE 2 — Background & Methods ─────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page_style(fig)
    ax = text_ax(fig)

    def heading(ax, y, txt, level=1):
        fs = 15 if level == 1 else 12
        col = HEAD if level == 1 else SUB
        ax.text(0.05, y, txt, ha='left', va='top', transform=ax.transAxes,
                fontsize=fs, fontweight='bold', color=col)

    def para(ax, y, txt, indent=0.05, width=90, fs=10):
        lines = textwrap.wrap(txt, width)
        for line in lines:
            ax.text(indent, y, line, ha='left', va='top',
                    transform=ax.transAxes, fontsize=fs, color=TXT)
            y -= 0.042
        return y

    def bullet(ax, y, items, indent=0.08, fs=10):
        for item in items:
            lines = textwrap.wrap(item, 85)
            ax.text(indent - 0.02, y, '•', ha='left', va='top',
                    transform=ax.transAxes, fontsize=fs, color=SUB)
            for k, line in enumerate(lines):
                ax.text(indent, y, line, ha='left', va='top',
                        transform=ax.transAxes, fontsize=fs, color=TXT)
                y -= 0.040
        return y

    y = 0.96
    heading(ax, y, '1.  Background', level=1); y -= 0.06
    y = para(ax, y,
        'Ocean tides impose periodic loading on the seafloor, causing oscillating '
        'normal and shear stresses on faults. At Axial Seamount, a frequently '
        'erupting submarine volcano on the Juan de Fuca Ridge (depth ~1500 m), the '
        'tidal amplitude is ~1–2 m, generating ~15–30 kPa of periodic stress. '
        'Tan et al. (2019, Science) showed that repeating earthquakes at Axial are '
        'strongly concentrated near low tide — when reduced water-column pressure '
        'maximises extensional (tensile) stress on ring-fault structures — '
        'consistent with pressure-sensitive fault slip.', width=100); y -= 0.01

    heading(ax, y, '2.  Phase Convention', level=2); y -= 0.05
    y = bullet(ax, y, [
        '0°  =  low tide (minimum ocean height)  =  peak extensional stress  '
        '(tidal loading removed → fault unclamped → slip promoted).',
        '±180°  =  high tide (maximum ocean height)  =  peak compressional loading '
        '(fault clamped → slip suppressed).',
        'Phase is computed via the Hilbert-transform instantaneous phase of the '
        'hourly-sampled predicted tide signal, then shifted by +180° so that 0° '
        'aligns with the tide minimum.',
    ]); y -= 0.01

    heading(ax, y, '3.  Rayleigh Statistics', level=2); y -= 0.05
    y = bullet(ax, y, [
        'R  (mean resultant length):  Each earthquake is represented as a unit '
        'vector at its tidal phase angle. R = magnitude of the vector mean / n. '
        'R = 0 → uniform (no preference); R = 1 → all at same phase.',
        'z  (Rayleigh test statistic):  z = n · R².  Combines both the strength '
        'of clustering (R) and the sample size (n).  Larger z = stronger overall '
        'evidence of non-uniform distribution.',
        'p-value:  Probability of observing z this large by chance if the true '
        'distribution were uniform.  p < 0.05 is conventionally significant; '
        'p ≈ 0 means the tidal signal is overwhelmingly unlikely to be noise.',
        'Combined ranking uses z = n · R², which naturally penalises high R '
        'from small samples and rewards consistent signals in large families.',
    ])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── PAGE 3 — Overall tidal phase histogram ─────────────────────────────────
    embed_png(pdf,
        f'{FIG_DIR}/13_tidal_triggering.png',
        title='Figure 1.  Overall Tidal Phase Distribution — All REDPy Families (2015–2021)',
        caption=(
            'Left: earthquake count per 20° tidal phase bin (red = above uniform, blue = below). '
            'Dashed line = expected count if uniform.  '
            'Right: percentage excess/deficit relative to uniform.  '
            'n = 29,536 events;  Rayleigh R = 0.314;  mean direction = 358.8° ≈ 0° (low tide);  p ≈ 0.  '
            'The strong peak at 0° confirms preferential triggering at peak extensional stress.'
        ))

    # ── PAGE 4 — Sanity check ──────────────────────────────────────────────────
    embed_png(pdf,
        f'{FIG_DIR}/13_tidal_sanity_check.png',
        title='Figure 2.  Method Verification',
        caption=(
            'Top: hourly tide level (m) for Jan 2015.  '
            'Middle: instantaneous tidal phase computed via Hilbert transform '
            '(0° = low tide, sawtooth oscillation confirms correct phase tracking).  '
            'Bottom: earthquake tidal phase histogram (all families, 2015–2021).'
        ))

    # ── PAGE 5 — Per-family ranking bar chart ──────────────────────────────────
    embed_png(pdf,
        f'{FIG_DIR}/14_tidal_sensitivity_ranking.png',
        title='Figure 3.  Tidal Sensitivity Ranking — Top 20 Families (by Rayleigh R)',
        caption=(
            'Families (top 20 by event count) ranked by Rayleigh R.  '
            'Green = high sensitivity, orange/red = low/no sensitivity.  '
            'p-values annotated above each bar.  '
            'Note: F428 and F57 have the highest R but smallest n; '
            'F2 and F16 combine large R with large n (most convincing evidence).  '
            'F537 (n=661) and F436 (n=435) show no statistically significant tidal preference.'
        ))

    # ── PAGE 6 — Per-family histograms ─────────────────────────────────────────
    embed_png(pdf,
        f'{FIG_DIR}/14_tidal_per_family.png',
        title='Figure 4.  Per-Family Tidal Phase Histograms (ranked by R)',
        caption=(
            'Each panel shows the percentage excess/deficit above a uniform distribution for one family '
            '(red = more earthquakes than expected, blue = fewer).  '
            'Families ordered left-to-right, top-to-bottom by decreasing Rayleigh R.  '
            'Background shading: green tint = high sensitivity, red tint = low.  '
            'Note F442 (rank 12) has its peak near ±180° (high tide) — opposite to all others.'
        ))

    # ── PAGE 7 — Tidal rank map ────────────────────────────────────────────────
    embed_png(pdf,
        f'{FIG_DIR}/15_tidal_rank_map.png',
        title='Figure 5.  Spatial Distribution — Top 20 Families Re-ranked by Tidal Sensitivity',
        caption=(
            'Map view (lon vs lat), right panel (depth vs lat), bottom panel (lon vs depth).  '
            'Families grouped into 4 sets of 5 by combined tidal rank (z = n·R²).  '
            'Labels include family number, tidal rank (#N), Rayleigh R, and event count.  '
            'Caldera rim shown as black outline; seismic stations as triangles.'
        ))

    # ── PAGE 8 — R vs time figure ─────────────────────────────────────────────
    embed_png(pdf,
        f'{FIG_DIR}/16_R_vs_time.png',
        title='Figure 6.  Rayleigh R Through Time — Tidal Sensitivity Evolution (2015–2021)',
        caption=(
            'Top: Rayleigh R in each 60-day sliding window (15-day step).  '
            'Blue dots = statistically significant (p < 0.05); grey = not significant.  '
            'Horizontal bars show period-mean R: pre-eruption (blue, R=0.390), '
            'co-eruption (red, R=0.103), post-eruption (green, R=0.347).  '
            'Middle: event count per window.  '
            'Bottom: mean tidal direction for significant windows (0° = low tide).  '
            'Red/orange dashed lines = eruption onset (Apr 24) and end (May 17, 2015).  '
            'Inset: 2015 zoom showing the sharp R collapse at eruption onset.'
        ))

    # ── PAGE 9 — R recovery figure ────────────────────────────────────────────
    embed_png(pdf,
        f'{FIG_DIR}/17_R_recovery.png',
        title='Figure 7.  Post-Eruption Recovery of Tidal Sensitivity (May 2015 – Dec 2021)',
        caption=(
            'Rayleigh R computed in sliding windows of 30 days (blue), 60 days (green), '
            'and 90 days (orange) with a 7-day step, starting from May 19, 2015.  '
            'Dotted lines show pre-eruption reference R = 0.390 (dark blue) and '
            'co-eruption R = 0.103 (red).  Yellow band = recovery zone between the two.  '
            'Bottom: event count per 7-day bin.  '
            'R quickly recovers to pre-eruption levels within ~1–2 months and '
            'remains elevated through 2021 with no clear long-term trend.'
        ))

    # ── PAGE 10 — Temporal findings text ──────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page_style(fig)
    ax = text_ax(fig)

    y = 0.96
    heading(ax, y, '4.  Temporal Evolution of Tidal Sensitivity', level=1); y -= 0.06

    heading(ax, y, '4.1  Period-averaged Rayleigh R', level=2); y -= 0.05

    # Mini summary table
    t_headers = ['Period', 'Date range', 'n events', 'R', 'Mean dir.', 'p']
    t_rows = [
        ('Pre-eruption',  'Jan 2015 – Apr 24 08:00',  '16,672', '0.390', '351°',  '≈ 0'),
        ('Co-eruption',   'Apr 24 – May 17, 2015',     '3,346',  '0.103', '199°',  '4.7×10⁻¹⁶'),
        ('Post-eruption', 'May 17, 2015 – Dec 2021',   '9,518',  '0.347', '16°',   '≈ 0'),
    ]
    col_x2 = [0.05, 0.22, 0.40, 0.51, 0.60, 0.70]
    row_colors = ['#D6EAF8', '#FADBD8', '#D5F5E3']
    ax.plot([0.04, 0.96], [y]*2, color=HEAD, lw=1.2, transform=ax.transAxes, clip_on=False)
    for xi, h in zip(col_x2, t_headers):
        ax.text(xi, y - 0.005, h, ha='left', va='top', transform=ax.transAxes,
                fontsize=9, fontweight='bold', color=HEAD)
    y -= 0.04
    ax.plot([0.04, 0.96], [y]*2, color=RULE, lw=0.7, transform=ax.transAxes, clip_on=False)
    for row, rc in zip(t_rows, row_colors):
        ax.add_patch(plt.Rectangle((0.04, y - 0.032), 0.92, 0.032,
                                    transform=ax.transAxes, color=rc,
                                    zorder=0, clip_on=False))
        for xi, v in zip(col_x2, row):
            ax.text(xi, y - 0.008, v, ha='left', va='top',
                    transform=ax.transAxes, fontsize=9, color=TXT)
        y -= 0.038
    ax.plot([0.04, 0.96], [y + 0.005]*2, color=RULE, lw=0.7,
            transform=ax.transAxes, clip_on=False)
    y -= 0.02

    heading(ax, y, '4.2  Eruption suppresses tidal triggering', level=2); y -= 0.045
    y = para(ax, y,
        'During the April 24 – May 17, 2015 co-eruption period, the Rayleigh R '
        'drops sharply from 0.390 to 0.103 — a 74% reduction. Although still '
        'statistically significant (p = 4.7×10⁻¹⁶, driven by the large n=3,346), '
        'the practical tidal influence is greatly diminished. The mean direction also '
        'shifts to ~199° (near high tide), suggesting the phase relationship inverts '
        'or becomes incoherent during the eruption. This is consistent with '
        'earthquake occurrence being dominated by rapidly evolving dike-intrusion '
        'and magma-pressurisation stress, which overwhelm the ~15–30 kPa tidal '
        'perturbation by orders of magnitude.', width=105); y -= 0.01

    heading(ax, y, '4.3  Pre-eruption period shows strongest sensitivity', level=2); y -= 0.045
    y = para(ax, y,
        'The pre-eruption R = 0.390 is the highest of the three periods, indicating '
        'that faults were near critically stressed before the eruption. In this '
        'state, the small periodic tidal stress increment is sufficient to advance '
        'the failure clock — a hallmark of a system close to instability. The '
        'progressive build-up of magmatic pressure beneath Axial in early 2015 '
        'likely brought fault segments to near-critical stress, making them '
        'exquisitely sensitive to even modest tidal perturbations.', width=105); y -= 0.01

    heading(ax, y, '4.4  Post-eruption partial recovery', level=2); y -= 0.045
    y = para(ax, y,
        'After the eruption, R recovers to 0.347 — strong and highly significant, '
        'but slightly below the pre-eruption value. Figure 7 shows the recovery '
        'in detail: R rapidly rebounds to pre-eruption levels within ~1–2 months '
        'of the eruption end and then remains broadly stable through 2021, with no '
        'systematic long-term trend. The slight overall reduction from pre-eruption '
        'could reflect: (1) partial stress release during the eruption leaving some '
        'faults farther from failure; (2) reorganisation of the fault network; or '
        '(3) influence of post-eruptive aseismic relaxation on background '
        'seismicity rates.', width=105)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── PAGE 10 — Catalog comparison figure ───────────────────────────────────
    embed_png(pdf,
        f'{FIG_DIR}/18_tidal_compare_catalogs.png',
        title='Figure 8.  Tidal Sensitivity: REDPy Repeating vs General Earthquake Population',
        caption=(
            'Top row: tidal phase histograms (% excess above uniform) for REDPy repeating earthquakes (blue, n=29,536) '
            'and the general DD+Felix catalog (orange, n=53,825).  Both peak sharply at 0° (low tide).  '
            'Middle: 60-day sliding-window R through time for both catalogs; '
            'solid horizontal bars = REDPy period means, dashed = general catalog means.  '
            'Bottom: period-averaged R comparison across all time, pre-, co-, and post-eruption.  '
            'REDPy repeating earthquakes are consistently ~0.01–0.02 higher in R than the general population, '
            'and show stronger tidal suppression during the eruption (R=0.103 vs 0.153).'
        ))

    # ── PAGE 12 — Summary table ────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page_style(fig)
    ax = text_ax(fig)

    heading(ax, 0.96, '5.  Summary of Per-Family Results', level=1)

    # Table
    headers = ['z-Rank', 'Family', 'n (2015–21)', 'R', 'z = n·R²', 'p', 'Interpretation']
    rows_data = [
        ( 1, 'F2',    13383, 0.455, 2771, '≈ 0',     'Largest family; dominant tidal signal'),
        ( 2, 'F16',    1241, 0.493,  302, '≈ 0',     'Strong signal, large n'),
        ( 3, 'F3',     5443, 0.173,  163, '≈ 0',     'Moderate R but very large n'),
        ( 4, 'F258',    381, 0.465,   82, '< 0.001', 'Strong signal'),
        ( 5, 'F8',      873, 0.286,   71, '< 0.001', 'Good evidence'),
        ( 6, 'F442',   1063, 0.233,   58, '< 0.001', '⚠ Mean dir ~212° — triggered at HIGH tide'),
        ( 7, 'F0',      498, 0.248,   31, '< 0.001', 'Moderate'),
        ( 8, 'F37',     190, 0.337,   22, '< 0.001', 'Moderate'),
        ( 9, 'F26',      94, 0.457,   20, '< 0.001', 'High R, modest n'),
        (10, 'F183',    140, 0.341,   16, '< 0.001', 'Moderate'),
        (11, 'F428',     25, 0.764,   15, '< 0.001', 'Highest R but tiny n — uncertain'),
        (12, 'F1',      147, 0.261,   10, '< 0.001', 'Moderate'),
        (13, 'F57',      18, 0.651,    8, '< 0.001', 'High R, tiny n — uncertain'),
        (14, 'F101',    270, 0.140,    5, '0.005',   'Marginal'),
        (15, 'F45',      88, 0.233,    5, '0.008',   'Marginal'),
        (16, 'F144',     77, 0.132,    1, '0.26',    'Not significant'),
        (17, 'F436',    435, 0.039,    1, '0.53',    'No tidal sensitivity'),
        (18, 'F185',     56, 0.105,    1, '0.54',    'Not significant'),
        (19, 'F537',    661, 0.023,    0, '0.71',    'Large family, NO tidal sensitivity'),
        (20, 'F706',      3,   None, None, 'n/a',   'Too few events in tide window'),
    ]

    col_x  = [0.03, 0.10, 0.19, 0.30, 0.39, 0.49, 0.59]
    col_w  = [0.07, 0.08, 0.10, 0.08, 0.09, 0.09, 0.41]
    y_hdr  = 0.88
    ax.plot([0.02, 0.98], [y_hdr + 0.015]*2, color=HEAD, lw=1.5,
            transform=ax.transAxes, clip_on=False)
    for xi, h in zip(col_x, headers):
        ax.text(xi, y_hdr, h, ha='left', va='top', transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', color=HEAD)
    ax.plot([0.02, 0.98], [y_hdr - 0.025]*2, color=RULE, lw=0.8,
            transform=ax.transAxes, clip_on=False)

    y_row = y_hdr - 0.035
    for i, (rank, fam, n, R, z, p, interp) in enumerate(rows_data):
        bg = '#EEF4FB' if i % 2 == 0 else BG
        ax.add_patch(plt.Rectangle((0.02, y_row - 0.005), 0.96, 0.030,
                                    transform=ax.transAxes,
                                    color=bg, zorder=0, clip_on=False))
        R_s = f'{R:.3f}' if R is not None else '—'
        z_s = f'{z:,.0f}' if z is not None else '—'
        vals = [str(rank), fam, f'{n:,}', R_s, z_s, p, interp]
        txt_color = '#CC2200' if '⚠' in interp else TXT
        for xi, v in zip(col_x, vals):
            ax.text(xi, y_row + 0.018, v, ha='left', va='center',
                    transform=ax.transAxes, fontsize=7.5, color=txt_color)
        y_row -= 0.033

    ax.plot([0.02, 0.98], [y_row + 0.015]*2, color=RULE, lw=0.8,
            transform=ax.transAxes, clip_on=False)

    ax.text(0.05, y_row - 0.01,
            '★  z = n · R²  combines signal strength (R) and sample size (n).  '
            'Ranking by z gives the most statistically robust ordering.',
            ha='left', va='top', transform=ax.transAxes,
            fontsize=8, color=SUB, style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── PAGE 9 — Key Findings ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page_style(fig)
    ax = text_ax(fig)

    y = 0.96
    heading(ax, y, '6.  Key Findings', level=1); y -= 0.06

    heading(ax, y, '6.1  Strong overall tidal modulation', level=2); y -= 0.045
    y = para(ax, y,
        'Across all 29,536 REDPy events in 2015–2021, the Rayleigh R = 0.314 with '
        'p ≈ 0, confirming that repeating earthquakes at Axial Seamount are strongly '
        'concentrated near low tide (tidal phase ≈ 0°). The peak bin (~0°) contains '
        '~90% more events than expected under a uniform distribution, while the '
        'high-tide bins (±180°) are depleted by ~40%. This is fully consistent with '
        'Tan et al. (2019), who identified tidal triggering as a primary control on '
        'repeating earthquake timing at Axial.', width=105); y -= 0.01

    heading(ax, y, '6.2  Most tidally sensitive families (by z = n·R²)', level=2); y -= 0.045
    y = bullet(ax, y, [
        'F2  (n=13,383, R=0.455, z=2771):  The largest family and dominant '
        'contributor to the overall signal. Extremely robust evidence.',
        'F16  (n=1,241, R=0.493, z=302):  Second strongest. High R sustained '
        'over a large sample — very reliable.',
        'F3  (n=5,443, R=0.173, z=163):  Modest R but large n places it third '
        'in combined evidence.',
    ]); y -= 0.01

    heading(ax, y, '6.3  Anomalous family — F442', level=2); y -= 0.045
    y = para(ax, y,
        'F442 (n=1,063, R=0.233, z=58, p<0.001) is the only family with a '
        'statistically significant tidal preference at HIGH tide (mean direction '
        '~212°, i.e. opposite to all others). This could indicate a fault segment '
        'with a different geometry (e.g. a compressional patch or thrust) where '
        'increased normal stress at high tide promotes rather than inhibits slip, '
        'or it could reflect coupling to a local pressure anomaly from the magmatic '
        'system.', width=105); y -= 0.01

    heading(ax, y, '6.4  Tidally insensitive families', level=2); y -= 0.045
    y = bullet(ax, y, [
        'F537  (n=661, R=0.023, p=0.71):  Large family with essentially NO '
        'tidal modulation. Slip rate appears driven by non-tidal processes '
        '(aseismic creep, magmatic pressurisation, or post-eruption relaxation).',
        'F436  (n=435, R=0.039, p=0.53):  Similarly insensitive despite a '
        'substantial event count.',
        'Three additional families (F144, F185, F706) also show p > 0.05.',
    ]); y -= 0.01

    heading(ax, y, '6.5  Tidal sensitivity suppressed then partially recovered across eruption', level=2); y -= 0.045
    y = para(ax, y,
        'Figure 6 shows the full temporal evolution. R peaks in the pre-eruption '
        'period (R=0.390), collapses to R=0.103 during the co-eruption, then '
        'recovers to a stable R≈0.35 post-eruption. This three-phase pattern '
        'mirrors the stress cycle of a magmatic system: stress accumulation → '
        'eruption-driven stress release → re-loading. See Section 4 for detailed '
        'discussion.', width=105); y -= 0.01

    heading(ax, y, '6.6  REDPy vs general earthquake population', level=2); y -= 0.045
    y = para(ax, y,
        'Figure 8 directly compares tidal sensitivity between REDPy repeating '
        'earthquakes (n=29,536, R=0.314) and the full DD+Felix catalog '
        '(n=53,825, R=0.306). Both are strongly concentrated at low tide with '
        'nearly identical mean directions (~359°), showing that tidal triggering '
        'is a property of the entire Axial seismic system, not unique to repeating '
        'earthquakes. However, REDPy events consistently show slightly higher R '
        '(by ~0.01–0.02) across all time periods, and stronger tidal suppression '
        'during the eruption (R=0.103 vs 0.153). This suggests that repeating '
        'earthquakes on persistent, well-defined fault patches are marginally more '
        'stress-sensitive than the general population, consistent with their '
        'near-critical stress state required for repeated slip.', width=105); y -= 0.01

    heading(ax, y, '6.7  Spatial pattern', level=2); y -= 0.045
    y = para(ax, y,
        'No obvious geographic separation exists between tidally sensitive and '
        'insensitive families (Figure 5). Sensitivity appears to depend on local '
        'fault mechanics (orientation, normal stress, pore pressure) rather than '
        'position within the caldera. Further analysis correlating sensitivity '
        'with focal mechanisms or depth could clarify the controlling factors.',
        width=105)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── PAGE 10 — References ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    page_style(fig)
    ax = text_ax(fig)

    y = 0.96
    heading(ax, y, '7.  References', level=1); y -= 0.06

    refs = [
        ('Tan et al. (2019)',
         'Y. J. Tan, M. Tolstoy, F. Waldhauser, W. S. D. Wilcock.  '
         '"Dynamics of a seafloor-spreading episode at the East Pacific Rise."  '
         'Science, 367(6473), 86–89.  DOI: 10.1126/science.aax0965'),
        ('Wilcock (2001)',
         'W. S. D. Wilcock.  "Tidal triggering of microearthquakes on the '
         'Juan de Fuca Ridge."  Geophys. Res. Lett., 28(20), 3999–4002.'),
        ('Stroup et al. (2007)',
         'D. F. Stroup, D. R. Bohnenstiehl, M. Tolstoy, F. Waldhauser, '
         'E. E. E. Hooft.  "Pulse of the seafloor: Tidal triggering of '
         'microearthquakes at 9°50′N East Pacific Rise."  '
         'Geophys. Res. Lett., 34, L15301.'),
        ('Zar (1999)',
         'J. H. Zar.  Biostatistical Analysis, 4th ed.  Prentice Hall.  '
         '[Rayleigh test approximation, Chapter 27.]'),
        ('REDPy',
         'Hotovec-Ellis & Jeffries (2016).  "REDPy: Repeating Earthquake '
         'Detector in Python."  Seismol. Res. Lett., 87(5), 1190–1198.'),
    ]

    for ref, txt in refs:
        ax.text(0.05, y, ref + ':', ha='left', va='top',
                transform=ax.transAxes, fontsize=10,
                fontweight='bold', color=SUB)
        y -= 0.035
        lines = textwrap.wrap(txt, 100)
        for line in lines:
            ax.text(0.07, y, line, ha='left', va='top',
                    transform=ax.transAxes, fontsize=9, color=TXT)
            y -= 0.032
        y -= 0.01

    ax.plot([0.05, 0.95], [0.06, 0.06], color=RULE, lw=0.8,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.5, 0.03,
            f'Report generated {TODAY}  ·  Axial Seamount Repeating Earthquake Project',
            ha='center', va='bottom', transform=ax.transAxes,
            fontsize=8.5, color='#666666')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # PDF metadata
    d = pdf.infodict()
    d['Title']   = 'Tidal Triggering of Repeating Earthquakes at Axial Seamount'
    d['Author']  = 'M. Zhang'
    d['Subject'] = 'Rayleigh statistics, tidal phase, REDPy, Axial Seamount'
    d['CreationDate'] = datetime.datetime.now()

print(f'Report saved: {OUT_PDF}')
