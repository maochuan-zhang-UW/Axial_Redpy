"""
Regenerate map and FM images for all families using updated cross-section layout.
FM plots are only saved when qual A/B focal mechanisms exist (pre-2022 catalog).
"""
import redpy
import redpy.outputs.mapping as mapping

CONFIG     = 'axial_settings.cfg'
CAT_CSV    = 'axial_catalog_dd.csv'
FM_MAT     = ('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/'
              'G_2015_HASH_Po_Clu_FM.mat')

d = redpy.Detector(CONFIG)
d.open()

nfam = len(d)
print(f'Regenerating maps and FM plots for {nfam} families...')

fm_catalog = mapping.load_fm_catalog(FM_MAT)
print(f'FM catalog loaded: {len(fm_catalog)} events')

map_ok = map_skip = fm_ok = fm_skip = 0
for fnum in range(nfam):
    # Location cross-section map
    try:
        mapping.create_axial_map(d, fnum, CAT_CSV)
        map_ok += 1
    except Exception as e:
        print(f'  map {fnum} ERROR: {e}')
        map_skip += 1

    # FM cross-section plot (silently skipped if no A/B matches)
    try:
        mapping.create_axial_fm_plot(d, fnum, CAT_CSV, fm_catalog)
        fm_ok += 1
    except Exception as e:
        print(f'  fm  {fnum} ERROR: {e}')
        fm_skip += 1

    if (fnum + 1) % 50 == 0:
        print(f'  {fnum+1}/{nfam} done  '
              f'(map: {map_ok} ok / {map_skip} skip, '
              f'fm: {fm_ok} ok / {fm_skip} skip)')

d.close()
print(f'\nDone. map: {map_ok} ok / {map_skip} errors, '
      f'fm: {fm_ok} generated / {fm_skip} errors')
