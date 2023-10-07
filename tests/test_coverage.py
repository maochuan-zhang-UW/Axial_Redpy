"""
Tests for coverage checking and script functionality.

Currently, this is intended to be run with 'coverage run -m pytest' in the
root 'REDPy' directory, and should result in all passing tests and 100%
coverage (minus excludes in .coveragerc and 'pragma: no cover' in-line
comments). The tests cover a known dataset and behaviors that should be
invariant to future changes. The only major script that is not currently
covered is redpy-remove-family-gui.
"""

import os
import shutil

import pandas as pd

import redpy


TEST_PATH = os.path.join('.','tests')
OUT_PATH = os.path.join(TEST_PATH, 'out')
RUN_PATH = os.path.join(OUT_PATH, 'test')


def check_table_lengths(
        detector=None, configfile=os.path.join(TEST_PATH, 'test0.cfg'),
        lengths=0):
    """
    Check the lengths of tables with a known set of lengths.

    Parameters
    ----------
    detector : Detector object, optional
        Primary interface for handling detections.
    configfile : str, optional
        Name of configuration file to read.
    lengths : int, int list
        If a list, specify the expected lengths of the tables in the order
        listed upon opening (i.e., [triggers, orphans, repeaters, families,
        junk, deleted, correlation pairs]). If an integer, set all expected
        lengths to be the value given (e.g., 0 for all empty).

    Returns
    -------
    bool
        True only if all lengths specify match the contents of the file
        given by either detector or configfile.

    """
    if detector is None:
        detector = redpy.Detector(configfile, verbose=False, opened=True)
    if isinstance(lengths, int):
        lengths = [lengths]*7
    all_same = True
    print('\nTesting if tables have expected lengths:')
    for i, table in enumerate(detector.get('tables')):
        print(f'{table}:  {len(detector.get(table))} == {lengths[i]} -> '
              f'{len(detector.get(table)) == lengths[i]}')
        if len(detector.get(table)) != lengths[i]:  # pragma: no cover
            all_same = False
    if configfile:
        detector.close()
    return all_same


def clean():  # pragma: no cover
    """Clean output and .cache directory if they exist."""
    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)
    if os.path.exists('.cache'):
        shutil.rmtree('.cache')
    os.mkdir(OUT_PATH)


def print_section_header(header):
    """Make section headers for tests look nice."""
    print(f'\n{header:-^80s}')


def test_settings():
    """Confirm settings.cfg matches default settings."""
    clean()  # Clean up first
    print_section_header('confirm default settings')
    detector = redpy.Detector(configfile='settings.cfg')
    assert ('with all default settings' in str(repr(detector)))
    assert detector.stats() is None


def test_testing_settings():
    """Confirm test*.cfg match expectations."""
    print_section_header('confirm test settings')
    detector = redpy.Detector(configfile=os.path.join(TEST_PATH, 'test0.cfg'))
    assert """with custom settings:
  title=TEST
  groupname=test
  filename=./tests/out/testtable.h5
  outputpath=./tests/out/
  server=fdsnws://http://service.iris.edu
  nsta=5
  station=['YEL', 'EDM', 'SHW', 'HSR', 'JUN']
  channel=['EHZ', 'EHZ', 'EHZ', 'EHZ', 'EHZ']
  network=['UW', 'UW', 'UW', 'UW', 'UW']
  location=['--', '--', '--', '--', '--']
  preload=0.1
  samprate=50.0
  nstac=4
  oratiomax=0.06
  telefi=-0.25
  winlen=512
  ncor=3
  maxorph=0.5
  occurbin=2.0
  anotfile=example_annotation.csv
  printsta=3
  max_famlen=100
  corr_nrecent=5
  corr_nyoungest=5
  corr_nlargest=10
  merge_ratio=0.6
  always_verbose=True
  stalats=[46.20955 46.17428 46.19347 46.19717 46.14706]
  stalons=[-122.18899 -122.18065 -122.23635 -122.15121 -122.15243]
  matchmax=10""" in str(repr(detector))
    detector = redpy.Detector(configfile=os.path.join(TEST_PATH, 'test1.cfg'))
    assert """with custom settings:
  title=TEST
  groupname=test
  filename=./tests/out/testtable.h5
  outputpath=./tests/out/
  server=file
  searchdir=./tests/data/
  filepattern=*.mseed
  preload=0.25
  nsta=5
  station=['YEL', 'EDM', 'SHW', 'HSR', 'JUN']
  channel=['EHZ', 'EHZ', 'EHZ', 'EHZ', 'EHZ']
  network=['UW', 'UW', 'UW', 'UW', 'UW']
  location=['', '', '', '', '']
  offset=0.17,0.50,0.60,0.49,1.09
  samprate=50.0
  nstac=4
  winlen=512
  ncor=3
  maxorph=0.5
  mrecplot=0.5
  anotfile=example_annotation.csv
  printsta=3
  verbosecatalog=True
  max_famlen=100
  corr_nrecent=5
  corr_nyoungest=5
  corr_nlargest=10
  merge_ratio=0.6
  always_verbose=True
  checkcomcat=True
  stalats=[46.20955 46.17428 46.19347 46.19717 46.14706]
  stalons=[-122.18899 -122.18065 -122.23635 -122.15121 -122.15243]
  matchmax=10""" in str(repr(detector))
    detector = redpy.Detector(configfile=os.path.join(TEST_PATH, 'test2.cfg'))
    assert """with custom settings:
  title=TEST
  groupname=test
  filename=./tests/out/testtable.h5
  outputpath=./tests/out/
  server=file
  searchdir=./tests/data/
  filepattern=*.mseed
  nsta=6
  station=['YEL', 'EDM', 'SHW', 'HSR', 'JUN', 'SOS']
  channel=['EHZ', 'EHZ', 'EHZ', 'EHZ', 'EHZ', 'EHZ']
  network=['UW', 'UW', 'UW', 'UW', 'UW', 'UW']
  location=['', '', '', '', '', '']
  offset=0.17,0.50,0.60,0.49,1.09,1.05
  samprate=50.0
  nstac=4
  winlen=512
  ncor=3
  maxorph=0.5
  occurbin=7.0
  mrecbin=1.0
  anotfile=example_annotation.csv
  printsta=3
  bokehendtime=now
  timeline_vs=triggers
  amplims=family
  verbosecatalog=True
  max_famlen=100
  corr_nrecent=5
  corr_nyoungest=5
  corr_nlargest=10
  merge_ratio=0.6
  always_verbose=True
  stalats=[46.20955 46.17428 46.19347 46.19717 46.14706 46.24386]
  stalons=[-122.18899 -122.18065 -122.23635 -122.15121 -122.15243 -122.13787]
  matchmax=10""" in str(repr(detector))


def test_make_meta():
    """Confirm redpy-make-meta makes meta.html."""
    print_section_header('redpy-make-meta')
    redpy.make_meta(path=OUT_PATH, topath='.', verbose=True)
    assert os.path.getsize(os.path.join(OUT_PATH, 'meta.html'))


def test_initialize():
    """Confirm redpy-initialize makes empty .h5 file."""
    print_section_header('redpy-initialize')
    redpy.initialize(
        configfile=os.path.join(TEST_PATH, 'test0.cfg'), verbose=True)
    assert check_table_lengths(configfile=os.path.join(TEST_PATH, 'test0.cfg'))


def test_catfill_junk():
    """Confirm redpy-catfill with known and forced junk."""
    print_section_header('redpy-catfill with junk')
    redpy.catfill(os.path.join(TEST_PATH, 'data', 'known_junk.txt'),
                  configfile=os.path.join(TEST_PATH, 'test0.cfg'),
                  verbose=True, delimiter='\t', name='Trigger Time (UTC)')
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test0.cfg'),
        lengths=[2, 2, 0, 0, 3, 0, 0])


def test_plot_junk():
    """Confirm redpy-plot-junk creates known outputs."""
    print_section_header('redpy-plot-junk')
    redpy.plot_junk(
        configfile=os.path.join(TEST_PATH, 'test0.cfg'), verbose=True)
    assert os.path.exists(
        os.path.join(RUN_PATH, 'junk', '20040923232035-kurt.png'))
    assert os.path.exists(
        os.path.join(RUN_PATH, 'junk', '20040923232730-both.png'))
    assert os.path.exists(
        os.path.join(RUN_PATH, 'junk', '20040923233515-freq.png'))
    assert os.path.getsize(os.path.join(RUN_PATH, 'catalog_junk.txt'))


def test_clear_junk():
    """Confirm redpy-clear-junk clears the junk table."""
    print_section_header('redpy-clear-junk')
    redpy.clear_junk(
        configfile=os.path.join(TEST_PATH, 'test0.cfg'), verbose=True)
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test0.cfg'),
        lengths=[2, 2, 0, 0, 0, 0, 0])


def test_catfill_force():
    """Confirm redpy-catfill properly forces triggers from catalog."""
    print_section_header('redpy-catfill with force')
    redpy.initialize(configfile=os.path.join(TEST_PATH, 'test0.cfg'))
    redpy.catfill(os.path.join(RUN_PATH, 'testcat.csv'),
                  configfile=os.path.join(TEST_PATH, 'test0.cfg'),
                  verbose=True, arrival=True, force=True, query=True,
                  starttime='2004-09-23T23:00', endtime='2004-09-24T00:00')
    assert os.path.getsize(os.path.join(RUN_PATH, 'testcat.csv'))
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test0.cfg'),
        lengths=[6, 3, 3, 1, 0, 0, 2])


def test_compare_catalog():
    """Confirm redpy-compare-catalog matches all event in testcat.csv."""
    print_section_header('redpy-compare-catalog')
    redpy.compare_catalog(
        os.path.join(RUN_PATH, 'testcat.csv'), arrival=False,
        configfile=os.path.join(TEST_PATH, 'test0.cfg'),
        delimiter=',', include_missing=True, junk=True, maxdtoffset=0.1,
        name='Arrival', outfile=os.path.join(OUT_PATH, 'matches.csv'),
        verbose=True)
    df = pd.read_csv(os.path.join(OUT_PATH, 'matches.csv'))
    assert (df['Family'] == ['orphan', 0, 0, 'orphan', 0, 'orphan']).all()


def test_remove_family():
    """Confirm redpy-remove-family removes the family."""
    print_section_header('redpy-remove-family')
    redpy.remove_family(
        0, configfile=os.path.join(TEST_PATH, 'test0.cfg'), verbose=True)
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test0.cfg'),
        lengths=[6, 3, 0, 0, 0, 1, 0])


def test_backfill_deleted():
    """Confirm redpy-backfill checks against deleted family."""
    print_section_header('redpy-backfill deleted')
    redpy.backfill(configfile=os.path.join(TEST_PATH, 'test0.cfg'),
                   verbose=True, starttime='2004-09-23T23:00',
                   endtime='2004-09-24T00:00', nsec=1800)
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test0.cfg'),
        lengths=[11, 8, 0, 0, 16, 1, 0])


def test_backfill_from_file():
    """Confirm redpy-backfill can read from local .mseed files."""
    print_section_header('redpy-backfill from file')
    redpy.backfill(configfile=os.path.join(TEST_PATH, 'test1.cfg'),
                   verbose=True, starttime='2004-10-01T00:00',
                   endtime='2004-10-01T01:00', nsec=3600)
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'),
        lengths=[21, 10, 0, 0, 16, 1, 0])


def test_backfill_updates():
    """Confirm repeated .update() calls update properly."""
    print_section_header('Detector.update() with updated span')
    detector = redpy.Detector(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'), verbose=True,
        opened=True)
    detector.update(
        'backfill', tstart='2004-10-01T01:00', tend='2004-10-01T02:00')
    detector.update(
        'backfill', tend='2004-10-02')
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'),
        lengths=[270, 55, 131, 23, 17, 1, 355])
    detector.close()


def test_backfill_empty_forget():
    """Confirm behavior with no data and forgetting remembered columns."""
    print_section_header('Detector.update() past mseed end date + forgetting')
    detector = redpy.Detector(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'), verbose=True,
        opened=True)
    detector.update(
        'backfill', tstart='2004-10-03T00:00', tend='2004-10-03T01:00')
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'),
        lengths=[270, 55, 131, 23, 17, 1, 355])
    detector.get('rtable').forget(['id', 'startTimeMPL'])
    assert list(detector.get('rtable').columns_in_memory.keys()) == [
        'windowAmp', 'windowCoeff', 'startTime', 'windowStart']
    detector.get('otable').forget('all')
    assert detector.get('otable').columns_in_memory == {}
    detector.close()


def test_distant_families(capsys):
    """Confirm output for distant families."""
    with capsys.disabled():
        print_section_header('redpy-distant-families')
    redpy.distant_families(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'), verbose=True,
        findphrase='Amboy', percent=90)
    captured = capsys.readouterr()
    assert ('Family    4 : L  2 | R  0 | T  0 | F  2 | Distant   0.0% | '
            '"Amboy" 100.0%') in captured.out
    assert ('Family    6 : L  1 | R  0 | T  0 | F  1 | Distant   0.0% | '
           '"Amboy" 100.0%') in captured.out
    assert ('Family    7 : L  0 | R  0 | T  1 | F  0 | Distant 100.0% | '
           '"Amboy"   0.0%') in captured.out
    assert ('Family   10 : L  1 | R  0 | T  0 | F  1 | Distant   0.0% | '
           '"Amboy" 100.0%') in captured.out
    assert ('Family   15 : L  1 | R  0 | T  0 | F  1 | Distant   0.0% | '
           '"Amboy" 100.0%') in captured.out
    assert '90%+ Teleseismic:\n 7\n\n' in captured.out
    assert '90%+ Regional+Teleseismic:\n 7\n\n' in captured.out
    assert ('90%+ Regional:\n\n\n90%+ Regional (ignore Teleseisms):\n\n\n'
            ) in captured.out
    assert '3+ Regional Matches: \n\n\n' in captured.out
    assert '90%+ Containing Phrase "Amboy": \n 4 6 10 15\n\n' in captured.out
    with capsys.disabled():
        print(captured.out)


def test_write_family_locations():
    """Confirm output for family locations."""
    print_section_header('redpy-write-family-locations')
    redpy.write_family_locations(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'), outfile='famlocs.csv',
        distant=False, regional=False, verbose=True)
    df = pd.read_csv(os.path.join(RUN_PATH, 'famlocs.csv'))
    assert len(df) == 23
    assert df['#Members'].sum() == 131
    assert df['#Located'].sum() == 5
    assert df['Longitude'][4] == -122.1881665


def test_create_report():
    """Confirm reports created."""
    print_section_header('redpy-create-report 1')
    redpy.create_report(
        0, configfile=os.path.join(TEST_PATH, 'test1.cfg'), verbose=True,
        ordered=False, matrixtofile=False, skip=True)
    assert os.path.getsize(os.path.join(RUN_PATH, 'reports', '0-report.html'))
    print_section_header('redpy-create-report 2')
    redpy.create_report(
        7, configfile=os.path.join(TEST_PATH, 'test1.cfg'), verbose=True,
        ordered=True, matrixtofile=True, skip=False)
    assert os.path.getsize(os.path.join(RUN_PATH, 'reports', '7-cmatrix.npy'))
    assert os.path.getsize(os.path.join(RUN_PATH, 'reports', '7-evtimes.npy'))
    assert os.path.getsize(os.path.join(RUN_PATH, 'reports', '7-report.html'))


def test_create_pdf_family():
    """Confirm pdf family created."""
    print_section_header('redpy-create-pdf-family')
    redpy.create_pdf_family(
        7, configfile=os.path.join(TEST_PATH, 'test1.cfg'), verbose=True,
        starttime='2004-10-01T08:00', endtime='2004-10-01T23:30')
    assert os.path.getsize(os.path.join(RUN_PATH, 'families', 'fam7.pdf'))


def test_create_pdf_overview():
    """Confirm pdf overview created."""
    print_section_header('redpy-create-pdf-overview')
    redpy.create_pdf_overview(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'), verbose=True,
        usehrs=True, binsize=2, starttime='2004-10-01T06:00',
        endtime='2004-10-01T20:00', minmembers=0, occurheight=1,
        plotformat='eqrate,fi,occurrence,occurrencefi,longevity')
    redpy.create_pdf_overview(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'), verbose=True,
        usehrs=False, minmembers=5, occurheight=1)
    assert os.path.getsize(os.path.join(RUN_PATH, 'overview.pdf'))


def test_remove_small_family(capsys):
    """Confirm behavior of small family removal."""
    with capsys.disabled():
        print_section_header('redpy-remove-small-family')
    redpy.remove_small_family(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'), minmembers=5,
        maxage=0, seedtime='2004-10-10', listonly=True, verbose=True)
    captured = capsys.readouterr()
    assert """Small families : 0 1 3 6 9 10 11 12 15 16 17 18 19 20 21 22
# Families     : 16/23
# Repeaters    : 37/131 (28.2%)""" in captured.out
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'),
        lengths=[270, 55, 131, 23, 17, 1, 355])
    with capsys.disabled():
        print(captured.out)
        redpy.remove_small_family(
            configfile=os.path.join(TEST_PATH, 'test1.cfg'), minmembers=5,
            maxage=0, seedtime='', listonly=False, verbose=True)
    assert check_table_lengths(
        configfile=os.path.join(TEST_PATH, 'test1.cfg'),
        lengths=[270, 55, 94, 7, 17, 1, 328])


def test_extend_table():
    """Confirm table can be extended."""
    print_section_header('redpy-extend-table')
    detector = redpy.Detector(
        os.path.join(TEST_PATH, 'test1.cfg'), opened=True)
    assert len(detector.get('rtable', 'windowFFT', 0)) == 2560
    detector.close()
    redpy.extend_table(
        os.path.join(TEST_PATH, 'test1.cfg'),
        os.path.join(TEST_PATH, 'test2.cfg'), verbose=True, noplot=False)
    detector = redpy.Detector(
        os.path.join(TEST_PATH, 'test2.cfg'), opened=True)
    assert len(detector.get('rtable', 'windowFFT', 0)) == 3072
    detector.close()


def test_force_plot():
    """Confirm behavior of force plotting."""
    print_section_header('redpy-force-plot')
    fam_path = os.path.join(RUN_PATH, 'families')
    shutil.rmtree(fam_path)
    os.mkdir(fam_path)
    redpy.force_plot(
        configfile=os.path.join(TEST_PATH, 'test2.cfg'), famplot=True,
        startfam=-2, endfam=-1)
    assert not os.path.exists(os.path.join(fam_path, '6.png'))
    assert not os.path.exists(os.path.join(fam_path, '0.png'))
    assert not os.path.exists(os.path.join(fam_path, '5.html'))
    assert os.path.getsize(os.path.join(fam_path, '5.png'))
    redpy.force_plot(
        configfile=os.path.join(TEST_PATH, 'test2.cfg'), famplot=True,
        startfam=6)
    assert not os.path.exists(os.path.join(fam_path, '0.png'))
    assert not os.path.exists(os.path.join(fam_path, '6.html'))
    assert os.path.getsize(os.path.join(fam_path, '6.png'))
    redpy.force_plot(
        configfile=os.path.join(TEST_PATH, 'test0.cfg'), htmlonly=True,
        resetlp=True, plotall=True)
    assert os.path.getsize(os.path.join(fam_path, '0.html'))
    assert not os.path.exists(os.path.join(fam_path, '0.png'))
    redpy.force_plot(configfile=os.path.join(TEST_PATH, 'test2.cfg'),
                     verbose=True, plotall=True)
    assert os.path.getsize(os.path.join(fam_path, '0.png'))


def test_overwrite_empty():
    """Confirm initialize overwrites with empty h5 file."""
    print_section_header('overwrite with empty')
    detector = redpy.Detector(
        configfile=os.path.join(TEST_PATH, 'test2.cfg'), verbose=True)
    detector.initialize()
    detector.output('force', plotall=True)
    assert check_table_lengths(configfile=os.path.join(TEST_PATH, 'test2.cfg'))
    detector.close()


def test_save_external():
    """Confirm save external writes empty catalog with header."""
    print_section_header('confirm save catalog')
    detector = redpy.Detector(os.path.join(TEST_PATH, 'test2.cfg'))
    detector.set('nsec', 1)
    catalog = redpy.locate.save_external_catalog(
        detector, os.path.join(OUT_PATH, 'emptycat.csv'))
    assert len(catalog) == 0
    assert os.path.getsize(os.path.join(OUT_PATH, 'emptycat.csv'))
    detector.close()
