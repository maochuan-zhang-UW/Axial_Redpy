# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling REDPy Trigger() object.

The Trigger() object holds the waveform data and associated metadata for
a single trigger, and has methods for populating itself into a table.
"""


class Trigger():
    """
    Primary interface for handling individual triggers.

    Attributes
    ----------
    trig_time : UTCDateTime object
        Time of original trigger.
    waveforms : Stream object
        Ordered waveform data for all stations/channels for the trigger.
    ratio : float
        Maximum value of STA/LTA ratio; used to calculate orphan expiration.
    """

    def __init__(self, detector, trig_time, ratio, stream):
        """
        Slice out waveform and set attribute structure.

        Parameters
        ----------
        detector : Detector object
            Primary interface for handling detections.
        trig : UTCDateTime
            Time of original trigger
        ratio : float
            Maximum value of STA/LTA ratio from triggering algorithm.
        stream : Stream object
            Ordered waveform data for all stations/channels for entire
            window used for triggering.

        """
        self.trig_time = trig_time
        self.waveforms = stream.slice(
            trig_time - detector.get('ptrig'),
            trig_time + detector.get('atrig') + 2/detector.get('samprate')
            ).copy()
        self.ratio = ratio

    def concat(self):
        """Concatenate waveform data together."""
        print('This will concatenate data together')

    def fft(self):
        """Calculate FFT."""
        print('This is going to calculate the FFT and FI')

#         # Trim, pad with zeros
#         tr = tr.trim(ttime - detector.get('ptrig'),
#                      ttime + detector.get('atrig')
#                      + 2/detector.get('samprate'), pad=True,
#                      fill_value=0)
#         for s in range(len(tr)):
#             # Cut out exact number of samples
#             tr[s].data = tr[s].data[0:detector.get('wshape')]
#             # Demean
#             tr[s].data -= np.mean(tr[s].data)
#             # !!! Preserve zeroes (replace demean)
#             #tr[s].data[tr[s].data!=0] -= np.mean(
#             #                              tr[s].data[tr[s].data!=0])
#             # Append
#             if s > 0:
#                 tr[0].data = np.append(tr[0].data, tr[s].data)
#         # Set 'maxratio' for orphan expiration
#         tr[0].stats.maxratio = ratios[n]
