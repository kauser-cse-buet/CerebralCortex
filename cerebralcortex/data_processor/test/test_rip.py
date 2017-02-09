# Copyright (c) 2016, MD2K Center of Excellence
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import datetime
import gzip
import os
import pytz
from cerebralcortex.data_processor.signalprocessing.rip import up_down_intercepts, filter_intercept_outlier, generate_peak_valley, correct_valley_position, correct_peak_position, remove_close_valley_peak_pair, filter_expiration_duration_outlier, filter_small_amp_expiration_peak_valley, filter_small_amp_inspiration_peak_valley, compute_peak_valley
from cerebralcortex.kernel.datatypes.datapoint import DataPoint
from cerebralcortex.kernel.datatypes.datastream import DataStream
from cerebralcortex.data_processor.signalprocessing.alignment import timestamp_correct
import unittest
from cerebralcortex.data_processor.signalprocessing.vector import smooth, moving_average_curve
import numpy as np
from typing import List

class TestPeakValleyComputation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestPeakValleyComputation, cls).setUpClass()
        tz = pytz.timezone('US/Eastern')
        cls.rip = []
        cls._fs = 21.33
        cls._smoothing_factor = 5
        cls._time_window = 8
        cls._expiration_amplitude_threshold_perc = 0.10
        cls._threshold_expiration_duration = 0.312
        cls._max_amplitude_change_peak_correction = 30
        cls._inspiration_amplitude_threshold_perc = 0.10
        cls._min_neg_slope_count_peak_correction = 4
        cls._minimum_peak_to_valley_time_diff = 0.31

        cls._window_length = int(round(cls._time_window * cls._fs))
        with gzip.open(os.path.join(os.path.dirname(__file__), 'res/rip.csv.gz'), 'rt') as f:
            for l in f:
                values = list(map(int, l.split(',')))
                cls.rip.append(
                    DataPoint.from_tuple(datetime.datetime.fromtimestamp(values[0] / 1000000.0, tz=tz), values[1]))
        cls.rip_datastream = DataStream(None,None)
        cls.rip_datastream.datapoints = cls.rip

    def test_smooth(self):
        ds = DataStream(None, None)
        ds.datapoints = self.rip

        result_smooth = smooth(ds.datapoints, self._smoothing_factor)
        sample_smooth_python = [i.sample for i in result_smooth[:5000]]

        sample_smooth_matlab = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'res/testmatlab_rip_smooth.csv'),delimiter=',', )
        self.assertTrue(np.alltrue(np.round(sample_smooth_matlab) == np.round(sample_smooth_python)))


    def test_moving_average_curve(self):
        ds = DataStream(None, None)
        ds.datapoints = self.rip

        data_smooth = smooth(ds.datapoints, self._smoothing_factor)
        result = moving_average_curve(data_smooth, self._window_length)

        sample_mac_python = [i.sample for i in result[:5000]]
        sample_mac_matlab = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'res/testmatlab_mac_sample.csv'),delimiter=',', )
        for i in range(0,len(sample_mac_matlab)):
            self.assertAlmostEqual(sample_mac_matlab[i],sample_mac_python[i],delta=0.1)


    def test_up_down_intercepts(self):
        ds = DataStream(None, None)
        ds.datapoints = self.rip

        data_smooth = smooth(ds.datapoints, self._smoothing_factor)
        data_mac = moving_average_curve(data_smooth, self._window_length)
        up_intercepts, down_intercepts = up_down_intercepts(data=data_smooth, mac=data_mac)

        ui_sample_python = [i.sample for i in up_intercepts]
        ui_start_time_python = [i.start_time for i in up_intercepts]
        sample_mac_matlab = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'res/testmatlab_rip_mac.csv'),delimiter=',', )

    def test_compute_peak_valley(self):
        ds = DataStream(None, None)
        ds.datapoints = self.rip

        peaks, valleys = compute_peak_valley(rip = ds)

    def test_filter_intercept_outlier(self):
        # test cases
        up_intercepts_case_list = []
        down_intercepts_case_list = []
        up_intercepts_expected_case_list = []
        down_intercepts_expected_case_list = []

        # first case
        up_intercepts_case_list.append(form_data_point_from_start_time_array([10,20,30,40,50]))
        down_intercepts_case_list.append(form_data_point_from_start_time_array([9, 11, 21, 31, 41]))
        up_intercepts_expected_case_list.append([10,20,30,40,50])
        down_intercepts_expected_case_list.append([9,11,21,31,41])

        # second case
        up_intercepts_case_list.append(form_data_point_from_start_time_array([10,20,30,40,50]))
        down_intercepts_case_list.append(form_data_point_from_start_time_array([8, 9, 11, 21, 31, 41, 42]))
        up_intercepts_expected_case_list.append([10,20,30,40,50])
        down_intercepts_expected_case_list.append([9,11,21,31,42])

        # third case
        up_intercepts_case_list.append(form_data_point_from_start_time_array([10,20,22,23,30,32,33,40,42,43,50,52,53]))
        down_intercepts_case_list.append(form_data_point_from_start_time_array([9, 11, 21, 31, 41]))
        up_intercepts_expected_case_list.append([10, 20, 30, 40, 53])
        down_intercepts_expected_case_list.append([9,11,21,31,41])

        # fourth case
        up_intercepts_case_list.append(form_data_point_from_start_time_array([10, 20, 30, 40, 50]))
        down_intercepts_case_list.append(form_data_point_from_start_time_array([7,8,9,11,12,13,21,22,23,31,32,33,41,42,43,51,52,53]))
        up_intercepts_expected_case_list.append([10, 20, 30, 40, 50])
        down_intercepts_expected_case_list.append([9,13,23,33,43])

        # fifth case
        up_intercepts_case_list.append(form_data_point_from_start_time_array([10,11,12, 16,17,18, ]))
        down_intercepts_case_list.append(form_data_point_from_start_time_array([7,8,9,11,12,13,21,22,23,31,32,33,41,42,43,51,52,53]))
        up_intercepts_expected_case_list.append([10, 20, 30, 40, 50])
        down_intercepts_expected_case_list.append([9,13,23,33,43])

        for i,item in enumerate(up_intercepts_case_list):
            up_intercepts = up_intercepts_case_list[i]
            down_intercepts = down_intercepts_case_list[i]
            up_intercepts_output, down_intercepts_output = filter_intercept_outlier(up_intercepts, down_intercepts)

            # test all are List[Datapoints]
            self.assertIsInstance(up_intercepts_output, List[DataPoint])
            self.assertIsInstance(down_intercepts_output, List[DataPoint])

            # test output match for first case
            up_intercepts_output_start_time = [i.start_time for i in up_intercepts_output]
            self.assertTrue(np.array_equal(up_intercepts_output_start_time, up_intercepts_expected_case_list[i]))
            down_intercepts_output_start_time = [i.start_time for i in down_intercepts_output]
            self.assertTrue(np.array_equal(down_intercepts_output_start_time, down_intercepts_expected_case_list[i]))


def form_data_point_from_start_time_array(start_time_list):
    datapoints = []
    for i in start_time_list:
        datapoints.append(DataPoint.from_tuple(i, 0))

    return datapoints

if __name__ == '__main__':
    unittest.main()
