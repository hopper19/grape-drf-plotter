#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Digital RF s16 with 3 subchannels example
# Author: rvolz
# GNU Radio version: 3.10.10.0

from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import gr_digital_rf
import numpy as np; import gr_digital_rf




class drf_read_example(gr.top_block):

    def __init__(self, source_drf='0'):
        gr.top_block.__init__(self, "Digital RF s16 with 3 subchannels example", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.source_drf = source_drf

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 8000
        self.decimation = decimation = 800
        self.cfreq_2 = cfreq_2 = 15e6
        self.cfreq_1 = cfreq_1 = 10e6
        self.cfreq_0 = cfreq_0 = 5e6

        ##################################################
        # Blocks
        ##################################################

        self.low_pass_filter_0_1 = filter.fir_filter_ccf(
            decimation,
            firdes.low_pass(
                1,
                samp_rate,
                50,
                5,
                window.WIN_HAMMING,
                6.76))
        self.low_pass_filter_0_0 = filter.fir_filter_ccf(
            decimation,
            firdes.low_pass(
                1,
                samp_rate,
                50,
                5,
                window.WIN_HAMMING,
                6.76))
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            decimation,
            firdes.low_pass(
                1,
                samp_rate,
                50,
                5,
                window.WIN_HAMMING,
                6.76))
        self.hilbert_fc_0_1 = filter.hilbert_fc(512, window.WIN_HAMMING, 6.76)
        self.hilbert_fc_0_0 = filter.hilbert_fc(512, window.WIN_HAMMING, 6.76)
        self.hilbert_fc_0 = filter.hilbert_fc(512, window.WIN_HAMMING, 6.76)
        self.gr_digital_rf_digital_rf_source_0 = gr_digital_rf.digital_rf_source(
            source_drf,
            channels=[
                'ch0',
            ],
            start=[
                None,
            ],
            end=[
                None,
            ],
            repeat=False,
            throttle=False,
            gapless=True,
            min_chunksize=None,
        )
        self.gr_digital_rf_digital_rf_sink_0 = gr_digital_rf.digital_rf_sink(
            'output/grape2DRF/w2naf/OBS2024-04-08T00-00',
            channels=[
                'ch0',
            ],
            dtype=np.complex64,
            subdir_cadence_secs=3600,
            file_cadence_millisecs=180000,
            sample_rate_numerator=(int(samp_rate/decimation)),
            sample_rate_denominator=1,
            start='2024-04-08T00:00:00Z',
            ignore_tags=False,
            is_complex=True,
            num_subchannels=3,
            uuid_str=None,
            center_frequencies=(
                [5, 10, 15]
            ),
            metadata={'lat':41.33,'long':-75.67},
            is_continuous=True,
            compression_level=0,
            checksum=False,
            marching_periods=False,
            stop_on_skipped=False,
            stop_on_time_tag=False,
            debug=False,
            min_chunksize=None,
        )
        self.blocks_vector_to_streams_0_0 = blocks.vector_to_streams(gr.sizeof_float*1, 3)
        self.blocks_streams_to_vector_0 = blocks.streams_to_vector(gr.sizeof_gr_complex*1, 3)
        self.blocks_short_to_float_0 = blocks.short_to_float(3, ((2**15 - 1)))
        self.blocks_multiply_xx_0_1 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, (-1000), 1, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0_0, 1))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0_1, 1))
        self.connect((self.blocks_multiply_xx_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.low_pass_filter_0_0, 0))
        self.connect((self.blocks_multiply_xx_0_1, 0), (self.low_pass_filter_0_1, 0))
        self.connect((self.blocks_short_to_float_0, 0), (self.blocks_vector_to_streams_0_0, 0))
        self.connect((self.blocks_streams_to_vector_0, 0), (self.gr_digital_rf_digital_rf_sink_0, 0))
        self.connect((self.blocks_vector_to_streams_0_0, 0), (self.hilbert_fc_0, 0))
        self.connect((self.blocks_vector_to_streams_0_0, 1), (self.hilbert_fc_0_0, 0))
        self.connect((self.blocks_vector_to_streams_0_0, 2), (self.hilbert_fc_0_1, 0))
        self.connect((self.gr_digital_rf_digital_rf_source_0, 0), (self.blocks_short_to_float_0, 0))
        self.connect((self.hilbert_fc_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.hilbert_fc_0_0, 0), (self.blocks_multiply_xx_0_0, 0))
        self.connect((self.hilbert_fc_0_1, 0), (self.blocks_multiply_xx_0_1, 0))
        self.connect((self.low_pass_filter_0, 0), (self.blocks_streams_to_vector_0, 0))
        self.connect((self.low_pass_filter_0_0, 0), (self.blocks_streams_to_vector_0, 1))
        self.connect((self.low_pass_filter_0_1, 0), (self.blocks_streams_to_vector_0, 2))


    def get_source_drf(self):
        return self.source_drf

    def set_source_drf(self, source_drf):
        self.source_drf = source_drf

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 50, 5, window.WIN_HAMMING, 6.76))
        self.low_pass_filter_0_0.set_taps(firdes.low_pass(1, self.samp_rate, 50, 5, window.WIN_HAMMING, 6.76))
        self.low_pass_filter_0_1.set_taps(firdes.low_pass(1, self.samp_rate, 50, 5, window.WIN_HAMMING, 6.76))

    def get_decimation(self):
        return self.decimation

    def set_decimation(self, decimation):
        self.decimation = decimation

    def get_cfreq_2(self):
        return self.cfreq_2

    def set_cfreq_2(self, cfreq_2):
        self.cfreq_2 = cfreq_2

    def get_cfreq_1(self):
        return self.cfreq_1

    def set_cfreq_1(self, cfreq_1):
        self.cfreq_1 = cfreq_1

    def get_cfreq_0(self):
        return self.cfreq_0

    def set_cfreq_0(self, cfreq_0):
        self.cfreq_0 = cfreq_0



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--source-drf", dest="source_drf", type=str, default='0',
        help="Set Address of the 8kHz bandwidth source Digital RF directory [default=%(default)r]")
    return parser


def main(top_block_cls=drf_read_example, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(source_drf=options.source_drf)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
