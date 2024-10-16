import os
import sys
import digital_rf as drf
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
import datetime
import pytz
import pickle
from scipy import signal


class PSWSDataReader:
    def __init__(self, datadir, cachedir=None, resampled_fs=10, batch_size_mins=30):
        self.datadir = datadir
        self.cachedir = cachedir
        if self.cachedir and not os.path.exists(self.cachedir):
            os.makedirs(self.cachedir)
        self.batch_size_mins = batch_size_mins

        self.dro, self.dmr = self._get_readers()
        self.fs = int(self.dmr.get_samples_per_second())
        self.resampled_fs = resampled_fs
        self.start_index, self.end_index = self.dro.get_bounds("ch0")
        self.rf_dict = self.dro.get_properties("ch0", sample=self.start_index)
        self.utc_date = datetime.datetime.fromtimestamp(
            self.rf_dict["init_utc_timestamp"], tz=pytz.utc
        ).date()

        # Retrieve metadata
        latest_meta = self.dmr.read_latest()
        latest_inx = list(latest_meta.keys())[0]
        self.center_frequencies = latest_meta[latest_inx]["center_frequencies"]
        self.station = latest_meta[latest_inx]["callsign"]
        self.node = latest_meta[latest_inx]["station_node_number"]

    def _get_readers(self):
        metadir = os.path.join(self.datadir, "ch0", "metadata")
        dro = drf.DigitalRFReader(self.datadir)
        dmr = drf.DigitalMetadataReader(metadir)
        return dro, dmr

    def read_data(self, channel: int):
        print(f"Reading DRF data for channel {channel}... (batch size = {self.batch_size_mins} mins)")
        if self.cachedir:
            ba_fpath = os.path.join(
                self.cachedir,
                self.utc_date.strftime("%Y-%m-%d")
                + "_"
                + self.node
                + "_RAWDATA_"
                + str(channel)
                + ".ba.pkl",
            )
            if not os.path.exists(ba_fpath):
                cont_data_arr = self.dro.get_continuous_blocks(
                    self.start_index, self.end_index, "ch0"
                )
                batch_size_samples = self.fs * 60 * self.batch_size_mins
                read_iters = math.ceil((self.end_index - self.start_index) / batch_size_samples)

                batches = []
                start_sample = list(cont_data_arr.keys())[0]
                for _ in tqdm(range(read_iters)):
                    batch = self.dro.read_vector(start_sample, batch_size_samples, "ch0", channel)
                    batches.append(batch)
                    start_sample += batch_size_samples            
                result = np.concatenate(batches)

                if not os.path.exists(ba_fpath):
                    with open(ba_fpath,'wb') as fl:
                        pickle.dump(result,fl)
                        print("Cached data!")
            else:
                print('Using cached file {!s}...'.format(ba_fpath))
                with open(ba_fpath,'rb') as fl:
                    result = pickle.load(fl)
                # print(result.shape)
                # print("decimating")
                decimation_factor = int(self.fs / self.resampled_fs)
                resampled_signal = signal.decimate(
                    result, decimation_factor, ftype="fir", zero_phase=True
                )
                resampled_cache_path = os.path.join(
                    self.cachedir,
                    self.utc_date.strftime("%Y-%m-%d")
                    + "_"
                    + self.node
                    + "_RAWDATA_"
                    + str(self.resampled_fs)
                    + "Hz_"
                    + str(channel)
                    + ".ba.pkl",
                )
                if not os.path.exists(resampled_cache_path):
                    with open(resampled_cache_path, "wb") as fl:
                        pickle.dump(resampled_signal, fl)
                print(resampled_signal.shape)

        else:
            cont_data_arr = self.dro.get_continuous_blocks(
                self.start_index, self.end_index, "ch0"
            )
            batch_size_samples = self.fs * 60 * self.batch_size_mins
            read_iters = math.ceil((self.end_index - self.start_index) / batch_size_samples)

            batches = []
            start_sample = list(cont_data_arr.keys())[0]
            for _ in tqdm(range(read_iters)):
                batch = self.dro.read_vector(start_sample, batch_size_samples, "ch0", channel)
                batches.append(batch)
                start_sample += batch_size_samples            
            result = np.concatenate(batches)

        print(f"Loaded data for channel {channel} successfully!")
        # if self.cachedir:
        #     ba_fpath = os.path.join(
        #         self.cachedir,
        #         self.utc_date.strftime("%Y-%m-%d")
        #         + "_"
        #         + self.node
        #         + "_RAWDATA_"
        #         + str(channel)
        #         + ".ba.pkl",
        #     )
        #     if not os.path.exists(ba_fpath):
        #         with open(ba_fpath,'wb') as fl:
        #             pickle.dump(result,fl)
        return result

    def get_metadata(self):
        return {
            "sampling_rate": self.fs,
            "center_frequencies": self.center_frequencies,
            "station": self.station,
            "utc_date": self.utc_date,
        }
