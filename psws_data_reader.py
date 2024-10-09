# data_reader.py
import os
import digital_rf as drf
import pandas as pd
import math
from tqdm import tqdm
import datetime
import pytz

class PSWSDataReader:
    def __init__(self, datadir, channel="ch0", batch_size_mins=30):
        self.datadir = datadir
        self.batch_size_mins = batch_size_mins
        self.channel = channel

        self.dro, self.dmr = self._get_readers()
        self.fs = int(self.dmr.get_samples_per_second())
        self.start_index, self.end_index = self.dro.get_bounds(self.channel)
        self.rf_dict = self.dro.get_properties(self.channel, sample=self.start_index)
        self.utc_date = datetime.datetime.fromtimestamp(
            self.rf_dict["init_utc_timestamp"], tz=pytz.utc
        ).date()

        # Retrieve metadata
        latest_meta = self.dmr.read_latest()
        latest_inx = list(latest_meta.keys())[0]
        self.center_frequencies = latest_meta[latest_inx]["center_frequencies"]
        self.station = latest_meta[latest_inx]["callsign"]

    def _get_readers(self):
        metadir = os.path.join(self.datadir, self.channel, "metadata")
        dro = drf.DigitalRFReader(self.datadir)
        dmr = drf.DigitalMetadataReader(metadir)
        return dro, dmr

    def read_data(self):
        print(f"Reading DRF data for channel {self.channel}... (batch size = {self.batch_size_mins} mins)")
        cont_data_arr = self.dro.get_continuous_blocks(
            self.start_index, self.end_index, self.channel
        )
        batch_size_samples = self.fs * 60 * self.batch_size_mins
        read_iters = math.ceil((self.end_index - self.start_index) / batch_size_samples)

        result = pd.DataFrame()
        start_sample = list(cont_data_arr.keys())[0]
        for _ in tqdm(range(read_iters)):
            batch = self.dro.read_vector(start_sample, batch_size_samples, self.channel)
            result = pd.concat([result, pd.DataFrame(batch)])
            start_sample += batch_size_samples

        print("Loaded data successfully!")
        return result

    def get_channel_data(self, channel=None):
        """Get data from a specific channel."""
        if channel:
            self.channel = channel
            self.start_index, self.end_index = self.dro.get_bounds(self.channel)
        return self.read_data()

    def get_metadata(self):
        """Retrieve metadata for the current channel."""
        return {
            "sampling_rate": self.fs,
            "center_frequencies": self.center_frequencies,
            "station": self.station,
            "utc_date": self.utc_date,
            "channel": self.channel,
        }
