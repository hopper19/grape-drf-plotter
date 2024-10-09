# data_reader.py
import os
import digital_rf as drf
import pandas as pd
import pickle
import math
from tqdm import tqdm
import datetime
import pytz


class PSWSDataReader:
    def __init__(self, datadir, batch_size_mins=30, cache_dir="output"):
        self.datadir = datadir
        self.batch_size_mins = batch_size_mins
        self.cache_dir = cache_dir

        self.dro, self.dmr = self._get_readers()
        self.fs = int(self.dmr.get_samples_per_second())
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

    def _get_readers(self):
        metadir = os.path.join(self.datadir, "ch0", "metadata")
        dro = drf.DigitalRFReader(self.datadir)
        dmr = drf.DigitalMetadataReader(metadir)
        return dro, dmr

    def read_data(self):
        # Check if cached file exists
        ba_fpath = os.path.join(
            self.cache_dir, f"{self.utc_date}_{self.station}_grape2DRF.ba.pkl"
        )
        if os.path.exists(ba_fpath):
            print(f"Using cached file {ba_fpath}...")
            with open(ba_fpath, "rb") as fl:
                return pickle.load(fl)

        print(f"Reading DRF data... (batch size = {self.batch_size_mins} mins)")
        cont_data_arr = self.dro.get_continuous_blocks(
            self.start_index, self.end_index, "ch0"
        )
        batch_size_samples = self.fs * 60 * self.batch_size_mins
        read_iters = math.ceil((self.end_index - self.start_index) / batch_size_samples)

        result = pd.DataFrame()
        start_sample = list(cont_data_arr.keys())[0]
        for _ in tqdm(range(read_iters)):
            batch = self.dro.read_vector(start_sample, batch_size_samples, "ch0")
            result = pd.concat([result, pd.DataFrame(batch)])
            start_sample += batch_size_samples

        print("Loaded data successfully!")
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(ba_fpath, "wb") as fl:
            pickle.dump(result, fl)
        return result
    
    def get_channel(index):
     # TODO: 
        pass

    def get_metadata(self):
        return {
            "sampling_rate": self.fs,
            "center_frequencies": self.center_frequencies,
            "station": self.station,
            "utc_date": self.utc_date,
        }
