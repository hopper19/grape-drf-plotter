# plotter.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
import gc
import pandas as pd
import os


class Plotter:
    def __init__(self, metadata, output_dir="output"):
        self.fs = metadata["sampling_rate"]
        self.center_frequencies = metadata["center_frequencies"]
        self.station = metadata["station"]
        self.utc_date = metadata["utc_date"]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_data(self, data, start_index, end_index, rf_dict):
        event_fname = f"{self.utc_date}_{self.station}_grape2DRF_new"
        png_fname = event_fname + ".png"
        png_fpath = os.path.join(self.output_dir, png_fname)

        # Plot configurations
        ncols = 1
        nrows = len(self.center_frequencies)
        ax_inx = 0
        fig = plt.figure(figsize=(15, 4 * nrows))
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            " ", ["black", "darkgreen", "green", "yellow", "red"]
        )

        for cfreq_idx in range(nrows):
            ax_inx += 1
            ax = fig.add_subplot(nrows, ncols, ax_inx)
            self._plot_ax(data[cfreq_idx], ax, rf_dict, start_index, end_index, cmap)

        fig.tight_layout()
        fig.savefig(png_fpath, bbox_inches="tight")
        print(f"Plot saved to {png_fpath}")

    def _plot_ax(self, data_channel, ax, rf_dict, start_index, end_index, cmap):
        ylabel = "Doppler Shift (Hz)"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("UTC")

        # Generate spectrogram
        number = 2**20
        f, t_spec, Sxx = signal.spectrogram(
            data_channel, fs=self.fs, nperseg=number, noverlap=0, window="hann"
        )
        spectrum_timevec = pd.to_datetime(
            np.linspace(
                rf_dict["init_utc_timestamp"],
                rf_dict["init_utc_timestamp"] + (end_index - start_index + 1) / self.fs,
                len(t_spec),
            ),
            unit="s",
        )
        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)
        Sxx_db = Sxx
        np.log10(Sxx_db, where=(Sxx_db > 0), out=Sxx_db)
        Sxx_db *= 10

        # Plot spectrogram
        mpbl = ax.pcolormesh(spectrum_timevec, f, Sxx_db, cmap=cmap)

        xticks = ax.get_xticks()
        xtkls = [mpl.dates.num2date(xtk).strftime("%H:%M") for xtk in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtkls)


# main.py
import sys
from psws_data_reader import PSWSDataReader as DataReader


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <data_directory>")
        return

    data_dir = sys.argv[1]
    data_reader = DataReader(data_dir)
    data = data_reader.read_data()
    metadata = data_reader.get_metadata()

    plotter = Plotter(metadata)
    plotter.plot_data(
        data.values.T,
        data_reader.start_index,
        data_reader.end_index,
        data_reader.rf_dict,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Program failed SPECTACULARLY!!!")
        print(str(e))
