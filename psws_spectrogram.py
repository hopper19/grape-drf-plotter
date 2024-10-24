# plotter.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
import gc
import pandas as pd
import os
import sys
from psws_data_reader import PSWSDataReader as DataReader


class Plotter:
    def __init__(self, data_reader, output_dir="output"):
        self.data_reader = data_reader
        self.metadata = data_reader.get_metadata()
        self.fs = self.data_reader.resampled_fs
        self.center_frequencies = self.metadata["center_frequencies"]
        self.station = self.metadata["station"]
        self.utc_date = self.metadata["utc_date"]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_spectrogram(self):
        event_fname = f"{self.utc_date}_{self.station}_grape2DRF_new"
        png_fname = event_fname + ".png"
        png_fpath = os.path.join(self.output_dir, png_fname)

        # Plot configurations
        ncols = 1
        nrows = len(self.center_frequencies)
        ax_inx = 0
        fig = plt.figure(figsize=(10, 5 * nrows))
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            " ", ["black", "darkgreen", "green", "yellow", "red"]
        )

        for cfreq_idx in range(nrows):
            data = self.data_reader.read_data(channel=cfreq_idx)
            ax_inx += 1
            ax = fig.add_subplot(nrows, ncols, ax_inx)
            self._plot_ax(
                data,
                ax,
                self.data_reader.rf_dict,
                self.data_reader.start_index,
                self.data_reader.end_index,
                cmap,
            )
        
        # plt.grid()
        fig.tight_layout()
        fig.savefig(png_fpath, bbox_inches="tight")
        print(f"Plot saved to {png_fpath}")

    def _plot_ax(self, data_channel, ax, rf_dict, start_index, end_index, cmap):
        ylabel = "Doppler Shift (Hz)"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("UTC")
        # ax.set_ylim(995, 1005)

        # Generate spectrogram
        number = 2**14
        f, t_spec, Sxx = signal.spectrogram(
            data_channel, fs=self.fs, nfft=1024, window="hann"
        )
        # spectrum_timevec = pd.to_datetime(
        #     np.linspace(
        #         rf_dict["init_utc_timestamp"],
        #         rf_dict["init_utc_timestamp"] + (end_index - start_index + 1) / self.fs,
        #         len(t_spec),
        #     ),
        #     unit="s",
        # )
        # print(f)
        f = np.fft.fftshift(f)
        # print(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)
        Sxx_db = Sxx
        np.log10(Sxx_db, where=(Sxx_db > 0), out=Sxx_db)
        Sxx_db *= 10

        # Plot spectrogram
        mpbl = ax.pcolormesh(np.arange(len(t_spec)), f, Sxx_db, cmap=cmap)

        # xticks = ax.get_xticks()
        # xtkls = [mpl.dates.num2date(xtk).strftime("%H:%M") for xtk in xticks]
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xtkls)
        t_spec_len = len(t_spec)

        # Generate the x-tick positions and labels dynamically
        positions = np.linspace(0, t_spec_len, num=13)
        labels = [f"{hour:02d}" for hour in range(0, 25, 2)]

        # Apply x-ticks with positions and labels
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)


def main():
    # if len(sys.argv) < 4:
    #     print("Usage: python psws_spectrogram.py <data_directory> <output_directory> <cache_directory")
    #     return

    # data_dir = sys.argv[1]
    # cache_dir = sys.argv[3]
    data_dir = sys.argv[1]
    cache_dir = 'cache'
    data_reader = DataReader(data_dir, cachedir=cache_dir, resampled_fs=4000)

    # output_dir = sys.argv[2]
    output_dir = 'output/grape2DRF'
    plotter = Plotter(data_reader, output_dir=output_dir)
    plotter.plot_spectrogram()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Program failed SPECTACULARLY!!!")
        print(str(e))
