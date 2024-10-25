# plotter.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
import os
import sys
from reader import Reader

# mpl.rcParams["font.size"] = 12
# mpl.rcParams["font.weight"] = "bold"
# mpl.rcParams["axes.grid"] = True
# mpl.rcParams["grid.linestyle"] = ":"
# mpl.rcParams["figure.figsize"] = np.array([15, 8])
# mpl.rcParams["axes.xmargin"] = 0

class Plotter:
    def __init__(self, data_reader, output_dir="output"):
        self.data_reader = data_reader
        self.metadata = data_reader.get_metadata()
        self.fs = self.data_reader.resampled_fs
        self.start_index, self.end_index = self.data_reader.start_index, self.data_reader.end_index
        self.center_frequencies = self.metadata["center_frequencies"]
        self.station = self.metadata["station"]
        self.utc_date = self.metadata["utc_date"]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cmap = mpl.colors.LinearSegmentedColormap.from_list(
            " ", ["black", "darkgreen", "green", "yellow", "red"]
        )
        plt.style.use("classic")
        plt.tight_layout()
        plt.grid()

    def plot_spectrogram(self, channels=None):
        """Plot selected channels or all if not specified."""
        if channels is None:
            channels = range(len(self.center_frequencies))
        else:
            channels = [int(ch) for ch in channels]

        ncols = 1
        nrows = len(channels)
        fig = plt.figure(figsize=(10, 2 * nrows))

        for ax_inx, cfreq_idx in enumerate(channels, start=1):
            data = self.data_reader.read_data(channel_index=cfreq_idx)
            print("Fisnihed reading data!")
            ax = fig.add_subplot(nrows, ncols, ax_inx)
            self._plot_ax(
                data,
                ax
            )

        plt.xlabel("UTC")
        # num_ticks = 13  # number of x-ticks
        # positions = np.linspace(t_spec[0], t_spec[-1], num=num_ticks)
        # labels = [(i * 24 / (num_ticks - 1)) for i in range(num_ticks)] # TODO: hard coded for 24 hours
        # plt.xticks(labels)

        # Save the figure
        event_fname = f"{self.utc_date}_{self.station}_grape2DRF_new.png"
        png_fpath = os.path.join(self.output_dir, event_fname)
        fig.savefig(png_fpath, bbox_inches="tight")
        print(f"Plot saved to {png_fpath}")

    def _plot_ax(self, data, ax):
        """Plot data on the given axes."""

        ax.set_ylabel("Doppler Shift (Hz)")

        f, t_spec, Sxx = signal.spectrogram(
            data, fs=self.fs, window="hann", nperseg=int(self.fs / 0.01)
        )
        Sxx_db = np.log10(Sxx, where=(Sxx > 0)) * 10
        f -= self.data_reader.target_bandwidth / 2
        ax.set_ylim(
            - self.data_reader.target_bandwidth / 2, self.data_reader.target_bandwidth / 2
        )
        ax.pcolormesh(t_spec, f, Sxx_db, cmap=self.cmap)
        ax.set_xticks([])

        # Set dynamic x-tick labels
        # num_ticks = 13 # number of x-ticks
        # positions = np.linspace(t_spec[0], t_spec[-1], num=num_ticks)
        # labels = [f"{(i * 24 / (num_ticks - 1)):.1f}h" for i in range(num_ticks)] # TODO: hard coded for 24 hours
        # ax.set_xticks(positions)
        # ax.set_xticklabels(labels)


def main():
    data_dir = "/home/cuong/drive/GRAPE2-SFTP/w2naf"
    output_dir = 'output'

    data_reader = Reader(data_dir)
    plotter = Plotter(data_reader, output_dir=output_dir)

    channels = sys.argv[2:] if len(sys.argv) > 2 else None
    plotter.plot_spectrogram(channels)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Program failed SPECTACULARLY!!!")
        print(str(e))
