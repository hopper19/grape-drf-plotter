# plotter.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
import os
import sys
import argparse
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
        # plt.tight_layout()
        # plt.grid()

    def plot_spectrogram(self, channels=None):
        # TODO: implement channel selection 
        """Plot selected channels or all if not specified."""
        if channels is None:
            channels = range(len(self.center_frequencies))
        else:
            channels = [int(ch) for ch in channels]

        ncols = 1
        nrows = len(channels)
        fig = plt.figure(figsize=(10, 4 * nrows))
        # fig.subplots_adjust(top=0.93)
        fig.suptitle(
            f"Grape Narrow Spectrum, {self.utc_date},\n" +
            f"Lat. {self.metadata['lat']}, Long. {self.metadata['lon']} (Grid{self.metadata['grid']}) " +
            f"Station: {self.metadata['station']}", fontsize=16
        )
        # use classic style
        plt.style.use("classic")

        for ax_inx, cfreq_idx in enumerate(channels, start=1):
            data = self.data_reader.read_data(channel_index=cfreq_idx)
            print("Fisnihed reading data!")
            ax = fig.add_subplot(nrows, ncols, ax_inx)
            self._plot_ax(
                data,
                ax,
                freq=self.center_frequencies[cfreq_idx],
                lastrow=ax_inx == len(channels),
            )

        plt.xlabel("UTC")
        # num_ticks = 13  # number of x-ticks
        # positions = np.linspace(t_spec[0], t_spec[-1], num=num_ticks)
        # labels = [(i * 24 / (num_ticks - 1)) for i in range(num_ticks)] # TODO: hard coded for 24 hours
        # plt.xticks(labels)

        # Save the figure
        fig.tight_layout(rect=[0, 0, 1, 1])  # Leave space for the suptitle
        event_fname = f"{self.utc_date}_{self.station}_grape2DRF_new.png"
        png_fpath = os.path.join(self.output_dir, event_fname)
        fig.savefig(png_fpath, bbox_inches="tight")
        print(f"Plot saved to {png_fpath}")

    def _plot_ax(self, data, ax, freq, lastrow=False):
        """Plot data on the given axes."""

        ax.set_ylabel("{:.2f}MHz\nDoppler Shift (Hz)".format(freq))

        f, t_spec, Sxx = signal.spectrogram(
            data, fs=self.fs, window="hann", nperseg=int(self.fs / 0.01)
        )
        Sxx_db = np.log10(Sxx, where=(Sxx > 0)) * 10
        f -= self.data_reader.target_bandwidth / 2
        ax.set_ylim(
            - self.data_reader.target_bandwidth / 2, self.data_reader.target_bandwidth / 2
        )
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            " ", ["black", "darkgreen", "green", "yellow", "red"]
        )
        cax = ax.pcolormesh(t_spec, f, Sxx_db, cmap=cmap)

        # Add colorbar
        cbar = plt.colorbar(cax, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label("Power (dB)")

        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.grid(
            visible=True,
            which="both",
        )
        # Set dynamic x-tick labels
        num_ticks = 13 # number of x-ticks
        positions = np.linspace(t_spec[0], t_spec[-1], num=num_ticks)
        labels = [f"{(i * 24 / (num_ticks - 1)):0.0f}" for i in range(num_ticks)] # TODO: hard coded for 24 hours
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)



def main():
    version = "1.0.1"
    
    parser = argparse.ArgumentParser(description="Grape2 Spectrogram Generator")
    parser.add_argument(
        "-i", "--input_dir", help="Path to the directory containing a ch0 subdirectory", required=True
    )
    parser.add_argument(
        "-o", "--output_dir", help="Output directory for plot", required=True
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s v{version}",
        help="show program version",
    )
    
    args = parser.parse_args()
    
    data_dir = args.input_dir
    output_dir = args.output_dir

    data_reader = Reader(data_dir)
    plotter = Plotter(data_reader, output_dir=output_dir)

    # channels = sys.argv[2:] if len(sys.argv) > 2 else None
    plotter.plot_spectrogram()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Program failed SPECTACULARLY!!!")
        print(str(e))
