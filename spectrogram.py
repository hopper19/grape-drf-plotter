"""
Command: 
@author: Cuong Nguyen
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
import os
import sys
import argparse
from reader import Reader
import datetime
import solarContext
import pandas as pd

mpl.rcParams['font.size']       = 12
mpl.rcParams['font.weight']     = 'bold'
mpl.rcParams['axes.grid']       = True
mpl.rcParams['axes.titlesize']  = 30
mpl.rcParams['grid.linestyle']  = ':'
mpl.rcParams['figure.figsize']  = np.array([15, 8])
mpl.rcParams['axes.xmargin']    = 0
mpl.rcParams['legend.fontsize'] = 'xx-large'

class Plotter:
    def __init__(self, data_reader, output_dir="output"):
        self.data_reader = data_reader
        self.metadata = data_reader.get_metadata()
        self.fs = self.data_reader.resampled_fs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.event_fname = '{!s}_{!s}_grape2DRF'.format(self.metadata["utc_date"].date(),self.metadata["station"])

    def plot_spectrogram(self, channel_indices=None):
        # TODO: implement channel selection
        """Plot selected channels or all if not specified."""
        print(f"Now plotting {self.event_fname}...")
        if channel_indices is None:
            channel_indices = range(len(self.metadata["center_frequencies"]))
        else:
            channel_indices = [int(ch) for ch in sorted(channel_indices)]

        ncols = 1
        nrows = len(channel_indices)
        fig = plt.figure(figsize=(22,nrows*5))
        fig.suptitle(
            f"{self.metadata["station"]} ({self.metadata["city_state"]})\n" +
            f"Grape 2 Spectrogram for {self.metadata["utc_date"].date()}",
            size=42
        )

        for i in range(len(channel_indices)):
            cfreq_idx = channel_indices[::-1][i]
            ax_inx = i + 1
            print(f"Plotting {self.metadata["center_frequencies"][cfreq_idx]} MHz...")
            data = self.data_reader.read_data(channel_index=cfreq_idx)
            ax = fig.add_subplot(nrows, ncols, ax_inx)
            self._plot_ax(
                data,
                ax,
                freq=self.metadata["center_frequencies"][cfreq_idx],
                lastrow=ax_inx == len(channel_indices),
            )

        fig.tight_layout()
        png_fpath = os.path.join(self.output_dir, self.event_fname + ".png")
        fig.savefig(png_fpath, bbox_inches="tight")
        print(f"Plot saved to {png_fpath}")

    def _plot_ax(self, data, ax, freq, lastrow=False):
        """Plot data on the given axes."""

        ax.set_ylabel("{:.2f}MHz\nDoppler Shift".format(freq))

        f, t_spec, Sxx = signal.spectrogram(
            data, fs=self.fs, window="hann", nperseg=int(self.fs / 0.01)
        )
        Sxx_db = np.log10(Sxx) * 10
        f -= self.data_reader.target_bandwidth / 2
        ax.set_ylim(
            - self.data_reader.target_bandwidth / 2, self.data_reader.target_bandwidth / 2
        )
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            " ", ["black", "darkgreen", "green", "yellow", "red"]
        )
        cax = ax.pcolormesh(
            pd.date_range(
                start=self.metadata["utc_date"],
                end=self.metadata["utc_date"] + datetime.timedelta(days=1),
                periods=len(t_spec),
            ),
            f,
            Sxx_db,
            cmap=cmap,
        )

        sts = solarContext.solarTimeseries(
            self.metadata["utc_date"],
            self.metadata["utc_date"] + datetime.timedelta(days=1),
            self.metadata["lat"],
            self.metadata["lon"],
        )
        odct = {'color': 'white', 'lw': 4, 'alpha': 0.75}
        sts.overlaySolarElevation(ax, **odct)
        sts.overlayEclipse(ax, **odct)

        # Add colorbar
        # cbar = plt.colorbar(cax, ax=ax, orientation='vertical', pad=0.02)
        # cbar.set_label("Power (dB)")

        # Set dynamic x-tick positions for all plots (needed for grid)
        num_ticks = 13  # number of x-ticks
        xticks  = ax.get_xticks()
        positions = pd.date_range(start="2024-04-08", end="2024-04-09", periods=num_ticks)
        labels = [f"{(i * 24 / (num_ticks - 1)):0.0f}" for i in range(num_ticks)]
        # labels go from 00:00 to 01:00, ..., to o 00:00

        # # Set tick positions for ALL plots (required for grid lines)
        ax.set_xticks(xticks)

        # # Only show labels on the bottom plot
        if lastrow:
            labels = [mpl.dates.num2date(xtk).strftime("%H:%M") for xtk in xticks]
            ax.set_xticklabels(labels)
            ax.set_xlabel("UTC")
            # Keep tick marks visible
        else:
            ax.set_xticklabels([""] * len(xticks))  # Empty labels but keep ticks
            # Hide the actual tick marks without affecting grid
            ax.tick_params(axis='x', which='both', length=0)

        # Ensure grid is visible on both axes
        ax.grid(visible=True, which='both', axis='both')

def main():
    version = "2.0"
    
    parser = argparse.ArgumentParser(description="Grape2 Spectrogram Generator")
    parser.add_argument(
        "-i", "--input_dir", help="Path to the directory containing a ch0 subdirectory", required=True
    )
    parser.add_argument(
        "-o", "--output_dir", help="Output directory for plot", required=True
    )
    parser.add_argument(
        "-x", "--clean_cache", action="store_true", help="Clean up cache files after processing"
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

    data_reader = Reader(data_dir, cleanup_cache=args.clean_cache)
    plotter = Plotter(data_reader, output_dir=output_dir)

    # channels = sys.argv[2:] if len(sys.argv) > 2 else None
    plotter.plot_spectrogram()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Program failed SPECTACULARLY!!!")
        print(str(e))
