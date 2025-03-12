"""
Grape 2 Spectrogram Generator

Generates spectrograms from Digital RF files created by Grape 2 receivers.

@author: Cuong Nguyen
"""
# Standard library imports
import os
import sys
import datetime
import argparse

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal

# Local imports
from grape2spectrogram.reader import Reader
from grape2spectrogram import solarContext

class Plotter:
    """
    Creates spectrograms from Grape 2 Digital RF data.
    
    Handles the visualization of frequency data across time, with solar
    context information overlaid.
    """

    # Default matplotlib configuration
    PLOT_CONFIG = {
        'font.size': 12,
        'font.weight': 'bold',
        'axes.grid': True,
        'axes.titlesize': 30,
        'grid.linestyle': ':',
        'figure.figsize': np.array([15, 8]),
        'axes.xmargin': 0,
        'legend.fontsize': 'xx-large',
    }
    
    # Solar context overlay configuration
    SOLAR_OVERLAY_CONFIG = {
        'color': 'white', 
        'lw': 4, 
        'alpha': 0.75
    }

    # HARC_PLOT configuration
    HARC_PLOT_CONFIG = {
        'figure.titlesize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'axes.labelsize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large',
        'legend.fontsize': 'large',
        'figure.titleweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    }

    def __init__(self, data_reader, output_dir="output"):
        """
        Initialize the plotter with a data reader and output directory.
        
        Args:
            data_reader: Reader object for accessing Digital RF data
            output_dir: Directory where plots will be saved
        """
        # Apply matplotlib styling
        for key, value in self.PLOT_CONFIG.items():
            mpl.rcParams[key] = value
        for key,value in self.HARC_PLOT_CONFIG.items():
            mpl.rcParams[key] = value
            
        self.data_reader = data_reader
        self.metadata = data_reader.get_metadata()
        self.fs = self.data_reader.resampled_fs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create filename for plots based on date and station
        self.event_fname = f"{self.metadata['utc_date'].date()}_{self.metadata['station']}_grape2DRF"

    def plot_spectrogram(self, channel_indices=None):
        """
        Plot spectrograms for selected channels or all if not specified.
        
        Args:
            channel_indices: List of channel indices to plot. If None, plot all channels.
        """
        print(f"Now plotting {self.event_fname}...")
        
        # Handle channel selection
        if channel_indices is None:
            channel_indices = range(len(self.metadata["center_frequencies"]))
        else:
            channel_indices = [int(ch) for ch in sorted(channel_indices)]
        
        # Create figure with appropriate size
        nrows = len(channel_indices)
        fig = plt.figure(figsize=(22, nrows * 5))
        
        # Add title
        station = self.metadata["station"]
        location = self.metadata["city_state"]
        date = self.metadata["utc_date"].date()
        fig.suptitle(
            f"{station} ({location})\nGrape 2 Spectrogram for {date}",
            size=42
        )

        # Plot each channel
        for i, idx in enumerate(range(len(channel_indices))):
            cfreq_idx = channel_indices[::-1][idx]
            plot_position = idx + 1
            
            print(f"Plotting {self.metadata['center_frequencies'][cfreq_idx]} MHz...")
            
            # Get data and create subplot
            data = self.data_reader.read_data(channel_index=cfreq_idx)
            ax = fig.add_subplot(nrows, 1, plot_position)
            
            # Plot the data
            self._plot_ax(
                data,
                ax,
                freq=self.metadata["center_frequencies"][cfreq_idx],
                lastrow=(plot_position == len(channel_indices)),
            )

        # Save the figure
        fig.tight_layout()
        png_fpath = os.path.join(self.output_dir, f"{self.event_fname}.png")
        fig.savefig(png_fpath, bbox_inches="tight")
        print(f"Plot saved to {png_fpath}")

    def _plot_ax(self, data, ax, freq, lastrow=False):
        """
        Plot spectrogram data on the given axes.
        
        Args:
            data: The signal data to plot
            ax: The matplotlib axis to plot on
            freq: Center frequency in MHz
            lastrow: Whether this is the bottom plot (for x-axis labels)
        """
        # Set y-axis label
        ax.set_ylabel(f"{freq:.2f}MHz\nDoppler Shift")

        # Generate spectrogram
        nperseg = int(self.fs / 0.01)  # 10ms segments
        f, t_spec, Sxx = signal.spectrogram(data, fs=self.fs, window="hann", nperseg=nperseg)
        
        # Convert to dB scale
        Sxx_db = np.log10(Sxx) * 10
        
        # Center frequencies around zero
        f -= self.data_reader.target_bandwidth / 2
        
        # Set y-axis limits to match bandwidth
        bandwidth = self.data_reader.target_bandwidth
        ax.set_ylim(-bandwidth/2, bandwidth/2)
        
        # Create custom colormap
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "grape_cmap", ["black", "darkgreen", "green", "yellow", "red"]
        )
        
        # Create time axis from UTC date
        time_range = pd.date_range(
            start=self.metadata["utc_date"],
            end=self.metadata["utc_date"] + datetime.timedelta(days=1),
            periods=len(t_spec),
        )
        
        # Plot spectrogram
        cax = ax.pcolormesh(time_range, f, Sxx_db, cmap=cmap)

        # Add solar context
        sts = solarContext.solarTimeseries(
            self.metadata["utc_date"],
            self.metadata["utc_date"] + datetime.timedelta(days=1),
            self.metadata["lat"],
            self.metadata["lon"],
        )
        
        # Overlay solar elevation and eclipse information
        sts.overlaySolarElevation(ax, **self.SOLAR_OVERLAY_CONFIG)
        sts.overlayEclipse(ax, **self.SOLAR_OVERLAY_CONFIG)

        # Get x-ticks for consistent grid across all subplots
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)

        # Only show x-axis labels on the bottom plot
        if lastrow:
            labels = [mpl.dates.num2date(xtk).strftime("%H:%M") for xtk in xticks]
            ax.set_xticklabels(labels)
            ax.set_xlabel("UTC")
        else:
            ax.set_xticklabels([""] * len(xticks))
            ax.tick_params(axis='x', which='both', length=0)  # Hide tick marks but keep grid

        # Ensure grid is visible
        ax.grid(visible=True, which='both', axis='both')

def main():    
    parser = argparse.ArgumentParser(description="Grape2 Spectrogram Generator")
    parser.add_argument(
        "-i", "--input_dir", 
        help="Path to the directory containing a ch0 subdirectory", 
        required=True
    )
    parser.add_argument(
        "-o", "--output_dir", 
        help="Output directory for plot", 
        required=True
    )
    parser.add_argument(
        "-k", "--keep_cache", 
        action="store_true", 
        help="Keep cache files after processing (by default, cache is removed)"
    )
    parser.add_argument(
        "-c", "--channels",
        nargs="*",
        help="Specific channel indices to plot (e.g., 0 1 2)"
    )    
    args = parser.parse_args()
    try:
        # Initialize reader and plotter
        data_reader = Reader(args.input_dir, cleanup_cache=not args.keep_cache)
        plotter = Plotter(data_reader, output_dir=args.output_dir)
        
        # Plot with specified channels or all channels
        plotter.plot_spectrogram(channel_indices=args.channels)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
