# Plot Dopplergram from 8kHz bandwidth DRF data
# Author: Cuong Nguyen KC3UAX

# IN ORDER OF IMPORTANCE
# TODO: test with gapped data
# TODO: optimize memory usage. Method: read one channel each, plot that ax, and repeat,
#                                       instead of reading all three at once
# TODO: ability to select which center frequency to plot
# TODO: determine where in the data pipeline to apply zero cal (on the Pi during conversion to DRF, on the server during plotting)

import os
import pytz
import digital_rf as drf
import sys
import math
import pickle
import datetime
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib as mpl
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

def get_readers():
    args = sys.argv[1:]
    datadir = args[0]
    metadir = os.path.join(datadir, 'ch0', "metadata")
    dro = drf.DigitalRFReader(datadir)
    dmr = drf.DigitalMetadataReader(metadir)
    return dro, dmr

def main():
    # POSSIBLE OBJECT PARAMS

    # how many mins of data to read per iterations
    # higher is more efficient
    # do NOT increase past 60
    batch_size_mins = 30

    # subdir to store the plots
    output_subdir = 'grape2DRF'

    dro, dmr = get_readers()
    start_index, end_index = dro.get_bounds("ch0")

    rf_dict = dro.get_properties('ch0', sample=start_index)
    print(rf_dict)
    utc_date = datetime.datetime.fromtimestamp(
        rf_dict["init_utc_timestamp"], tz=pytz.utc
    ).date()
    print('Data Date:', utc_date)

    latest_meta = dmr.read_latest()
    latest_inx = list(latest_meta.keys())[0]
    center_frequencies = latest_meta[latest_inx]["center_frequencies"]
    station = latest_meta[latest_inx]["callsign"]
    fs = int(dmr.get_samples_per_second())
    print("Sampling rate:", fs)
    print("Center frequencies:", center_frequencies)
    print("Station:", station)

    event_fname = f'{utc_date}_{station}_{output_subdir}'
    png_fname = event_fname + '.png'
    output_dir = os.path.join('output', output_subdir)
    png_fpath = os.path.join(output_dir, png_fname)
    ba_fpath = os.path.join(output_dir, event_fname + '.ba.pkl')

    if not os.path.exists(ba_fpath):
        cont_data_arr = dro.get_continuous_blocks(start_index, end_index, 'ch0')
        # print(f'Number of continuous blocks: {len(cont_data_arr)}')

        batch_size_samples = fs*60*batch_size_mins   # batch size in number of samples
        read_iters = math.ceil((end_index - start_index) / batch_size_samples)

        print(f'Reading DRF data... (batch size = {batch_size_mins} mins)')    
        result = pd.DataFrame()
        start_sample = list(cont_data_arr.keys())[0]
        for i in tqdm(range(read_iters)):
            batch = dro.read_vector(start_sample, batch_size_samples, "ch0")
            result = pd.concat([result, pd.DataFrame(batch)])
            start_sample += batch_size_samples

        print("Loaded data successfully!")
        with open(ba_fpath,'wb') as fl:
            pickle.dump(result,fl)
    else:
        print('Using cached file {!s}...'.format(ba_fpath))
        with open(ba_fpath,'rb') as fl:
            result = pickle.load(fl)        

    print(f'Now plotting {event_fname}...')

    ncols       = 1
    nrows       = len(center_frequencies)
    ax_inx      = 0
    fig = plt.figure(figsize=(15, 4 * nrows))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        " ", ["black", "darkgreen", "green", "yellow", "red"]
    )
    def plot_ax(cfreq_idx, ax):
        ylabel = []
        ylabel.append("Doppler Shift (Hz)")
        ax.set_ylabel("\n".join(ylabel))
        ax.set_xlabel("UTC")

        f, t_spec, Sxx = signal.spectrogram(
            result[cfreq_idx], fs=fs, nfft=1024, window="hann"
        )
        print(len(t_spec))
        # ts_vec = np.linspace(rf_dict["init_utc_timestamp"], rf_dict['init_utc_timestamp']+end_index-start_index, len(t_spec))
        # spectrum_timevec = [datetime.datetime.fromtimestamp(x) for x in ts_vec]

        # print(t_spec[0], t_spec[-1])
        # print(
        #     rf_dict["init_utc_timestamp"],
        #     rf_dict["init_utc_timestamp"] + (end_index - start_index + 1) / fs,
        # )
        spectrum_timevec = pd.to_datetime(
            np.linspace(
                rf_dict["init_utc_timestamp"],
                rf_dict["init_utc_timestamp"] + (end_index - start_index + 1) / fs,
                len(t_spec),
            ),
            unit='s'
        )
        # print(spectrum_timevec[0], spectrum_timevec[-1])
        # spectrum_timevec = [
        #     datetime.datetime.fromtimestamp(x)
        #     for x in np.linspace(
        #         rf_dict["init_utc_timestamp"],
        #         rf_dict["init_utc_timestamp"] + (end_index-start_index+1)/fs,
        #         len(t_spec),
        #     )
        # ]
        # print(spectrum_timevec[0], spectrum_timevec[-1])

        # f = (np.fft.fftshift(f)).astype("float64")  # Frequency needs to be in float64 for some reason...
        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)
        print('ghot here')
        Sxx_db = 10 * np.log10(Sxx)
        del Sxx, t_spec
        mpbl = ax.pcolormesh(spectrum_timevec, f, Sxx_db, cmap=cmap)
        # cbar = fig.colorbar(mpbl, label="PSD [dB]")

        xticks = ax.get_xticks()
        xtkls = []
        for xtk in xticks:
            dt = mpl.dates.num2date(xtk)
            xtkl = dt.strftime("%H:%M")
            xtkls.append(xtkl)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtkls)

    for cfreq_idx, cfreq in enumerate(center_frequencies):
        print('   {!s} MHz...'.format(cfreq))
        ax_inx      += 1
        ax          = fig.add_subplot(nrows,ncols,ax_inx)
        plot_ax(cfreq_idx,ax)
    fig.tight_layout()
    fig.savefig(png_fpath, bbox_inches="tight")
    print(png_fpath)
    # radio_index = 0
    # f, t_spec, Sxx = signal.spectrogram(result[radio_index],fs=fs,window='hann',nperseg=int(fs/0.01))
    # Sxx_db = 10*np.log10(Sxx)

    # flim = (990,1010)
    # fig = plt.figure(figsize=(20,6))
    # ax  = fig.add_subplot(1,1,1)
    # mpbl = ax.pcolormesh(t_spec,f,Sxx_db)
    # cbar = fig.colorbar(mpbl,label='PSD [dB]')
    # # ax.set_title(dir_path+"_R"+str(radio_index+1))
    # ax.set_xlabel('t [s]')
    # ax.set_ylabel('Frequency [Hz]')
    # ax.set_ylim(flim)
    # fig.savefig('plots/grape_spectrogram.png',bbox_inches='tight')

# def plot_figure(df):
#     print('Now plotting...')
#     cfreqs = [5, 10, 15]

#     ncols       = 1
#     nrows       = len(cfreqs)

#     ax_inx      = 0
#     fig         = plt.figure(figsize=(15,4*nrows))
#     for cfreq_idx, cfreq in enumerate(cfreqs):
#         print('   {!s} MHz...'.format(cfreq))
#         ax_inx      += 1
#         ax          = fig.add_subplot(nrows,ncols,ax_inx)
#         plot_ax(df, cfreq_idx, cfreq,ax)

# def plot_ax(result, cfreq_idx, cfreq, ax, cmap=None,plot_colorbar=False,xlim=None):
#     ylabel  = []
# #        ylabel.append('{!s} MHz'.format(cfreq))
#     ylabel.append('Doppler Shift (Hz)')
#     ax.set_ylabel('\n'.join(ylabel))
#     ax.set_xlabel('UTC')

#     f, t_spec, Sxx  = signal.spectrogram(result[cfreq_idx],fs=8000,nfft=1024,window='hann',return_onesided=False)


# def plot_ax(cfreq,ax,cmap=None,plot_colorbar=False,xlim=None):

#     sDate   = self.sDate
#     eDate   = self.eDate

#     ylabel  = []
# #        ylabel.append('{!s} MHz'.format(cfreq))
#     ylabel.append('Doppler Shift (Hz)')
#     ax.set_ylabel('\n'.join(ylabel))
#     ax.set_xlabel('UTC')

#     result  = self.result
#     props   = result['properties']
#     bigarray = self.result['bigarray_dct'].get(cfreq)
#     if bigarray is None:
#         msg = 'ERROR: No data for {!s} MHz'.format(cfreq)
#         ax.text(0.5,0.5,msg,ha='center',va='center',transform=ax.transAxes)
#         print(msg)
#         return

#     f, t_spec, Sxx  = signal.spectrogram(bigarray,fs=self.fs,nfft=1024,window='hann',return_onesided=False)
#     if self.spectrum_timevec is None:
#         # TODO: Make this more clean in the future.
#         # ts0                     = min(result['timevec_utc']).timestamp()
#         # ts1                     = max(result['timevec_utc']).timestamp()
#         ts0                     = result['tmin']
#         ts1                     = result['tmax']
#         print('ts0 ts1',ts0, ts1)
#         ts_vec                  = np.linspace(ts0,ts1,len(t_spec))
#         self.spectrum_timevec   = [datetime.datetime.utcfromtimestamp(x) for x in ts_vec]

#     f               = (np.fft.fftshift(f)).astype('float64') # Frequency needs to be in float64 for some reason...
#     Sxx             = np.fft.fftshift(Sxx,axes=0)
#     Sxx_db          = 10*np.log10(Sxx)
#     if cmap is None:
#         cmap = self.cmap
#     mpbl            = ax.pcolormesh(self.spectrum_timevec,f,Sxx_db,cmap=cmap)

#     if plot_colorbar:
#         cbar = fig.colorbar(mpbl,label='PSD [dB]')

#     # sts     = solarContext.solarTimeseries(sDate,eDate,solar_lat,solar_lon)
#     # odct    = {'color':'white','lw':4,'alpha':0.75}
#     # if overlaySolarElevation:
#     #     sts.overlaySolarElevation(ax,**odct)

#     # if overlayEclipse:
#     #     sts.overlayEclipse(ax,**odct)

#     if xlim is None:
#         xlim = (sDate,eDate)

#     # TODO: hardcode for now only
#     # ax.set_ylim(-5, 5)

#     xticks  = ax.get_xticks()
#     xtkls   = []
#     for xtk in xticks:
#         dt      = mpl.dates.num2date(xtk)
#         xtkl    = dt.strftime('%H:%M')
#         xtkls.append(xtkl)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xtkls)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Program failed SPECTACULARLY!!!')
        print(str(e))
