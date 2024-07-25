import os
import digital_rf
import sys
import threading
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

def drf_read_segment(dro, start_sample, nsamps):
    return dro.read_vector(start_sample, nsamps, 'ch0')

def drf_read_segment_2(dro, start_sample, end_sample):
    return dro.read(start_sample, end_sample, 'ch0')


def main():
    args = sys.argv[1:]
    Sdrf_dir = args[0]
    data_date = args[1]
    datadir = os.path.join(Sdrf_dir, f'OBS{data_date}T00-00')
    metadir = os.path.join(datadir, 'ch0', 'metadata')
    dro = digital_rf.DigitalRFReader(datadir)
    
    start_index, end_index = dro.get_bounds('ch0')
    print(f'Start index: {start_index}\nEnd index:   {end_index}')
    cont_data_arr = dro.get_continuous_blocks(start_index, end_index, 'ch0')
    print(
        (
            'The following is a OrderedDict of all continuous block of data in'
            '(start_sample, length) format: %s'
        )
        % (str(cont_data_arr))
    )
    
    # nsamps = end_index - start_index
    
    # nsamps = 28800000
    nsamps = 14400000
    start_sample = list(cont_data_arr.keys())[0]
    # for subchannel in range(3):
    #     print("Reading vector", subchannel)
    #     result = dro.read_vector(start_sample, nsamps, 'ch0')
    #     df = pd.DataFrame(result)
    #     print(df)
    #     # print(result)
    #     print(
    #         'got %i samples starting at sample %i'
    #         % (len(result), start_sample)
    #     )
    #     start_sample += nsamps
    
    
    df = pd.DataFrame()
    end_hour = 25
    while start_sample < end_index and end_hour > (start_sample-start_index)/8000/3600:
        print(f"At hour {(start_sample-start_index)/8000/3600}")
        result = dro.read_vector(start_sample, nsamps, 'ch0')
        df = pd.concat([df, pd.DataFrame(result)])
        start_sample += nsamps
        print(df.shape)
    
    # t1 = threading.Thread(target=drf_read_segment, args=(dro, 13700275200000, 1000000))
    # t2 = threading.Thread(target=drf_read_segment, args=(dro, 13700375200000, 1000000))
    # t3 = threading.Thread(target=drf_read_segment, args=(dro, 13700475200000, 1000000))
    
    # t1 = threading.Thread(target=drf_read_segment_2, args=(dro, 13700275200000, 13700276200000))
    # t2 = threading.Thread(target=drf_read_segment_2, args=(dro, 13700375200000, 13700376200000))
    # t3 = threading.Thread(target=drf_read_segment_2, args=(dro, 13700475200000, 13700476200000))
    
    # t1.start()
    # t2.start()
    # t3.start()
    
    # t1.join()
    # t2.join()
    # t3.join()
    
    
    fs = 8000
    radio_index = 2
    f, t_spec, Sxx = signal.spectrogram(df[radio_index],fs=fs,window='hann',nperseg=int(fs/0.01))
    Sxx_db = 10*np.log10(Sxx)
    
    flim = (990,1010)
    fig = plt.figure(figsize=(20,6))
    ax  = fig.add_subplot(1,1,1)
    mpbl = ax.pcolormesh(t_spec,f,Sxx_db)
    cbar = fig.colorbar(mpbl,label='PSD [dB]')
    # ax.set_title(dir_path+"_R"+str(radio_index+1))
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(flim)
    fig.savefig('Generated_Plots/grape_spectrogram.png',bbox_inches='tight')
    
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
        print(str(e))