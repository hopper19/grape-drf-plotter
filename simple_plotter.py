# A short program, not full featured but operational for plotting DigitalRF data
# Author: Cuong Nguyen

# TODO: marked for archive
import os
import digital_rf
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib as mpl
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import pytz
import pickle
import sys
import logging

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin'] = 0

def main():
    datadir = "/home/cuong/drive/GRAPE2-SFTP/w2naf"
    metadir = os.path.join(datadir, 'ch0', 'metadata')
    dro = digital_rf.DigitalRFReader(datadir)
    dmr = digital_rf.DigitalMetadataReader(metadir)
    
    start_index, end_index = dro.get_bounds('ch0')
    print(f'Start index: {start_index}\nEnd index:   {end_index}')
    
    radio_index = 2    
    
    ####################   
    rf_dict = dro.get_properties('ch0', sample=start_index)
    utc_date = datetime.fromtimestamp(
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
    # subdir to store the plots
    output_subdir = 'g2plots'
    event_fname = f'{utc_date}_{station}_{output_subdir}'
    png_fname = event_fname + '.png'
    output_dir = os.path.join('output', output_subdir)
    png_fpath = os.path.join(output_dir, png_fname)
    ba_fpath = os.path.join(output_dir, event_fname + f'_{radio_index}.ba.pkl')
    #######################################
    
    cont_data_arr = dro.get_continuous_blocks(start_index, end_index, 'ch0')
    print(
        'OrderedDict of all continuous block of data in'
        '(start_sample, length) format: %s' % str(cont_data_arr)
    )

    df = pd.DataFrame()
    if not os.path.exists(ba_fpath):
        nsamps = 14400000
        start_sample = list(cont_data_arr.keys())[0]

        end_hour = 25

        while start_sample < end_index and end_hour > (start_sample - start_index) / 8000 / 3600:
            print(f"At hour {(start_sample - start_index) / 8000 / 3600}")
            result = dro.read_vector(start_sample, nsamps, 'ch0')
            df = pd.concat([df, pd.DataFrame(result)])
            start_sample += nsamps
            print(df.shape)
    else:
        print('Using cached file {!s}...'.format(ba_fpath))
        with open(ba_fpath,'rb') as fl:
            df = pickle.load(fl)        

    print(f'Now plotting {event_fname}...')

    # downsample signal to 2000 Hz bandwidth with scipy decimate
    decimation_factor = int(fs / 4000)
    df = signal.decimate(df, decimation_factor, ftype="fir", zero_phase=True)
    fs = 4000

    f, t_spec, Sxx = signal.spectrogram(df, fs=fs, window='hann', nperseg=int(fs / 0.01))
    Sxx_db = 10 * np.log10(Sxx)

    flim = (995, 1005)
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 1, 1)
    mpbl = ax.pcolormesh(t_spec, f, Sxx_db)
    cbar = fig.colorbar(mpbl, label='PSD [dB]')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(flim)
    fig.savefig('Generated_Plots/grape_spectrogram_4000.png', bbox_inches='tight')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e))
