# Samuel Moijueh
############################
## This script plots the mfccs
## code in calc_mfccs() obtained from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
############################

import warnings
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import seaborn as sns


fg_color = 'white'

def plot_mfccs(samples):
    warnings.filterwarnings('ignore')
    sns.set(font_scale=1)
    with plt.rc_context({'xtick.color':fg_color, 'ytick.color':fg_color}):
        # Temporary rc parameters in effect
        fig, ax = plt.subplots(figsize=(20,9), facecolor='None')
        plt.suptitle("Mel Frequency Cepstral Coefficients for t=2 seconds",x=0.5,y=1.05,fontsize=22, color=fg_color)
        for i, f in enumerate(samples, 1):
            if i < 6:
                mfcc = calc_mfccs(f)
                plt.subplot(2, 3, i)
                # interpolation might throw it off
                im = plt.imshow(mfcc.T, cmap=plt.cm.jet, aspect='auto', interpolation='nearest', vmin=np.amin(mfcc), vmax=np.amax(mfcc)+20)
                plt.xticks(np.arange(0, (mfcc.T).shape[1],
                (mfcc.T).shape[1] / 4.017),
                ['0', '0.5', '1.0', '1.5','2.0','2.5'])  
                plt.yticks(np.arange(0, mfcc.shape[1],1.89),
                ['0', '2', '4', '6','8','10', '12'])
                ax = plt.gca()
                ax.invert_yaxis()
                plt.title(f.split("_")[0], fontsize=18, color=fg_color)
                plt.xlabel('Time (seconds)', fontsize=14, color=fg_color)
                plt.ylabel('Cepstral Coefficients', fontsize=14, color=fg_color)
        cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()


def calc_mfccs(file):
    sample_rate, signal = scipy.io.wavfile.read('two_second_audio/' + file) 
    pre_emphasis = 0.97

    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_size = 0.025
    frame_stride = 0.01

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)

    NFFT = 512

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    nfilt = 40

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    num_ceps = 13

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    
    cep_lifter = 22

    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return(mfcc)