#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import random, time, os
from IPython.display import Audio
from sklearn import metrics
from scipy.io import wavfile
import librosa
import librosa.display
import re

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

params = {'n_mels': 128,
          'n_fft': 1024,
          'hop_length': 512,
          'power': 2.0}

def load_audio(audio_path):
    return librosa.load(audio_path, sr=None, duration=10)

def plot_audio(audio_data):
    # Intervalle de temps entre deux points.
    audio, fe = load_audio(audio_data)
    dt= 1/fe
    # Variable de temps en seconde.
    t = dt*np.arange(len(audio)) 
    plt.plot(t,audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Amplitude en fonction du temps");
    

    
def spectrogram(audio_data, fe, dt):
    
    return np.abs(librosa.stft(audio_data, n_fft=int(dt*fe), hop_length=int(dt*fe)))
       
def plot_spectrogram(audio, fe, label="normal", dt=0.025):
    im = spectrogram(audio, fe, dt)
    sns.heatmap(np.rot90(im.T), cmap='inferno', vmin=0, vmax=np.max(im)/3)
    loc, labels = plt.xticks()
    l = np.round((loc-loc.min())*len(audio)/fe/loc.max(), 2)
    plt.xticks(loc, l)
    loc, labels = plt.yticks()
    l = np.array(loc[::-1]*fe/2/loc.max(), dtype=int)
    plt.yticks(loc, l)
    plt.xticks(rotation=90)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram ");
    

def logMelSpectrogram(audio_data, fe, dt):
    # Spectrogram
   
    stfts = np.abs(librosa.stft(audio_data,
                        n_fft = int(dt*fe),
                        hop_length = int(dt*fe),
                        center = True
                        )).T
    
    num_spectrogram_bins = stfts.shape[-1]
    # MEL filter
    linear_to_mel_weight_matrix = librosa.filters.mel(
                                sr=fe,
                                n_fft=int(dt*fe) + 1,
                                n_mels=num_spectrogram_bins,
                    ).T

    # Apply the filter to the spectrogram
    mel_spectrograms = np.tensordot(
                stfts,
                linear_to_mel_weight_matrix,
                1
            )
    return np.log(mel_spectrograms + 1e-6)

def plot_logMelSpectrogram(audio, fe=16000, dt=0.025):
    sns.heatmap(np.rot90(logMelSpectrogram(audio, fe, dt)), cmap='inferno', vmin = -6)
    loc, labels = plt.xticks()
    l = np.round((loc-loc.min())*len(audio)/fe/loc.max(), 2)
    plt.xticks(loc, l)
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Mel)")
    plt.title("LogMelSpectrogram ");
    
def logMelEnergy(audio, fe):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=fe,
                                                     n_fft=1024,
                                                     hop_length=512,
                                                     n_mels=128,
                                                     power=2.0)
    # MFECs
    log_mel_energy = librosa.core.power_to_db(mel_spectrogram)
    return log_mel_energy

def plot_logMelEnergies(audio, fe):   
    img = librosa.display.specshow(logMelEnergy(audio, fe), sr=fe, x_axis='time')
    plt.colorbar(img, format="%+2.f")
    plt.xlabel('Time (s)')
    plt.ylabel('MFEC');
    

