# -*- coding: utf-8 -*-

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from .improved_audio_segmentation import improved_audio_segmentation


def waveplot(filename):
    if not os.path.isfile(filename):
        return

    audio_data, sampling_rate = librosa.load(filename, sr=None, mono=False)
    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(audio_data, sr=sampling_rate)
    plt.title("Waveplot")
    plt.show()


def silence_removed_audio_segmentation_plot(filename):
    improved_audio_segmentation(filename, plot=True)


def onset_detection_plot(filename):
    if not os.path.isfile(filename):
        return

    audio_data, sampling_rate = librosa.load(filename, sr=None, mono=False)
    o_env = librosa.onset.onset_strength(audio_data, sr=sampling_rate)
    onset_times = librosa.frames_to_time(np.arange(len(o_env)), sr=sampling_rate)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sampling_rate)

    D = librosa.stft(audio_data)
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             x_axis='time', y_axis='log')
    plt.title("Espectrograma de força")
    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(onset_times, o_env, label='Onset strength')
    plt.vlines(onset_times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
               linestyle='--', label='Onsets')
    plt.axis('tight')
    plt.legend(frameon=True, framealpha=0.75)
    plt.show()


def histogram_audio_characters_plot(collection=[]):
    labels, values = zip(*collection.items())
    indexes = np.arange(len(labels))
    width = 1

    plt.figure()
    plt.bar(indexes, values, width)
    plt.title("Histograma; distribuição dos caracteres")
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()


def histogram_audio_length_plot(collection=[]):
    plt.figure()
    plt.hist(list(map(lambda x: np.round(x['audio_length']), collection)), bins='auto')
    plt.title("Histograma; comprimento dos áudios em segundos")
    plt.show()
