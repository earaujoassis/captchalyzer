# -*- coding: utf-8 -*-

import os
import librosa
import librosa.feature
import numpy as np
from sklearn.preprocessing import StandardScaler


def naive_audio_segmentation(filename, limit=4):
    if not os.path.isfile(filename):
        return

    audio_data, sampling_rate = librosa.load(filename, sr=None, mono=True)
    data_per_segmentation = []
    audio_length = audio_data.shape[0] / sampling_rate
    step = int(np.floor(audio_length / limit))
    window_size = sampling_rate * step
    steps = list(range(0, audio_data.shape[0], window_size))
    for i in range(limit):
        local_audio_data = audio_data[steps[i]:steps[i + 1]]
        audio_mfcc = librosa.feature.mfcc(local_audio_data, sr=sampling_rate, n_mfcc=12).T
        scaler = StandardScaler()
        scaled_audio_mfcc = scaler.fit_transform(audio_mfcc)
        data_per_segmentation.append(scaled_audio_mfcc)

    return data_per_segmentation


def onset_strength_envelope_audio_segmentation(filename, limit=4):
    if not os.path.isfile(filename):
        return

    data_per_segmentation = []
    audio_data, sampling_rate = librosa.load(filename, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sampling_rate)
    onset_times = librosa.frames_to_time(onset_frames, sr=sampling_rate)

    return data_per_segmentation
