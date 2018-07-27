# -*- coding: utf-8 -*-

import os
import librosa
import librosa.display
import pandas as pd
import numpy as np


##
# References
# http://zderadicka.eu/decoding-audio-captchas-in-python/
# https://github.com/izderadicka/adecaptcha


def naive_audio_segmentation(filename, limit=4):
    if not os.path.isfile(filename):
        return

    data_per_segmentation = []
    audio_data, sampling_rate = librosa.load(filename, sr=None, mono=False)
    audio_length = audio_data.shape[0] / sampling_rate
    step = int(np.floor(audio_length / limit))
    window_size = sampling_rate * step
    steps = list(range(0, audio_data.shape[0], window_size))
    for i in range(limit):
        data_per_segmentation.append(pd.Series(audio_data[steps[i]:steps[i + 1]]))

    return data_per_segmentation


def onset_strength_envelope_audio_segmentation(filename, limit=4):
    if not os.path.isfile(filename):
        return

    data_per_segmentation = []
    audio_data, sampling_rate = librosa.load(filename, sr=None, mono=False)
    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sampling_rate)
    onset_times = librosa.frames_to_time(onset_frames, sr=sampling_rate)

    return data_per_segmentation
