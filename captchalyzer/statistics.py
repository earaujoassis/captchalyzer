# -*- coding: utf-8 -*-

import os
import librosa
import numpy as np

from collections import Counter
from .classifiers_utils import DataFile
from .audio_segmentation import naive_audio_segmentation, onset_strength_envelope_audio_segmentation


def collection_data(folders=[], training=True):
    total_files = 0
    audios_data = []
    for folder in folders:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            total_files += 1
            filepath = os.path.join(folder, file)
            audio_data, sampling_rate = librosa.load(filepath, sr=None, mono=False)
            characters_str = file.replace('.wav', '')
            characters = list(characters_str)
            entry = {
                'audio_length': audio_data.shape[0] / sampling_rate,
                'filepath': filepath,
                'file': file,
                'characters': characters
            }
            if training:
                entry['data'] = DataFile(
                    X=naive_audio_segmentation(filepath),
                    y=characters)
            else:
                entry['data'] = DataFile(
                    X=naive_audio_segmentation(filepath),
                    expectation=characters_str)
            audios_data.append(entry)
    return audios_data


def basic_statistics(collection=[]):
    all_audios_length = list(map(lambda x: x['audio_length'], collection))
    all_characters = list(map(lambda x: x['characters'], collection))
    audios_characters_counter = Counter(np.array(all_characters).flatten())

    return {
        'mean_audio_length': np.mean(all_audios_length),
        'smallest_audio_length': np.min(all_audios_length),
        'largest_audio_length': np.max(all_audios_length),
        'audios_characters_counter': audios_characters_counter
    }
