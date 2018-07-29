# -*- coding: utf-8 -*-

import os
import librosa
import json
import numpy as np
from .data_entry import DataEntry

from collections import Counter


def collection_data(folders=[], training=True):
    total_files = 0
    audios_data = []
    for folder in folders:
        filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for filename in filenames:
            total_files += 1
            filepath = os.path.join(folder, filename)
            entry = {
                'data': DataEntry(filepath),
                'filepath': filepath
            }
            audios_data.append(entry)
    return audios_data


def output_classifier_results(results, filename='results.json'):
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, 'w') as outfile:
        json.dump(results, outfile, indent=4, separators=(',', ': '))


def basic_statistics(collection=[]):
    all_audios_length = list(map(lambda x: x['data'].audio_length, collection))
    all_characters = list(map(lambda x: x['data'].characters, collection))
    audios_characters_counter = Counter(np.array(all_characters).flatten())

    return {
        'mean_audio_length': np.mean(all_audios_length),
        'smallest_audio_length': np.min(all_audios_length),
        'largest_audio_length': np.max(all_audios_length),
        'audios_characters_counter': audios_characters_counter
    }
