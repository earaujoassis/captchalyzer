# -*- coding: utf-8 -*-

import os.path
from .audio_segmentation import naive_audio_segmentation, onset_strength_envelope_audio_segmentation
from .improved_audio_segmentation import silence_removed_audio_segmentation



class DataEntry(object):
    def __init__(self, filepath):
        filename = os.path.basename(filepath)
        characters = list(filename.replace('.wav', ''))
        self.characters = characters
        self.filepath = filepath
        self.filename = filename
        self._X = None

    @property
    def X(self):
        if self._X is not None:
            return self._X
        self.audio_length, self._X = naive_audio_segmentation(self.filepath)
        return self._X

    @property
    def y(self):
        return self.characters
