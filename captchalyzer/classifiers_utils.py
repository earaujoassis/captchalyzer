# -*- coding: utf-8 -*-

import numpy as np
import functools

from collections import Counter


rng = np.random.RandomState(3141592653)


class DataFile(object):
    def __init__(self, X, y=None, expectation=None):
        self.expectation = expectation
        self._rawX = X
        self.X = np.vstack([segment for segment in X])
        if not y is None:
            self._rawY = y
            self.y = np.chararray(
                (functools.reduce(
                    lambda x, y: x + y, [segment.shape[0] for segment in X]),))
            for i, letter in enumerate(y):
                self.y[i * len(X[0]):(i + 1) * len(X[0])] = letter

    def infer_prediction(self, predictions):
        X = self._rawX
        letters = []
        for i, letter in enumerate(self.expectation):
            letters.append(str(Counter(predictions[i * len(X[0]):(i + 1) * len(X[0])]).most_common(1)[0][0]))
        return letters


def result_object(entry, obj={}):
    result = {
        'filename': entry['file'],
        'audio_length': entry['audio_length'],
        'characters': entry['characters'],
    }
    result.update(obj)

    return result


def general_classifier(classification_data, clf):
    results = []
    errors_counter = 1
    failed_counter = 1
    succeeded_counter = 1
    local_accuracies = 1
    for entry in classification_data['training_data']:
        clf.fit(entry['data'].X, entry['data'].y)
    samples_counter = len(classification_data['validation_data'])
    for entry in classification_data['validation_data']:
        data_file = entry['data']
        try:
            predictions = data_file.infer_prediction(clf.predict(data_file.X))
            predictions_size = len(predictions)
            expectation = data_file.expectation
            difference = [i for i in range(predictions_size) if predictions[i] != expectation[i]]
            difference_size = len(difference)
            local_accuracy = (predictions_size - difference_size) / predictions_size
            local_accuracies += local_accuracy
            if difference_size > 0:
                errors_counter += 1
            succeeded_counter += 1
            results.append(
                result_object(entry, {
                    'succeeded': True,
                    'prediction': predictions,
                    'accuracy': local_accuracy,
                    'different_positions': difference
                })
            )
        except Exception as e:
            print(e)
            failed_counter += 1
            results.append(
                result_object(entry, {
                    'succeeded': False,
                    'error_reason': str(e).strip()
                })
            )
    return {
        'overall_accuracy': (succeeded_counter - errors_counter) / succeeded_counter,
        'mean_local_accuracies': local_accuracies / succeeded_counter,
        'total_errors': errors_counter,
        'total_failed': failed_counter,
        'total_succeeded': succeeded_counter,
        'samples_results': results,
    }
