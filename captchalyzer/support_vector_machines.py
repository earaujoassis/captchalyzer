# -*- coding: utf-8 -*-

from .classifiers_utils import rng

from sklearn.svm import SVC


def classify_through_support_vetor(classification_data={}):
    errors_counter = 0
    predictions_per_sample = []
    clf = SVC(random_state=rng)
    for entry in classification_data['training_data']:
        clf.fit(entry['data'].X, entry['data'].y)
    samples_counter = len(classification_data['validation_data'])
    for entry in classification_data['validation_data']:
        try:
            predictions = ''.join(clf.predict(entry['data'].X))
            predictions_per_sample.append(predictions)
            if not ''.join(predictions) == entry['data'].expectation:
                errors_counter += 1
        except Exception as e:
            errors_counter += 1
    return {
        'predictions_per_sample': predictions_per_sample,
        'accuracy': (samples_counter - errors_counter) / samples_counter,
        'errors': errors_counter
    }
