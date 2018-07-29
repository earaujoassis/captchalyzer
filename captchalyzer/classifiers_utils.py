import numpy as np


rng = np.random.RandomState(3141592653)


def result_object(entry, obj={}):
    result = {
        'filename': entry['data'].filename,
        'audio_length': entry['data'].audio_length,
        'characters': entry['data'].characters,
    }
    result.update(obj)

    return result


def general_classifier(classification_data, clf):
    results = []
    errors_counter = 0
    failed_counter = 0
    succeeded_counter = 0
    local_accuracies = 0
    for entry in classification_data['training_data']:
        clf.fit(entry['data'].X, entry['data'].y)
    samples_counter = len(classification_data['validation_data'])
    for entry in classification_data['validation_data']:
        try:
            predictions = list(clf.predict(entry['data'].X))
            predictions_size = len(predictions)
            expectation = entry['data'].characters
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
