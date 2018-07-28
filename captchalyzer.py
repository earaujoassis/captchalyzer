#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import captchalyzer

def main():
    training_collection = captchalyzer.collection_data(
        ['audiofiles/base_treinamento_I'])
    validation_collection = captchalyzer.collection_data(
        ['audiofiles/base_validacao_I'], training=False)
    data_for_analysis = {
        'training_data': training_collection,
        'validation_data': validation_collection
    }
    captchalyzer.output_classifier_results(
        captchalyzer.classify_through_support_vetor(data_for_analysis),
        filename='support_vetor_results.json')
    captchalyzer.output_classifier_results(
        captchalyzer.classify_through_random_forest(data_for_analysis),
        filename='random_forest_results.json')
    captchalyzer.output_classifier_results(
        captchalyzer.classify_through_kneighbors(data_for_analysis),
        filename='kneighbors_results.json')
#    captchalyzer.output_classifier_results(
#        captchalyzer.classify_through_discriminant_analysis(data_for_analysis),
#        filename='discriminant_analysis_results.json')


if __name__ == '__main__':
    main()
