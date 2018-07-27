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
    print('Support Vector accuracy: {0}'.format(
        captchalyzer.classify_through_support_vetor(data_for_analysis)['accuracy']))
    print('Random Forest accuracy: {0}'.format(
        captchalyzer.classify_through_random_forest(data_for_analysis)['accuracy']))


if __name__ == '__main__':
    main()
