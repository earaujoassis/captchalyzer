# -*- coding: utf-8 -*-

from .classifiers_utils import rng, general_classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def classify_through_random_forest(classification_data={}, random_forest_max_depth=4):
    clf = RandomForestClassifier(max_depth=random_forest_max_depth, random_state=rng)
    return general_classifier(classification_data, clf)


def classify_through_support_vetor(classification_data={}):
    clf = SVC(kernel='linear', probability=True, random_state=rng)
    return general_classifier(classification_data, clf)


def classify_through_kneighbors(classification_data={}, n_neighbors=4):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    return general_classifier(classification_data, clf)


def classify_through_discriminant_analysis(classification_data={}):
    clf = QuadraticDiscriminantAnalysis()
    return general_classifier(classification_data, clf)
