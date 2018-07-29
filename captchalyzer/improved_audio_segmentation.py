# -*- coding: utf-8 -*-

import numpy as np
import os.path
import librosa
import librosa.feature
import matplotlib.pyplot as plt
from .classifiers_utils import rng
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.fftpack.realtransforms import dct
from scipy.fftpack import fft
from pydub import AudioSegment


##
# The functions belows were obtained from the pyAnalysis libary. Captchalyzer uses some
# of the techniques presented by the pyAnalysis library to segment audio files.
#
# The technique is explained in details below, but basically:
#  1. It extracts short-term features from the audio data
#  2. It trains a binary SVM classifier of low vs high energy frames
#. 3. It detects onset frame indices
#  4. It groups frame indices to onset segments
#  5. It removes very small segments from the output
#. 6. It returns an array with tuples of segments; silence segments were removed
#
# pyAnalysis was created by Theodoros Giannakopoulos and
# LICENSED under the Apache License Version 2.0
# It is available at https://github.com/tyiannak/pyAudioAnalysis


eps = 0.00000001


def silence_removed_audio_segmentation(filename, limit=4, plot=False):
    data_per_segmentation = []
    audio_data, sampling_rate = readAudioFile(filename)
    audio_length = audio_data.shape[0] / sampling_rate
    segments = silenceRemoval(audio_data, sampling_rate, 0.020, 0.020, 1.0, 0.3, plot=plot)
    if len(segments) > limit:
        segments = segments[0:limit]
    for segment_start, segment_end in segments:
        local_audio_data = audio_data[int(segment_start * sampling_rate):int(segment_end * sampling_rate)]
        audio_mfcc = librosa.feature.mfcc(local_audio_data, sr=sampling_rate).T
        scaler = StandardScaler()
        scaled_audio_mfcc = scaler.fit_transform(audio_mfcc)
        reshaped_scaled_audio = np.reshape(scaled_audio_mfcc, (np.product(scaled_audio_mfcc.shape),))
        data_per_segmentation.append(reshaped_scaled_audio)

    return audio_length, np.array(data_per_segmentation)


def readAudioFile(filename):
    '''
    This function returns a numpy array that stores the audio samples of a specified WAV of AIFF file
    Modified by Ewerton Carlos Assi (MIT License, 2018); inspired by pyAnalysis (Apache License)
    '''
    if not os.path.isfile(filename):
        return

    audiofile = AudioSegment.from_file(filename)
    if audiofile.sample_width == 2:
        data = np.fromstring(audiofile._data, np.int16)
    elif audiofile.sample_width == 4:
        data = np.fromstring(audiofile._data, np.int32)
    else:
        return (-1, -1)
    Fs = audiofile.frame_rate
    x = []
    for chn in range(audiofile.channels):
        x.append(data[chn::audiofile.channels])
    x = np.array(x, dtype=float).T

    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x.flatten()

    return x, Fs


def silenceRemoval(x, Fs, stWin, stStep, smoothWindow, Weight, plot=False):
    '''
    Event Detection (silence removal)
    ARGUMENTS:
         - x:                the input audio signal
         - Fs:               sampling freq
         - stWin, stStep:    window size and step in seconds
         - smoothWindow:     (optinal) smooth window (in seconds)
         - Weight:           (optinal) weight factor (0 < Weight < 1) the higher, the more strict
         - plot:             (optinal) True if results are to be plotted
    RETURNS:
         - segmentLimits:    list of segment limits in seconds (e.g [[0.1, 0.9], [1.4, 3.0]] means that
                             the resulting segments are (0.1 - 0.9) seconds and (1.4, 3.0) seconds
    '''

    if Weight >= 1:
        Weight = 0.99
    if Weight <= 0:
        Weight = 0.01

    # Step 1: Short-term feature extraction
    ShortTermFeatures = stFeatureExtraction(x, Fs, stWin * Fs, stStep * Fs)

    # Step 2: Train binary SVM classifier of low vs high energy frames
    EnergySt = ShortTermFeatures[1, :]                                  # keep only the energy short-term sequence (2nd feature)
    E = np.sort(EnergySt)                                               # sort the energy feature values:
    L1 = int(len(E) / 10)                                               # number of 10% of the total short-term windows
    T1 = np.mean(E[0:L1]) + 0.000000000000001                           # compute "lower" 10% energy threshold
    T2 = np.mean(E[-L1:-1]) + 0.000000000000001                         # compute "higher" 10% energy threshold
    Class1 = ShortTermFeatures[:, np.where(EnergySt <= T1)[0]]          # get all features that correspond to low energy
    Class2 = ShortTermFeatures[:, np.where(EnergySt >= T2)[0]]          # get all features that correspond to high energy
    featuresSS = [Class1.T, Class2.T]                                   # form the binary classification task and ...

    [featuresNormSS, MEANSS, STDSS] = normalizeFeatures(featuresSS)     # normalize and ...
    SVM = trainSVM(featuresNormSS, 1.0)                                 # train the respective SVM probabilistic model (ONSET vs SILENCE)

    # Step 3: Compute onset probability based on the trained SVM
    ProbOnset = []
    for i in range(ShortTermFeatures.shape[1]):                         # for each frame
        curFV = (ShortTermFeatures[:, i] - MEANSS) / STDSS              # normalize feature vector
        ProbOnset.append(SVM.predict_proba(curFV.reshape(1, -1))[0][1]) # get SVM probability (that it belongs to the ONSET class)
    ProbOnset = np.array(ProbOnset)
    ProbOnset = smoothMovingAvg(ProbOnset, smoothWindow / stStep)       # smooth probability

    # Step 4: detect onset frame indices:
    ProbOnsetSorted = np.sort(ProbOnset)      # find probability Threshold as a weighted average of top 10% and lower 10% of the values
    Nt = int(ProbOnsetSorted.shape[0] / 10)
    T = (np.mean((1 - Weight) * ProbOnsetSorted[0:Nt]) + Weight * np.mean(ProbOnsetSorted[-Nt::]))

    MaxIdx = np.where(ProbOnset > T)[0]                                 # get the indices of the frames that satisfy the thresholding
    i = 0
    timeClusters = []
    segmentLimits = []

    # Step 5: Group frame indices to onset segments
    while i < len(MaxIdx):                                              # for each of the detected onset indices
        curCluster = [MaxIdx[i]]
        if i == len(MaxIdx) - 1:
            break
        while MaxIdx[i + 1] - curCluster[-1] <= 2:
            curCluster.append(MaxIdx[i + 1])
            i += 1
            if i == len(MaxIdx) - 1:
                break
        i += 1
        timeClusters.append(curCluster)
        segmentLimits.append([curCluster[0] * stStep, curCluster[-1] * stStep])

    # Step 6: Post process: remove very small segments:
    minDuration = 0.2
    segmentLimits2 = []
    for s in segmentLimits:
        if s[1] - s[0] > minDuration:
            segmentLimits2.append(s)
    segmentLimits = segmentLimits2

    if plot:
        timeX = np.arange(0, x.shape[0] / float(Fs), 1.0 / Fs)

        plt.subplot(2, 1, 1)
        plt.plot(timeX, x)
        for s in segmentLimits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(0, ProbOnset.shape[0] * stStep, stStep), ProbOnset)
        plt.title('Signal')
        for s in segmentLimits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.title('SVM Probability')
        plt.show()

    return segmentLimits


def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(countZ) / np.float64(count - 1.0)


def stEnergy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))


def stEnergyEntropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    Eol = np.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(np.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
            frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -np.sum(s * np.log2(s + eps))
    return Entropy


def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(X) + 1)) * (fs / (2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return C, S


def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = np.sum(X ** 2)               # total spectral energy

    subWinLength = int(np.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = np.sum(subWindows ** 2, axis=0) / (Eol + eps)                         # compute spectral sub-energies
    En = -np.sum(s*np.log2(s + eps))                                          # compute spectral entropy

    return En


def stSpectralFlux(X, Xprev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = np.sum(X + eps)
    sumPrevX = np.sum(Xprev + eps)
    F = np.sum((X / sumX - Xprev / sumPrevX) ** 2)

    return F


def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = np.sum(X ** 2)
    fftLength = len(X)
    Threshold = c * totalEnergy

    # Find the spectral rolloff as the frequency position where the respective
    # spectral energy is equal to `c * totalEnergy`
    CumSum = np.cumsum(X ** 2) + eps
    [a, ] = np.nonzero(CumSum > Threshold)
    if len(a) > 0:
        mC = np.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return mC


def stMFCC(X, fbank, nceps):
    """
    Computes the MFCCs of a frame, given the fft mag

    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:     MFCCs (13 element vector)

    Note: MFCC calculation is, in general, taken from the scikits.talkbox library (MIT Licence),
          with a small number of modifications to make it more compact and suitable for the pyAudioAnalysis lib
    """

    mspec = np.log10(np.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps


def stFeatureExtraction(signal, Fs, Win, Step):
    """
    This function implements the short-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = np.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)                                      # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2

    fbank, freqs = mfccInitFilterBanks(Fs, nFFT)       # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, Fs)

    numOfTimeSpectralFeatures = 8
    numOfHarmonicFeatures = 0
    nceps = 13
    numOfChromaFeatures = 13
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures + numOfChromaFeatures

    stFeatures = []
    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos + Win]                  # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:int(nFFT)]                               # normalize fft
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = np.zeros((totalNumOfFeatures, 1))
        curFV[0] = stZCR(x)                              # zero crossing rate
        curFV[1] = stEnergy(x)                           # short-term energy
        curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        curFV[3], curFV[4] = stSpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread
        curFV[5] = stSpectralEntropy(X)                  # spectral entropy
        curFV[6] = stSpectralFlux(X, Xprev)              # spectral flux
        curFV[7] = stSpectralRollOff(X, 0.90, Fs)        # spectral rolloff
        curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures + nceps, 0] = stMFCC(X, fbank, nceps).copy()    # MFCCs

        chromaNames, chromaF = stChromaFeatures(X, Fs, nChroma, nFreqsPerChroma)
        curFV[numOfTimeSpectralFeatures + nceps:numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF
        curFV[numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF.std()
        stFeatures.append(curFV)
        Xprev = X.copy()

    stFeatures = np.concatenate(stFeatures, 1)
    return stFeatures


def stChromaFeatures(X, fs, nChroma, nFreqsPerChroma):
    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X ** 2
    if nChroma.max() < nChroma.shape[0]:
        C = np.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:
        I = np.nonzero(nChroma > nChroma.shape[0])[0][0]
        C = np.zeros((nChroma.shape[0],))
        C[nChroma[0:I - 1]] = spec
        C /= nFreqsPerChroma
    finalC = np.zeros((12, 1))
    newD = int(np.ceil(C.shape[0] / 12.0) * 12)
    C2 = np.zeros((newD,))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0] / 12), 12)
    finalC = np.matrix(np.sum(C2, axis=0)).T
    finalC /= spec.sum()

    return chromaNames, finalC


def normalizeFeatures(features):
    '''
    This function normalizes a feature set to 0-mean and 1-std.
    Used in most classifier trainning cases.

    ARGUMENTS:
        - features:    list of feature matrices (each one of them is a numpy matrix)
    RETURNS:
        - featuresNorm:    list of NORMALIZED feature matrices
        - MEAN:        mean vector
        - STD:        std vector
    '''
    X = np.array([])
    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = np.vstack((X, f))
            count += 1

    MEAN = np.mean(X, axis=0) + 0.00000000000001;
    STD = np.std(X, axis=0) + 0.00000000000001;

    featuresNorm = []
    for f in features:
        ft = f.copy()
        for nSamples in range(f.shape[0]):
            ft[nSamples,:] = (ft[nSamples,:] - MEAN) / STD
        featuresNorm.append(ft)
    return featuresNorm, MEAN, STD


def listOfFeatures2Matrix(features):
    '''
    listOfFeatures2Matrix(features)

    This function takes a list of feature matrices as argument and returns a single concatenated
    feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - X:            a concatenated matrix of features
        - Y:            a vector of class indeces
    '''

    X = np.array([])
    Y = np.array([])
    for i, f in enumerate(features):
        if i == 0:
            X = f
            Y = i * np.ones((len(f), 1))
        else:
            X = np.vstack((X, f))
            Y = np.append(Y, i * np.ones((len(f), 1)))
    return X, Y


def smoothMovingAvg(inputSignal, windowLen=11):
    windowLen = int(windowLen)
    if inputSignal.ndim != 1:
        raise ValueError("")
    if inputSignal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return inputSignal
    s = np.r_[2 * inputSignal[0] - inputSignal[windowLen - 1::-1],
              inputSignal,
              2 * inputSignal[-1] - inputSignal[-1:-windowLen:-1]]
    w = np.ones(windowLen, 'd')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[windowLen:-windowLen + 1]


def mfccInitFilterBanks(fs, nfft):
    """
    Computes the triangular filterbank for MFCC computation (used in the stFeatureExtraction
    function before the stMFCC function call). This function is taken from the scikits.talkbox
    library (MIT Licence): https://pypi.python.org/pypi/scikits.talkbox
    """

    # filter bank params:
    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = np.zeros(nFiltTotal + 2)
    freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** np.arange(1, numLogFilt + 3)
    heights = 2. / (freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nFiltTotal, int(nfft)))
    nfreqs = np.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i + 1]
        highTrFreq = freqs[i + 2]

        lid = np.arange(np.floor(lowTrFreq * nfft / fs) + 1, np.floor(cenTrFreq * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = np.arange(np.floor(cenTrFreq * nfft / fs) + 1, np.floor(highTrFreq * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def stChromaFeaturesInit(nfft, fs):
    """
    This function initializes the chroma matrices used in the calculation of the chroma features
    """
    freqs = np.array([((f + 1) * fs) / (2 * nfft) for f in range(int(nfft))])
    Cp = 27.50
    nChroma = np.round(12.0 * np.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = np.zeros((nChroma.shape[0],))

    uChroma = np.unique(nChroma)
    for u in uChroma:
        idx = np.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape

    return nChroma, nFreqsPerChroma


def trainSVM(features, Cparam):
    '''
    Train a multi-class probabilitistic SVM classifier.
    Note:     This function is simply a wrapper to the sklearn functionality for SVM training
              See function trainSVM_feature() to use a wrapper on both the feature extraction
              and the SVM training (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [numOfSamples x numOfDimensions]
        - Cparam:           SVM parameter C (cost of constraints violation)
    RETURNS:
        - svm:              the trained SVM variable

    NOTE:     This function trains a linear-kernel SVM for a given C value. For a different
              kernel, other types of parameters should be provided.
    '''

    X, Y = listOfFeatures2Matrix(features)
    svm = SVC(C=Cparam, kernel='linear',  probability=True, random_state=rng)
    svm.fit(X, Y)

    return svm
