import os
import shutil
import csv

import jiwer
import pandas as pd
from pydub import AudioSegment
import re
import wave
import contextlib
import speech_recognition as sr
from num2words import num2words
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import statistics
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt2
from scipy.io import wavfile # get the api
from scipy.fftpack import fft
from pylab import *
import scipy



SPHINX = 3
GOOGLE = 4
GOOGLE_CLOUD = 5
WIT = 6
MICROSOFT = 7
IBM = 8

SPHINX_ARRAY = 0
GOOGLE_ARRAY = 1
GOOGLE_CLOUD_ARRAY = 2
WIT_ARRAY = 3
MICROSOFT_ARRAY = 4
IBM_ARRAY = 5

FRAMEWORK_ARRAY = ['SPHINX', 'GOOGLE', 'GOOGLE_CLOUD', 'WIT']

ATCO2_ARRAY_INDEX = 0
ATCOSIM_ARRAY_INDEX = 1

clipsDirectory = "/home/natalia/bachelors/en/clips"
testCsv = "/home/natalia/bachelors/en/test.tsv"
recognitionResult = "/home/natalia/bachelors/en/recognition.tsv"
googleCLoudFile = "/home/natalia/bachelors/keys/google_cloud.json"

ground_truth = [[], []]  # extend if microsoft and ibm used
hypothesis = [[[], [], [], []], [[], [], [], []]]  # 0. index is for ATCO2, 1. for ATCOSIM


def calculate_WER():
    googleCloudKey = ""
    with open(googleCLoudFile, 'r') as file:
        googleCloudKey = file.read().replace('\n', '')

    testCsvFile = open(testCsv, mode='r', encoding='utf-8')
    reader = csv.reader(testCsvFile, delimiter="\t")
    out_file = open(recognitionResult, mode='wt', encoding='utf-8')
    # with open(recognitionResult, 'wt') as out_file:  # processing ATCOSIM data
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for rownum, line in enumerate(reader):
        if line[1] == "path":
            line[SPHINX] = 'Recognized by sphinx'
            line[GOOGLE] = 'Recognized by google'
            line[GOOGLE_CLOUD] = 'Recognized by google cloud'
            line[WIT] = 'Recognized by wit.ai'
            line[MICROSOFT] = 'Recognized by Microsoft Bing'
            line[IBM] = 'Recognized by IBM'
            tsv_writer.writerow(line)
            continue

        ground_truth.append(line[2])

        AUDIO_FILE = clipsDirectory + "/" + line[1] + ".wav"

        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)  # read the entire audio file

        # recognize speech using Sphinx
        try:
            line[SPHINX] = digit_to_text(r.recognize_sphinx(audio))
        except sr.UnknownValueError:
            line[SPHINX] = ""
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            line[SPHINX] = ""
            print("Sphinx error; {0}".format(e))
        if if_ATCO2(line[1]):
            hypothesis[ATCO2_ARRAY_INDEX][SPHINX_ARRAY].append(line[SPHINX])
        else:
            hypothesis[ATCOSIM_ARRAY_INDEX][SPHINX_ARRAY].append(line[SPHINX])

        # recognize speech using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            line[GOOGLE] = digit_to_text(r.recognize_google(audio))
        except sr.UnknownValueError:
            line[GOOGLE] = ""
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            line[GOOGLE] = ""
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        if if_ATCO2(line[1]):
            hypothesis[ATCO2_ARRAY_INDEX][GOOGLE_ARRAY].append(line[GOOGLE])
        else:
            hypothesis[ATCOSIM_ARRAY_INDEX][GOOGLE_ARRAY].append(line[GOOGLE])

        # recognize speech using Google Cloud Speech
        GOOGLE_CLOUD_SPEECH_CREDENTIALS = googleCloudKey
        try:
            line[GOOGLE_CLOUD] = digit_to_text(
                r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS))
        except sr.UnknownValueError:
            line[GOOGLE_CLOUD] = ""
            print("Google Cloud Speech could not understand audio")
        except sr.RequestError as e:
            line[GOOGLE_CLOUD] = ""
            print("Could not request results from Google Cloud Speech service; {0}".format(e))
        if if_ATCO2(line[1]):
            hypothesis[ATCO2_ARRAY_INDEX][GOOGLE_CLOUD_ARRAY].append(line[GOOGLE_CLOUD])
        else:
            hypothesis[ATCOSIM_ARRAY_INDEX][GOOGLE_CLOUD_ARRAY].append(line[GOOGLE_CLOUD])

        # recognize speech using Wit.ai
        WIT_AI_KEY = "FA6O64UIZF5L32M2YNV77ZLOZDVOKMNB"  # Wit.ai keys are 32-character uppercase alphanumeric strings
        try:
            line[WIT] = digit_to_text(r.recognize_wit(audio, key=WIT_AI_KEY))
        except sr.UnknownValueError:
            line[WIT] = ""
            print("Wit.ai could not understand audio")
        except sr.RequestError as e:
            line[WIT] = ""
            print("Could not request results from Wit.ai service; {0}".format(e))
        if if_ATCO2(line[1]):
            hypothesis[ATCO2_ARRAY_INDEX][WIT_ARRAY].append(line[WIT])
        else:
            hypothesis[ATCOSIM_ARRAY_INDEX][WIT_ARRAY].append(line[WIT])

        print(line)
        tsv_writer.writerow(line)

    # wer = [[], [], [], []]

    testCsvFile.close()
    out_file.close()


def evaluate_recognized():
    recognized_file = open(recognitionResult, mode='r', encoding='utf-8')
    reader = csv.reader(recognized_file, delimiter="\t")

    for rownum, line in enumerate(reader):
        if line[1] == "path":
            continue

        if if_ATCO2(line[1]):
            t = re.sub("^[A-Za-z]+$", '', line[2].replace('  ', ' ').strip().lower())
            if t == 0:
                continue
            ground_truth[ATCO2_ARRAY_INDEX].append(t)

            hypothesis[ATCO2_ARRAY_INDEX][SPHINX_ARRAY].append(re.sub("^[A-Za-z]+$", '', line[SPHINX].replace('  ', ' ').strip().lower()))
            hypothesis[ATCO2_ARRAY_INDEX][GOOGLE_ARRAY].append(re.sub("^[A-Za-z]+$", '', line[GOOGLE].replace('  ', ' ').strip().lower()))
            hypothesis[ATCO2_ARRAY_INDEX][GOOGLE_CLOUD_ARRAY].append(re.sub("^[A-Za-z]+$", '', line[GOOGLE_CLOUD].replace('  ', ' ').strip().lower()))
            hypothesis[ATCO2_ARRAY_INDEX][WIT_ARRAY].append(re.sub("^[A-Za-z]+$", '', line[WIT].replace('  ', ' ').strip().lower()))
        else:
            t = re.sub("^[A-Za-z]+$", '', line[2].replace('  ', ' ').strip().lower())
            if not t:
                continue
            ground_truth[ATCOSIM_ARRAY_INDEX].append(t)

            hypothesis[ATCOSIM_ARRAY_INDEX][SPHINX_ARRAY].append(re.sub("^[A-Za-z]+$", '', line[SPHINX].replace('  ', ' ').strip().lower()))
            hypothesis[ATCOSIM_ARRAY_INDEX][GOOGLE_ARRAY].append(re.sub("^[A-Za-z]+$", '', line[GOOGLE].replace('  ', ' ').strip().lower()))
            hypothesis[ATCOSIM_ARRAY_INDEX][GOOGLE_CLOUD_ARRAY].append(re.sub("^[A-Za-z]+$", '', line[GOOGLE_CLOUD].replace('  ', ' ').strip().lower()))
            hypothesis[ATCOSIM_ARRAY_INDEX][WIT_ARRAY].append(re.sub("^[A-Za-z]+$", '', line[WIT].replace('  ', ' ').strip().lower()))

    evaluate()


# TODO: replace ten with zero
def digit_to_text(string):
    string = string.replace('1', 'one ')
    string = string.replace('2', 'two ')
    string = string.replace('3', 'three ')
    string = string.replace('4', 'four ')
    string = string.replace('5', 'five ')
    string = string.replace('6', 'six ')
    string = string.replace('7', 'seven ')
    string = string.replace('8', 'eight ')
    string = string.replace('9', 'nine ')
    string = string.replace('0', 'zero ')  # was written ten -> need to replace in files
    string = string.replace('  ', ' ')
    string = string.strip()
    return string


def if_ATCO2(string):
    return "LK" in string or "LZ" in string or "LS" in string or "LG" in string or "YSSY" in string


def evaluate():
   
    wer_atco2 = []
    wer_atcosim = []
    wil_atco2 = []
    wil_atcosim = []
    for index_d, dataset in enumerate(hypothesis):
        print(hypothesis[0][2][15])
        print(ground_truth[0][15])

        if index_d == 0:
            print('ATCO2:')
        else:
            print('ATCOSIM')
        for index_f, framework in enumerate(dataset):
            # try:
            wer = jiwer.wer(ground_truth[index_d], framework)
            mer = jiwer.mer(ground_truth[index_d], framework)
            wil = jiwer.wil(ground_truth[index_d], framework)
            cer = jiwer.cer(ground_truth[index_d], framework)
            wip = jiwer.wip(ground_truth[index_d], framework)
            # except Exception as e:
            #     print(ground_truth[index_d])
            #     print(framework)
            #     print(e)
            #     return

            if index_d == 0:
                wer_atco2.append(wer)
                wil_atco2.append(wil)
            else:
                wer_atcosim.append(wer)
                wil_atcosim.append(wil)


            print(FRAMEWORK_ARRAY[index_f])
            print('wer: {}'.format(wer))
            print('mer: {}'.format(mer))
            print('wil: {}'.format(wil))
            print('cer: {}'.format(cer))
            print('wip: {}'.format(wip))

    barWidth = 0.25
    # set heights of bars
    bars1 = wer_atco2
    bars2 = wil_atco2
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # Make the plot
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='WER', color="#D3D3D3")
    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='WIL', color="#808080")
    plt.xticks([r + barWidth for r in range(len(bars1))], ['CMU Sphinx', 'Google Speech \nRecognition', 'Google Cloud \nSpeech API', 'Wit.ai'], fontsize=10)
    # Create legend & Show graphic
    plt.legend(bbox_to_anchor=(0.45, 1.0), ncol=2)
    plt.title('ATCO2')
    plt.show()

    print('okvod')

    barWidth = 0.25
    # set heights of bars
    bars1 = wer_atcosim
    bars2 = wil_atcosim
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # Make the plot
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='WER', color="#D3D3D3")
    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='WIL', color="#808080")
    plt.xticks([r + barWidth for r in range(len(bars1))],
               ['CMU Sphinx', 'Google Speech \nRecognition', 'Google Cloud \nSpeech API', 'Wit.ai'], fontsize=10)
    # Create legend & Show graphic
    plt.legend(bbox_to_anchor=(0.45, 1.0), ncol=2)
    plt.title('ATCOSIM')
    plt.show()


    barWidth = 0.25
    # set heights of bars
    bars1 = wer_atcosim
    bars2 = wer_atco2
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # Make the plot
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='ATCOSIM', color="#D3D3D3")
    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='ATCO2', color="#808080")
    plt.xticks([r + barWidth for r in range(len(bars1))],
               ['CMU Sphinx', 'Google Speech \nRecognition', 'Google Cloud \nSpeech API', 'Wit.ai'], fontsize=10)
    # Create legend & Show graphic
    plt.legend(bbox_to_anchor=(0.45, 1.0), ncol=2)
    plt.title('WER comparison')
    plt.show()

    barWidth = 0.25
    # set heights of bars
    bars1 = wil_atcosim
    bars2 = wil_atco2
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # Make the plot
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='ATCOSIM', color="#D3D3D3")
    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='ATCO2', color="#808080")
    plt.xticks([r + barWidth for r in range(len(bars1))],
               ['CMU Sphinx', 'Google Speech \nRecognition', 'Google Cloud \nSpeech API', 'Wit.ai'], fontsize=10)
    # Create legend & Show graphic
    plt.legend(bbox_to_anchor=(0.8, 0.95), loc='center right', ncol=2)
    plt.title('WIL comparison')
    plt.show()


def plotDFT():
    fs_rate, signal = wavfile.read(clipsDirectory + "/sm2_07_090.wav")
    print("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    print("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print("secs", secs)
    Ts = 1.0 / fs_rate  # sampling interval in time
    print("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts)  # time vector as scipy arange field / numpy.ndarray
    FFT = abs(fft(signal))
    FFT_side = FFT[range(N // 2)]  # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1] - t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N // 2)]  # one side frequency range
    fft_freqs_side = np.array(freqs_side)
    plt.subplot(211)
    p1 = plt.plot(t, signal, "g")  # plotting the signal
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(212)

    p3 = plt.plot(freqs_side, abs(FFT_side), "b")  # plotting the positive fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.subplots_adjust(left=0.2,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.5,
                        hspace=0.4)
    plt.show()

def plotFremeworksWerMean():
    data = [0.0, 0.18, 0.0, 0.07, 0.0, 0.17, 0.0, 0.0, 0.0, 0.54, 0.0, 0.0, 0.14, 0.38, 0.0, 0.22, 0.0, 0.0]
    print("Gogle API WER mean by Këpuska, V.: {}".format(statistics.median(data)))
    fig = plt.figure(figsize=(10, 7))
    # Creating plot
    plt.boxplot(data)
    # show plot
    plt.title('WER Google API')
    plt.show()

    print("Gogle API WER mean by Këpuska, V.: {}".format(statistics.median(data)))


    data = [0.25, 0.36, 0.22, 0.29, 0.67, 0.58, 0.33, 0.07, 0.11, 0.29, 0.62, 0.55, 0.71, 0.43, 0.5, 0.11, 0.67, 0.17, 0.14]
    print("CMU Sphinx WER mean by Këpuska, V.: {}".format(statistics.median(data)))
    fig = plt.figure(figsize=(10, 7))
    # Creating plot
    plt.boxplot(data)
    # show plot
    plt.title('CMU Sphinx')
    plt.show()

    # print("Gogle API WER mean by Këpuska, V.: {}".format(statistics.median(data)))

