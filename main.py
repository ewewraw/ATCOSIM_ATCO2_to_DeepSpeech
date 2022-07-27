# This is a sample Python script.

import os
import shutil
import csv
import pandas as pd
from pydub import AudioSegment
import re
import wave
import contextlib
import ASR
import wav2vec
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

ATCOSIMdirectory = "/home/natalia/Downloads/ATCOSIM"
WAVdirectory = ATCOSIMdirectory + "/WAVdata"
destinationDirectory = "/home/natalia/bachelors/en/clips"
tsvFile = '/home/natalia/bachelors/en/output.tsv'
ATCO2DataDirectory = '/home/natalia/Downloads/ATCO2-ASRdataset-v1_beta/DATA'
destinationDirectory2 = "/home/natalia/bachelors/en2/clips"
tsvFile2 = '/home/natalia/bachelors/en2/output.tsv'
mergeDirectory = "/home/natalia/bachelors/merge/en/clips"
mergeDirectoryRoot = "/home/natalia/bachelors/merge"
cleaned_train = "/home/natalia/bachelors/data_ready_for_training/train.tsv"
cleaned_dev = "/home/natalia/bachelors/data_ready_for_training/dev.tsv"
cleaned_test = "/home/natalia/bachelors/data_ready_for_training/test.tsv"


def wav_to_mp3():
    for filename in os.listdir(mergeDirectory):
        if ".wav" in filename:
            f = os.path.join(mergeDirectory, filename)
            mpF = f.replace(".wav", ".mp3")
            AudioSegment.from_wav(f).export(mpF, format="mp3")
            os.remove(f)


def delete_wav_from_filename_in_csv():
    tsvFile = open(mergeDirectoryRoot + "/output.tsv", mode='r', encoding='utf-8')
    tsvFileCleared = open(mergeDirectoryRoot + "/output_cleared.tsv", mode='wt', encoding='utf-8')

    reader = csv.reader(tsvFile, delimiter="\t")
    writer = csv.writer(tsvFileCleared, delimiter='\t')

    for rownum, line in enumerate(reader):
        line[1] = line[1].replace('.wav', '')
        writer.writerow(line)

    tsvFile.close()
    tsvFileCleared.close()


def split_data_set():
    dataset_tsv_name = mergeDirectoryRoot + "/output_cleared.tsv"
    dataset_tsv = open(dataset_tsv_name, mode='r', encoding='utf-8')

    train_tsv = open(mergeDirectoryRoot + "/train.tsv", mode='wt', encoding='utf-8')
    dev_tsv = open(mergeDirectoryRoot + "/dev.tsv", mode='wt', encoding='utf-8')
    test_tsv = open(mergeDirectoryRoot + "/test.tsv", mode='wt', encoding='utf-8')

    train_tsv_writer = csv.writer(train_tsv, delimiter='\t')
    dev_tsv_writer = csv.writer(dev_tsv, delimiter='\t')
    test_tsv_writer = csv.writer(test_tsv, delimiter='\t')

    reader = csv.reader(dataset_tsv, delimiter="\t")
    deleted = 0
    deletedATCOSIM = 0
    remainingATCOSIM = 0
    remainingATCOSIM_duration = 0
    deletedATCOSIM_duration = 0

    deletedATCO2 = 0
    remainingATCO2 = 0
    remainingATCO2_duration = 0
    deletedATCO2_duration = 0

    for rownum, line in enumerate(reader):
        if line[1] != "path":
            try:
                fname = destinationDirectory + "/" + line[1] + ".wav"
                with contextlib.closing(wave.open(fname, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
            except FileNotFoundError:
                print("not found {}".format(line[1]))
        else:
            duration = 0

        if "[EMPTY]" in line[2] or "[FRAGMENT]" in line[2] or "<FL>" in line[2] or "</FL>" in line[2] \
                or "<OT>" in line[2] or "[HNOISE]" in line[2] or "[UNKNOWN]" in line[2] or "[NONSENSE]" in line[2]\
                or "~" in line[2] or "@" in line[2] or "=" in line[2] or "_" in line[2] or "<" in line[2]:
            line[2] = line[2].replace('[EMPTY]', '')
            line[2] = line[2].replace('[FRAGMENT]', '')
            line[2] = line[2].replace('[UNKNOWN]', '')
            line[2] = line[2].replace('[HNOISE]', '')
            line[2] = line[2].replace('[NONSENSE]', '')
            line[2] = line[2].replace('<FL>', '')
            line[2] = line[2].replace('</FL>', '')
            line[2] = line[2].replace('<OT>', '')
            line[2] = line[2].replace('</OT>', '')
            line[2] = line[2].replace('~', '')
            line[2] = line[2].replace('@', '')
            line[2] = line[2].replace('=', '')
            line[2] = line[2].replace('_', ' ')
            line[2] = re.sub('<.*?>', '', line[2])

            if not line[2] or line[2] == '' or line[2].isspace():
                deleted = deleted + 1
                if "LK" in line[1] or "LZ" in line[1] or "LS" in line[1] or "LG" in line[1] or "YSSY" in line[1]:
                    deletedATCO2 = deletedATCO2 + 1
                    deletedATCO2_duration = deletedATCO2_duration + duration
                else:
                    deletedATCOSIM = deletedATCOSIM + 1
                    deletedATCOSIM_duration = deletedATCOSIM_duration + duration
                continue



        if "LK" in line[1] or "LZ" in line[1] or "LS" in line[1] or "LG" in line[1] or "YSSY" in line[1]:
            remainingATCO2 = remainingATCO2 + 1
            remainingATCO2_duration = remainingATCO2_duration + duration
        else:
            remainingATCOSIM = remainingATCOSIM + 1
            remainingATCOSIM_duration = remainingATCOSIM_duration + duration




        if rownum == 0:
            train_tsv_writer.writerow(line)
            dev_tsv_writer.writerow(line)
            test_tsv_writer.writerow(line)
            continue
        if rownum % 10 == 0:  # add to test
            test_tsv_writer.writerow(line)
            continue

        if rownum % 10 == 2 or rownum % 10 == 4 or rownum % 10 == 6:  # add to dev
            dev_tsv_writer.writerow(line)
            continue

        train_tsv_writer.writerow(line)

    print(deleted)
    print("deleted ATCOSIM = {}".format(deletedATCOSIM))
    print("remining ATCOSIM = {}".format(remainingATCOSIM))
    print("deleted ATCOSIM duration = {}".format(deletedATCOSIM_duration))
    print("remining ATCOSIM duration = {}".format(remainingATCOSIM_duration))

    print("deleted ATCO2 = {}".format(deletedATCO2))
    print("remainig ATCO2 = {}".format(remainingATCO2))
    print("deleted ATCO2 duration = {}".format(deletedATCO2_duration))
    print("remainig ATCO2 duration = {}".format(remainingATCO2_duration))
    dataset_tsv.close()
    train_tsv.close()
    dev_tsv.close()
    test_tsv.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('hehe')
    wav2vec.fine_tune();

    # wav_to_mp3()
    # delete_wav_from_filename_in_csv()
    # split_data_set()
    # ASR.calculate_WER()
    # ASR.evaluate_recognized()
    # ASR.plotDFT()
    # ASR.plotFremeworksWerMean()

    # with open(tsvFile, 'wt') as out_file: # processing ATCOSIM data
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     tsv_writer.writerow(['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accents', 'locale', 'segment'])
    #
    #     for airportName in os.listdir(WAVdirectory): # for each airport
    #         currentAirportDirectory = WAVdirectory + "/" + airportName
    #         for directoryName in os.listdir(currentAirportDirectory): # for each dataset of airport
    #             directoryWithDataPath = currentAirportDirectory + "/" + directoryName
    #             for filename in os.listdir(directoryWithDataPath): # for each file
    #                 f = os.path.join(directoryWithDataPath, filename)
    #                 # checking if it is a file
    #                 if os.path.isfile(f):
    #
    #                     currentSentencePath = f.replace('WAVdata', 'TXTdata')
    #                     currentSentencePath = currentSentencePath.replace('.wav', '.txt')
    #                     # print(f)
    #
    #                     with open(currentSentencePath) as f:
    #                         currentSentence = f.readlines()
    #
    #                     # print(currentSentencePath)
    #                     # print(currentSentence)
    #                     currentSentence = currentSentence[0]
    #
    #                     tsv_writer.writerow(
    #                         ['xxx-id', filename, currentSentence, '', '', '', '', '',
    #                          'en', ''])
    #

    # with open(tsvFile2, 'wt') as out_file:  # processing ATCOSIM data
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     tsv_writer.writerow(
    #         ['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accents', 'locale',
    #          'segment'])
    #
    #     for filename in os.listdir(ATCO2DataDirectory):
    #         f = os.path.join(ATCO2DataDirectory, filename)
    #         # checking if it is a file
    #         if os.path.isfile(f):
    #             # resultSentence = ""
    #             # currentFile = ""
    #
    #             # if '.wav' in filename: # copy wav data to new directory
    #             #     currentFile = filename
    #             #     # shutil.copy(f, destinationDirectory2)
    #             #     continue
    #
    #             if '.cnet' in filename:
    #                 file = open(f, 'r')
    #                 lines = file.readlines()
    #
    #                 resultSentence = ""
    #                 count = 0
    #                 for line in lines:
    #                     count += 1
    #                     array = line.strip().split(' ')
    #                     if array[4] != "<eps>":
    #                         resultSentence = resultSentence + " " + array[4]
    #                 # print(resultSentence.strip())
    #                 currentFile = filename.replace('.cnet', '.wav')
    #                 fWav = os.path.join(ATCO2DataDirectory, currentFile)
    #                 print(fWav)
    #
    #
    #                 shutil.copy(fWav, destinationDirectory2)
    #                 tsv_writer.writerow(
    #                     ['xxx-id', currentFile, resultSentence, '', '', '', '', '',
    #                      'en', ''])

    # tsv1 = pd.read_csv(tsvFile, sep='\t')
    # tsv2 = pd.read_csv(tsvFile2, sep='\t')
    #
    # for filename in os.listdir(destinationDirectory):
    #     f = os.path.join(destinationDirectory, filename)
    #     shutil.copy(f, mergeDirectory)
    # for filename in os.listdir(destinationDirectory2):
    #     f = os.path.join(destinationDirectory2, filename)
    #     shutil.copy(f, mergeDirectory)
    #
    # combined_csv = pd.concat([tsv1, tsv2])
    # combined_csv.to_csv(mergeDirectoryRoot + "/output.tsv", sep="\t", header=True, index=False)
