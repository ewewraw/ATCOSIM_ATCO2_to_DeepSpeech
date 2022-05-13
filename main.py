# This is a sample Python script.

import os
import shutil
import csv
import pandas as pd


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ATCOSIMdirectory = "/home/natalia/Downloads/ATCOSIM"
    WAVdirectory = ATCOSIMdirectory + "/WAVdata"
    destinationDirectory = "/home/natalia/bachelors/en/clips"
    tsvFile = '/home/natalia/bachelors/en/output.tsv'

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
    ATCO2DataDirectory = '/home/natalia/Downloads/ATCO2-ASRdataset-v1_beta/DATA'
    destinationDirectory2 = "/home/natalia/bachelors/en2/clips"
    tsvFile2 = '/home/natalia/bachelors/en2/output.tsv'

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

    mergeDirectory = "/home/natalia/bachelors/merge/clips"
    mergeDirectoryRoot = "/home/natalia/bachelors/merge"
    tsv1 = pd.read_csv(tsvFile, sep='\t')
    tsv2 = pd.read_csv(tsvFile2, sep='\t')

    for filename in os.listdir(destinationDirectory):
        f = os.path.join(destinationDirectory, filename)
        shutil.copy(f, mergeDirectory)
    for filename in os.listdir(destinationDirectory2):
        f = os.path.join(destinationDirectory2, filename)
        shutil.copy(f, mergeDirectory)

    # Output_df = pd.merge(tsv1, tsv2)
    combined_csv = pd.concat([tsv1, tsv2])
    # export to csv
    # combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
    combined_csv.to_csv(mergeDirectoryRoot + "/output.tsv", sep="\t", header=True, index=False)
