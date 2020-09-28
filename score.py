#!/usr/bin/env python3
## score.py -- calculate edit distance between label and prediction files
##
## Copyright (C) 2019, Tom Dörr <tom.doerr@tum.de>, 
## Karla Markert <karla.markert@aisec.fraunhofer.de>, 
## Nicolas Müller <nicolas.mueller@aisec.fraunhofer.de>,
## Konstantin Böttinger <konstantin.boettinger@aisec.fraunhofer.de> 
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import Levenshtein
import sys
import os
import re

PATH_OUTPUT_AUDIO_TEST = './' + sys.argv[1] + '/'


num_editops_list = []
num_normal_editops_list = []
num_wav_files = 0
dir_dict = {}

def calculate_edit_distance(path_output_audio_test, prediction_file_ending='prediction', max_number_points_to_plot=-1):
    num_editops_list = []
    num_normal_editops_list = []
    num_wav_files = 0
    dir_dict = {}
    for e in os.listdir(path_output_audio_test):
        if '.wav' in e:
            num_wav_files += 1
        if prediction_file_ending in e:
            all_ids_file = ''.join(c for c in e if c.isdigit())
            dir_dict[all_ids_file] = e

    for e in sorted(list(dir_dict), key=int)[:max_number_points_to_plot]:
        if prediction_file_ending in dir_dict[e]:
            filename_prediction = dir_dict[e]
            filename_label = filename_prediction.replace(prediction_file_ending, '_label')
            with open(path_output_audio_test + filename_prediction, 'r') as f:
                prediction_raw = f.read()
            if len(sys.argv) > 2 and sys.argv[2] == '--except_predict_search':
                try:
                    prediction = re.search('(\w(\ |\w|\')*\w)', prediction_raw).group(1)
                except Exception:
                    pass
            else:
                prediction = re.search('(\w(\ |\w|\')*\w)', prediction_raw).group(1)

            with open(path_output_audio_test + filename_label, 'r') as f:
                label_raw = f.read()
            label = re.search('(\w(\ |\w|\')*\w)', label_raw).group(1).replace('_', ' ')

            editops = Levenshtein.editops(prediction, label)
            print()
            print(filename_prediction.replace(prediction_file_ending, ''))
            print(prediction_file_ending + ': ' + prediction)
            print('label: ' + label)
            print('editops: ' + str(editops))
            print('Normal Levenshtein distance: ' + str(len(editops)))
            num_normal_editops_list.append(len(editops))
            num_editops = 0
            for e in editops:
                num_editops += 1
                if e[0] == 'replace':
                    num_editops += 1

            print('insert-delete-editdistance (num_editops): ' + str(num_editops))
            num_editops_list.append(num_editops)

    avg_num_editops = (sum(num_editops_list)/len(num_editops_list))
    avg_num_normal_editops = (sum(num_normal_editops_list)/len(num_normal_editops_list))
    print('\n\n')
    print('Number of .wav files: ' + str(num_wav_files))
    print('Average number insert delete editops: **{:.2f}**'.format(avg_num_editops))
    print('Average Levenshtein editops: **{:.2f}**'.format(avg_num_normal_editops))
    return num_normal_editops_list


if __name__ == '__main__':
    calculate_edit_distance(PATH_OUTPUT_AUDIO_TEST)









