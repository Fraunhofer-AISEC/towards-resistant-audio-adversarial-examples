#!/usr/bin/env python3
## add_audio_to_start.py -- add offsets to audio file
##
## Copyright (C) 2019, Tom Dörr <tom.doerr@tum.de>, 
## Karla Markert <karla.markert@aisec.fraunhofer.de>, 
## Nicolas Müller <nicolas.mueller@aisec.fraunhofer.de>,
## Konstantin Böttinger <konstantin.boettinger@aisec.fraunhofer.de> 
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import scipy.io.wavfile as wav
import sys
import os
import numpy as np
import shutil
import argparse

OFFSET_DIR = 'offset_audio_added'
SAMPLES_TO_ADD_IF_APPENDING=960

parser = argparse.ArgumentParser()
parser.add_argument('audio_paths', nargs='*',
                    help='Paths to audio files (separated by spaces)')
parser.add_argument('--bin', default=False, action='store_true', 
                    help='Add offsets up to the distance between two Deepspeech bins')
parser.add_argument('--append', default=False, action='store_true',
                    help='Add padding at the end of the audio file')
parser.add_argument('--max_offset', type=int, default=800,
                    help='Specifies the maximum offset added')
parser.add_argument('--max_samples_append', type=int, default=960,
                    help='Specifies the maximum padding added to the end of the audio file')
args = parser.parse_args()

for audio_path in args.audio_paths:
    path_label_file = audio_path.replace('.wav', '_label')
    destination_folder = OFFSET_DIR + '/' + audio_path.replace('.wav', '') + '/'

    if args.append and args.max_offset > args.max_samples_append:
        raise ValueError('''max_offset can not be larger than max_samples_append.
                Please specify a value for --max_samples_append > --max_offset.''')

    os.makedirs(destination_folder, 0o755, exist_ok=True)
    _, audio = wav.read(audio_path)

    append_audio = False

    if args.bin:
        if args.max_offset != 800:
            raise ValueError('Can not specify max_offsset with --bin option')
        offset_distance = 320
    else:
        offset_distance = args.max_offset

    if args.append:
        append_audio = True

    for i in range(offset_distance):
        print()
        print('i: ' + str(i))
        print('length original audio: ' + str(audio.shape))
        output_audio = np.concatenate((np.zeros(i, dtype=np.int16), audio))
        print('lengh audio with offset at the start: ' + str(output_audio.shape))
        if append_audio:
            output_audio = np.concatenate((output_audio, np.zeros(args.max_samples_append-i, dtype=np.int16)))
            print('lenght audio with offset at the start and end: ' + str(output_audio.shape))
        wav.write(destination_folder + str(i) + '.wav', 16000, output_audio)
        try:
            shutil.copy2(path_label_file, destination_folder + str(i) + '_label')
        except Exception as e:
            print(e)






