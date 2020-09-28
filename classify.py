#! /usr/bin/env python3
## classify.py -- actually classify a sequence with DeepSpeech
##
## Copyright (C) 2019, Tom Dörr <tom.doerr@tum.de>, 
## Karla Markert <karla.markert@aisec.fraunhofer.de>, 
## Nicolas Müller <nicolas.mueller@aisec.fraunhofer.de>,
## Konstantin Böttinger <konstantin.boettinger@aisec.fraunhofer.de> 
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
import argparse

import scipy.io.wavfile as wav

import time
import os
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")
import DeepSpeech

os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import pydub
    import struct
except:
    print("pydub was not loaded, MP3 compression will not work")

from tf_logits import get_logits


# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"



def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('input_files', type=str,
                        nargs='+',
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    args = parser.parse_args()
    restore_path='deepspeech-0.4.1-checkpoint/model.v0.4.1'
    for input_file in args.input_files:
        tf.reset_default_graph()
        with tf.Session() as sess:
            if input_file.split(".")[-1] == 'mp3':
                raw = pydub.AudioSegment.from_mp3(input_file)
                audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
            elif input_file.split(".")[-1] == 'wav':
                _, audio = wav.read(input_file)
            else:
                raise Exception("Unknown file format")
            prediction_output_path = input_file.split('.')[0] + '_041_prediction'
            N = len(audio)
            new_input = tf.placeholder(tf.float32, [1, N])
            lengths = tf.placeholder(tf.int32, [1])

            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                logits = get_logits(new_input, lengths)

            saver = tf.train.Saver()
            saver.restore(sess, restore_path)

            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)

            print('logits shape', logits.shape)
            length = (len(audio)-1)//320
            l = len(audio)
            r = sess.run(decoded, {new_input: [audio],
                                   lengths: [length]})
            prediction =  "".join([toks[x] for x in r[0].values])

            print("-"*80)
            print("-"*80)

            print("Classification:")
            print(prediction)
            print("-"*80)
            print("-"*80)

            with open(prediction_output_path, 'w') as f:
                f.write(prediction)

main()
