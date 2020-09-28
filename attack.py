#! /usr/bin/env python3
## attack.py -- generate audio adversarial examples
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
from shutil import copyfile

import scipy.io.wavfile as wav

import Levenshtein
import struct
import time
import os
import sys
from collections import namedtuple
from tensorflow.core.framework import summary_pb2
import random
sys.path.append("DeepSpeech")

try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")

try:
    import ipdb
except:
    pass

import DeepSpeech

from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

def convert_mp3(new, lengths):
    import pydub
    wav.write("/tmp/load.wav", 16000,
              np.array(np.clip(np.round(new[0][:lengths[0]]),
                               -2**15, 2**15-1),dtype=np.int16))
    pydub.AudioSegment.from_wav("/tmp/load.wav").export("/tmp/saved.mp3")
    raw = pydub.AudioSegment.from_mp3("/tmp/saved.mp3")
    mp3ed = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])[np.newaxis,:lengths[0]]
    return mp3ed
    

class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=5000, batch_size=1, max_offset=320,
                 mp3=False, l2penalty=float('inf'), restore_path=None, adversarial_signal_limit=2000.0):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """
        
        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        self.mp3 = mp3
        self.max_offset = max_offset
        self.adversarial_signal_limit = adversarial_signal_limit

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_phrase_lengths')
        self.learning_rate_tensor = tf.Variable(np.ones((1), dtype=np.float32), name='qq_learning_rate_tensor')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -adversarial_signal_limit, adversarial_signal_limit)*self.rescale

        # We set the new input to the model to be the above delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = self.apply_delta*mask + original

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape,
                                 stddev=2)
        pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in, lengths)

        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

        # Choose the loss function we want -- either CTC or CW
        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths)
            
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)

            # Slight hack: an infinite l2 penalty means that we don't penalize l2 distortion
            # The code runs faster at a slight cost of distortion, and also leaves one less
            # paramaeter that requires tuning.
            if not np.isinf(l2penalty):
                loss = tf.reduce_mean((self.new_input-self.original)**2,axis=1) + l2penalty*ctcloss
            else:
                loss = ctcloss
            self.expanded_loss = tf.constant(0)
            
        elif loss_fn == "CW":
            raise NotImplemented("The current version of this project does not include the CW loss function implementation.")
        else:
            raise

        self.loss = loss
        self.ctcloss = ctcloss

        # Set up the Adam optimizer to perform gradient descent for us
        start_vars = set(x.name for x in tf.global_variables())
        tf.summary.scalar('Learning Rate', self.learning_rate_tensor[0])
        optimizer = tf.train.AdamOptimizer(self.learning_rate_tensor[0])
        self.optimizer = optimizer

        grad,var = optimizer.compute_gradients(self.loss, [delta])[0]
        self.grad_sign = grad_sign = tf.sign(grad)
        self.train = optimizer.apply_gradients([(grad_sign,var)])

        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        sess.run(tf.variables_initializer(new_vars+[delta]))

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

        self.merged = tf.summary.merge_all()

    def attack(self, audio, lengths, target, summary_writer, finetune=None, rescale_constant=0.8, enable_random_offset=False,
            succ_iter_till_reduce=100):
        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths)-1)//320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths)-1)//320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))
        sess.run(self.learning_rate_tensor.assign([self.learning_rate]))
        init_local = tf.local_variables_initializer()
        sess.run(init_local)

        self.random_offset_int_list = []
        levenshtein_distance_list = []
        levenshtein_mean_iterations = []
        adversarial_signal_limit = self.adversarial_signal_limit
        bound = adversarial_signal_limit

        old_random_offsets_list = [0] * self.batch_size
        # Here we'll keep track of the best solution we've found so far
        best_solution = [None]*self.batch_size

        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune-audio))
        
        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        timestamp_last_iteration = time.time()
        MAX = self.num_iterations
        for i in range(MAX):
            iteration = i
            now = time.time()


            # Print out some debug information every 10 iterations.
            if i%10 == 0:

                new, delta, r_out, r_logits = sess.run((self.new_input, self.delta, self.decoded, self.logits))
                lst = [(r_out, r_logits)]
                if self.mp3:
                    mp3ed = convert_mp3(new, lengths)
                    
                    mp3_out, mp3_logits = sess.run((self.decoded, self.logits),
                                                   {self.new_input: mp3ed})
                    lst.append((mp3_out, mp3_logits))

                for out, logits in lst:
                    chars = out[0].values

                    res = np.zeros(out[0].dense_shape)+len(toks)-1
                
                    for ii in range(len(out[0].values)):
                        x,y = out[0].indices[ii]
                        res[x,y] = out[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    print("\n".join(res))
                    
                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))

                    for res_e in res:

                        phrase = "".join([toks[x] for x in target[0]])
                        levenshtein_distance = Levenshtein.distance(res_e, phrase)
                        levenshtein_distance_list.append(levenshtein_distance)
                print("levenshtein_distances: " + str(levenshtein_distance_list))
                levenshtein_distance_mean = sum(levenshtein_distance_list) / len(levenshtein_distance_list)
                levenshtein_mean_iterations.append((i, levenshtein_distance_mean))
                print("levenshtein_distance_mean: " + str(levenshtein_distance_mean))
                levenshtein_distance_list = []

                sum_val = summary_pb2.Summary.Value(tag='Levenshtein Distance', simple_value=levenshtein_distance_mean)
                levenshtein_summary = summary_pb2.Summary(value=[sum_val])

            if self.mp3:
                new = sess.run(self.new_input)
                mp3ed = convert_mp3(new, lengths)
                feed_dict = {self.new_input: mp3ed}
            else:
                feed_dict = {}
                
            # Check how long the iterations take
            seconds_since_last_iterations = time.time() - timestamp_last_iteration
            sum_val = summary_pb2.Summary.Value(tag='Iteration Time', simple_value=seconds_since_last_iterations)
            iteration_time_summary = summary_pb2.Summary(value=[sum_val])
            print('Iter: {},  Iter time: {:3.2f}s'.format(i, seconds_since_last_iterations))
            timestamp_last_iteration = time.time()
            if enable_random_offset:
                # Add new offset to original audio
                original = sess.run(self.original)
                random_offsets_list = self.generate_random_value_list(self.batch_size, 0, self.max_offset)
                original_new_offsets = self.new_random_offset(original, old_random_offsets_list, random_offsets_list) 
                self.original.load(original_new_offsets, sess)

                # Add new offset to adversarial signals
                delta = sess.run(self.delta)
                delta_without_offsets = self.remove_offset(delta, old_random_offsets_list)
                delta_mean = np.mean(delta_without_offsets, axis=0)
                delta_tile = np.tile(np.expand_dims(delta_mean, axis=0), [self.batch_size, 1])
                delta_with_offsets = self.add_offset(delta_tile, random_offsets_list)
                self.delta.load(delta_with_offsets, sess)

                old_random_offsets_list = random_offsets_list



            # Actually do the optimization step
            m, d, el, cl, l, logits, new_input, _ = sess.run((self.merged, self.delta, self.expanded_loss,
                                                           self.ctcloss, self.loss,
                                                           self.logits, self.new_input,
                                                           self.train),
                                                          feed_dict)
                    

            sum_val = summary_pb2.Summary.Value(tag='Loss', simple_value=np.mean(cl))
            loss_mean = summary_pb2.Summary(value=[sum_val])

            # Log metrics for tensorboard
            summary_writer.add_summary(m, i)
            summary_writer.add_summary(levenshtein_summary, i)
            summary_writer.add_summary(loss_mean, i)
            summary_writer.add_summary(iteration_time_summary, i)

            # Report progress
            print("%.3f"%np.mean(cl), "\t", "\t".join("%.3f"%x for x in cl))

            logits = np.argmax(logits,axis=2).T
            for ii in range(self.batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                if (self.loss_fn == "CTC" and i%10 == 0 and res[ii] == "".join([toks[x] for x in target[ii]])) \
                   or (i == MAX-1 and best_solution[ii] is None):
                    # Get the current constant
                    sum_edit_distance_mean = sum([e[1] for e in levenshtein_mean_iterations[-succ_iter_till_reduce:]])
                    rescale = sess.run(self.rescale)
                    if not enable_random_offset or sum_edit_distance_mean == 0 or (i == MAX-1 and best_solution[ii] is None):
                        if rescale[ii]*adversarial_signal_limit > np.max(np.abs(d)):
                            # If we're already below the threshold, then
                            # just reduce the threshold to the current
                            # point and save some time.
                            print("It's way over", np.max(np.abs(d[ii]))/adversarial_signal_limit)
                            rescale[ii] = np.max(np.abs(d[ii]))/adversarial_signal_limit
                        # Otherwise reduce it by some constant. The closer
                        # this number is to 1, the better quality the result
                        # will be. The smaller, the quicker we'll converge
                        # on a result but it will be lower quality.
                        rescale[ii] *= rescale_constant
                        best_solution[ii] = new_input[ii]

                    # Adjust the best solution found so far

                    bound = adversarial_signal_limit*rescale[ii][0]
                    print("Worked i=%d ctcloss=%f bound=%f"%(ii,cl[ii], bound))
                    sum_val = summary_pb2.Summary.Value(tag='Bound', simple_value=bound)
                    bound_summary = summary_pb2.Summary(value=[sum_val])
                    summary_writer.add_summary(bound_summary, i)
                    self.rescale.load(rescale, sess)

                    # Just for debugging, save the adversarial example
                    # to /tmp so we can see it if we want
                    wav.write("/tmp/adv.wav", 16000,
                              np.array(np.clip(np.round(new_input[ii]),
                                               -2**15, 2**15-1),dtype=np.int16))

        # l: loss
        # levenshtein_mean_iterations: mean edit distance of the last succ_iter_till_reduce
        #   times we calculated the edit distance
        return best_solution, np.mean(l), levenshtein_mean_iterations
    

    def generate_random_value_list(self, size, minimum, maximum):
        return_list = [0]*size
        for i in range(size):
            return_list[i] = self.get_new_random_value()
        return return_list


    def get_new_random_value(self):
        random_offset_int_list = self.random_offset_int_list
        if len(random_offset_int_list) == self.max_offset:
            random_offset_int_list = []

        while True:
            random_offset_int = random.randint(0, self.max_offset - 1)
            if random_offset_int not in random_offset_int_list:
                random_offset_int_list.append(random_offset_int)
                break
        return random_offset_int

    def load_value_array(self, variables, values):
        for i, variable in enumerate(variables):
            variable.load(values[i], self.sess)


    def new_random_offset(self, t, old_offsets, new_offsets):
        for i, e in enumerate(t):
            audio_no_padding = e[old_offsets[i]:(old_offsets[i] + t.shape[1] - self.max_offset)]
            zeros_to_append = np.zeros((self.max_offset - new_offsets[i])) 
            if new_offsets[i] != 0:
                t[i] = np.concatenate((
                np.zeros((new_offsets[i]), dtype=int),
                audio_no_padding,
                zeros_to_append
                ), axis=0)
            else:
                t[i] = np.concatenate((
                audio_no_padding,
                zeros_to_append
                ), axis=0)

        return t

    def remove_offset(self, t, old_offsets):
        return self.new_random_offset(t, old_offsets, [0] * t.shape[0])

    def add_offset(self, t, new_offsets):
        return self.new_random_offset(t, [0] * t.shape[0], new_offsets)



    
def main():
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input",
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    parser.add_argument('--target', type=str,
                        required=False,
                        help="Target transcription")
    parser.add_argument('--out', type=str,
                        required=False,
                        help="Path for the adversarial example(s)")
    parser.add_argument('--finetune', type=str,
                        required=False,
                        help="Initial .wav file(s) to use as a starting point")
    parser.add_argument('--lr', type=int,
                        required=False, default=100,
                        help="Learning rate for optimization")
    parser.add_argument('--iterations', type=int,
                        required=False, default=1000,
                        help="Maximum number of iterations of gradient descent")
    parser.add_argument('--l2penalty', type=float,
                        required=False, default=float('inf'),
                        help="Weight for l2 penalty on loss function")
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    parser.add_argument('--enable_random_offset', action='store_true', default=False,
                        help='Enables random offset during training')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for random offset training')
    parser.add_argument('--max_offset', type=int, default=320, 
                        help='Maximum offset used when --enable_random_offset is set')
    parser.add_argument('--succ_iter_till_reduce', type=int, default=100, 
                        help='For offset-training: Number of times we need to hit the target phrase \
                                until we reduce the adversarial signal limit')
    parser.add_argument('--rescale_constant', type=float, dest="rescale_constant",
                        required=False, default=0.8,
                        help="Constant used to rescal distortion level.")
    parser.add_argument('--adversarial_signal_limit', type=float, dest="adversarial_signal_limit",
                        required=False, default=2000.0,
                        help="Maximum intensity of adversarial audio signals")
    parser.add_argument('--target_label_file', type=str, default='',
                        help='File containing the targeted label, can be used as an alternative to --target')
    args = parser.parse_args()

    while len(sys.argv) > 1:
        sys.argv.pop()
    
    if args.target_label_file == '':
        try:
            args.target
            phrase = args.target
        except:
            raise Exception('''Either a target_label_file (--target_label_file xxx) or 
            a target phrase (--target "xxx") must be specified.''')
    else:
        with open(args.target_label_file, 'r') as f:
            phrase = f.read()

    phrase = phrase.rstrip()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter('tensorboard_logdir/' + args.out.replace('.wav', ''),
                flush_secs=10)

        finetune = []
        audios = []
        lengths = []

        assert args.out is not None
        
        # Load the inputs that we're given
        fs, audio = wav.read(args.input)
        assert fs == 16000
        assert audio.dtype == np.int16
        print('source dB', 20*np.log10(np.max(np.abs(audio))))
        audios.append(list(audio))
        lengths.append(len(audio))

        if args.finetune is not None:
            finetune.append(list(wav.read(args.finetune[i])[1]))



        if args.enable_random_offset:
            maxlen = max(map(len,audios)) + args.max_offset
            audios = np.array([x+[0]*args.max_offset+[0]*(maxlen-len(x)-args.max_offset) for x in audios])
        else:
            maxlen = max(map(len,audios))
            audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])

        if args.enable_random_offset:
            audio = audios[0]
            audios = []
            lengths = []
            for i in range(args.batch_size):
                audios.append(audio)
                lengths.append(len(audio))

        # Set up the attack class and run it
        attack = Attack(sess, 'CTC', len(phrase), maxlen,
                        batch_size=len(audios),
                        learning_rate=args.lr,
                        num_iterations=args.iterations,
                        l2penalty=args.l2penalty,
                        restore_path=args.restore_path,
                        max_offset=args.max_offset,
                        adversarial_signal_limit=args.adversarial_signal_limit)
        best_solution, loss, levenshtein_mean_iterations = attack.attack(audios,
                               lengths,
                               [[toks.index(x) for x in phrase]]*len(audios),
                               summary_writer,
                               finetune,
                               rescale_constant=args.rescale_constant,
                               succ_iter_till_reduce=args.succ_iter_till_reduce,
                               enable_random_offset=args.enable_random_offset)

        # And now save it to the desired output
        path = args.out

    i=0
    wav.write(path, 16000,
                  np.array(np.clip(np.round(best_solution[i][:lengths[i]]),
                                   -2**15, 2**15-1),dtype=np.int16))

    path_label_file = path.replace('.wav', '_label')
    with open(path_label_file, 'w') as f:
        f.write(phrase)

if __name__ == '__main__':
    main()
