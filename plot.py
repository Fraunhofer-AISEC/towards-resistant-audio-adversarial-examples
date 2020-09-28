#!/usr/bin/env python3
## plot.py -- plot edit-distances for different offsets
##
## Copyright (C) 2019, Tom Dörr <tom.doerr@tum.de>, 
## Karla Markert <karla.markert@aisec.fraunhofer.de>, 
## Nicolas Müller <nicolas.mueller@aisec.fraunhofer.de>,
## Konstantin Böttinger <konstantin.boettinger@aisec.fraunhofer.de> 
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import score
import sys
import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
import numpy
import argparse

plotting_multiple_samples = False

parser = argparse.ArgumentParser()
parser.add_argument('paths', nargs='+',
        help='Path(s) to folder(s) containing the audio predictions and labels')
parser.add_argument('--style_info', type=str, default='+',
        help='Syle info for matplotlib')
parser.add_argument('--prediction_ending', type=str, required=False,
        default='_041_prediction', help='File ending of the prediction file')
parser.add_argument('--save_as', default='',
        help='Output path for plot')
parser.add_argument('--x_label', type=str, default='Added offset in samples', 
        help='Label for x-axis')
parser.add_argument('--y_label', type=str, default='Edit distance',
        help='Label for y-axis')
parser.add_argument('--max_offset', type=int, default=-1,
        help='Maximum offset value to include in plots')
parser.add_argument('--in_one_plot', action='store_true', default=False,
        help='Plot all plots in one figure')




def output_plot():
    if args.save_as != '':
        plt.savefig(args.save_as, dpi=300)
    else:
        plt.show()


args = parser.parse_args()
print(args)

data_path = args.paths[0]
if args.in_one_plot:
    fig, ax = plt.subplots()

for data_path in args.paths:
    if data_path[-1] != '/':
        data_path += '/'

    try:
        with open(data_path + 'title', 'r') as f:
            title = f.read()
    except:
        title = ''

    files_in_dir = os.listdir(data_path)
    if [e for e in files_in_dir if '.wav' in e] != []:
        data_path_list = [data_path]
    else:
        print('Plotting multiple audio samples')
        data_path_list = [data_path + e + '/' for e in files_in_dir if e != 'title']


    if not args.in_one_plot:
        fig, ax = plt.subplots()

    for data_path_element in data_path_list:
        score_values = score.calculate_edit_distance(data_path_element, args.prediction_ending, args.max_offset)
        logged_score_values = []
        if False:
            for i, e in enumerate(score_values):
                log_metric('edit_distance', e, step=i) 
                logged_score_values.append(e)
                mean_edit_distance_up_until_now = sum(logged_score_values)/len(logged_score_values)
                log_metric('mean_edit_distance', mean_edit_distance_up_until_now, step=i)

        ax.plot(range(len(score_values)), score_values, args.style_info)

        if False:
            # source: https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
            z = numpy.polyfit(range(len(score_values)), score_values, 100)
            p = numpy.poly1d(z)
            ax.plot(range(len(score_values)),p(range(len(score_values))),"-")

    ax.set(xlabel=args.x_label, ylabel=args.y_label,
            title=title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    print('\n\n\n\n' + data_path)
    if not args.in_one_plot:
        output_plot()

if args.in_one_plot:
    output_plot()
