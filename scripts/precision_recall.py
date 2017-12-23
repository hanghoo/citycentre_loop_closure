#!/usr/bin/env python

import os
import sys
from IPython import embed

import matplotlib
matplotlib.use('GTKAgg')

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.io as sio
import numpy as np

#GROUND_TRUTH_PATH = os.path.expanduser(
#   '~/bags/IJRR_2008_Dataset/Data/NewCollege/masks/NewCollegeGroundTruth.mat')

GROUND_TRUTH_PATH = os.path.expanduser(
   '~/DBoW2-master/build/images/NewCollegeGroundTruth.mat')

#WORK_FOLDER = os.path.expanduser(
#   '~/dev/simple_slam_loop_closure/out/')

WORK_FOLDER = os.path.expanduser(
    '~/simple_slam_loop_closure/out/')

if __name__ == "__main__":
    gt_data = sio.loadmat(GROUND_TRUTH_PATH)['truth'][::2, ::2] 
    gt_data = gt_data + np.eye(1073) + gt_data.T

    bow_data = np.loadtxt(os.path.join(
        WORK_FOLDER, 'confusion_matrix.txt'))
    # Take the lower triangle only
    #bow_data = np.tril(bow_data, -1)
    bow_data = bow_data

    prec_recall_curve = []

    for thresh in np.arange(0, 1, 0.02): 
        # precision: fraction of retrieved instances that are relevant
        # recall: fraction of relevant instances that are retrieved
        true_positives = (bow_data > thresh) & (gt_data == 1)
        all_positives = (bow_data > thresh)

        try:
            precision = float(np.sum(true_positives)) / np.sum(all_positives)
            recall = float(np.sum(true_positives)) / np.sum(gt_data == 1)

            prec_recall_curve.append([thresh, precision, recall])
        except:
            break

    prec_recall_curve = np.array(prec_recall_curve)

    plt.plot(prec_recall_curve[:, 1], prec_recall_curve[:, 2])

    for thresh, prec, rec in prec_recall_curve[30::5]:
        plt.annotate(
            str(thresh),
            xy=(prec, rec),
            xytext=(8, 8),
            textcoords='offset points')

    plt.xlabel('Precision', fontsize=14)
    plt.ylabel('Recall', fontsize=14)

    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(
        WORK_FOLDER, 'prec_recall_curve.png'),
        bbox_inches='tight')
