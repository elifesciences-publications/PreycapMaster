import numpy as np
import pickle
import cv2
import math
from matplotlib import pyplot as pl
import os
from phinalIR_cluster_wik import Variables
from phinalFL_cluster import Fluorescence_Analyzer
from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans, convolve
import seaborn as sb


def fluor_wrapper(drct_lists_by_condition):

    def fill_condition_list(drct_list):
        fl_gutvals = []
        fl_gutintensity = []
        fl_gutarea = []
        fl_lowres = []
        gkern = Gaussian1DKernel(.5)
        for drct in drct_list:
            fish_fl = pickle.load(open(
                drct + '/fluordata.pkl', 'rb'))
            fl_gutvals.append(fish_fl.gut_values)
            g_int = [np.max([g1, g2]) for g1, g2 in zip(
                fish_fl.gutintensity_xy,
                fish_fl.gutintensity_xz)]
            fl_gutintensity.append(g_int)
            g_area = [np.max([g1, g2]) for g1, g2 in zip(
                fish_fl.gutintensity_xy,
                fish_fl.gutintensity_xz)]
            fl_gutarea.append(g_area)
            g_lowres = [np.max([g1, g2]) for g1, g2 in zip(
                fish_fl.lowres_gut_xy,
                fish_fl.lowres_gut_xz)]
            fl_lowres.append(g_lowres)
        filt_gutvals = [convolve(flgv, gkern,
                                 preserve_nan=False) for flgv in fl_gutvals]
        filt_gutintensity = [convolve(fl_int, gkern,
                                      preserve_nan=False)
                             for fl_int in fl_gutintensity]
        filt_gutarea = [convolve(fla, gkern,
                                 preserve_nan=False) for fla in fl_gutarea]
        filt_lowres = [convolve(lr, gkern,
                                preserve_nan=False) for lr in fl_lowres]
        return [filt_gutvals, filt_gutintensity,
                filt_gutarea, filt_lowres]
    
    fl_condition_list = []
    # fl_condition_list will contain lists with 4 elements each
    # each of the four elements is a fl readout (gutval, avg, lowres, size)
    # each element contains x lists of that value, where x is the
    # number of fish (i.e. directories) for that condition
    # you want an axis for each of these for a tsplot
    for drct_list in drct_lists_by_condition:
        fl_per_condition = fill_condition_list(drct_list)
        fl_condition_list.append(fl_per_condition)
    fig, axes = pl.subplots(2, 2, figsize=(10, 6))
    # want 4 plots, with the length of fl_condition_list
    # lines on each plot
    for fl_instance in fl_condition_list:
        sb.tsplot(fl_instance[0], ax=axes[0, 0])
        sb.tsplot(fl_instance[1], ax=axes[1, 0])
        sb.tsplot(fl_instance[2], ax=axes[0, 1])
        sb.tsplot(fl_instance[3], ax=axes[1, 1])
    pl.show()

fl_directory = '020419_1'
fl_obj = pickle.load(open(
    fl_directory + '/fluordata.pkl', 'rb'))

fluor_wrapper([[fl_directory]])
