import os
import copy
import csv
import cv2
import pandas as pd
import imageio
import itertools
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.cluster import SpectralClustering
import numpy as np
import pickle
import math
from scipy.ndimage.filters import gaussian_filter
from collections import Counter
import scipy.signal
from toolz.itertoolz import sliding_window, partition
# from sympy import Point3D, Plane
import matplotlib.cm as cm
from matplotlib import pyplot as pl
from matplotlib.colors import Normalize, ListedColormap
import seaborn as sb
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as anim
from scipy.spatial.distance import pdist, squareform
from phinalIR_cluster_wik import Variables
from phinalFL import Fluorescence_Analyzer
from pvidFINAL import Para, ParaMaster, return_paramaster_object
sb.set_style('darkgrid')


# Clustering on eye1, eye2, delta yaw, and interbout yields nice sep between aborts and strikes with spectral clustering and Bayesian mixture. Gaussian mixture is no good. 

# idea will be to have HD hold the para interp windows for hunts you actually care about
# when you add a hunt to hd, it will carry with it only the interp windows for the hunted para
# have to transform them using intwin and contwin just like in hunted_para_descriptor.
# can get this by simply adding myexp.paradata to the hd.update function.
# update function will do exactly what hunt_descriptor does now. no need to save anything. 



''' Instructions for using prey cap master code:

To begin, first create an Experiment class and use its bout_detector method to find all bouts. A nanfilter is run over the fish variables to assign values to unknown fish coords. Next, create a bouts and flags (BoutsFlags) instance for every Experiment bout detection. This is used by the DimensionalityReduce class to cluster either single fish behaviors or multiple fish. Finally, use the Experiment class method create_unit_vectors to eventually map all bouts to self-referenced spherical coordinates. 

Once the Experiment class has been set up, you have the option to cluster behaviors on one or many fish. Start with single fish clustering and see what you can pull out. Create a DimensionalityReduce instance and run dim_reduce(2) on it (3 if you want 3D clustering, which doesn't really help). Based on your choice of clusters, you now enter the hunt_windows into the class by calling find_hunts on your chosen hunt and deconverge clusters. 

To interact with hunts, first call watch_hunt on the first index. Every hunt bout is contained in dim.hunt_wins. Hunts begin at candidate hunt initiation bouts and run for 10 bouts. To encapsulate hunts, you can extend the hunt window (extend_hunt_window) or cap it (cap_hunt_window). Capping gives you possible deconverge bouts from clustering, and you choose the correct index to end on. Once you've decided that the hunt is encapsulated by an initiation and a deconvergence, you can decide to characterize the paramecia environment. This is done by pushing "1" during watch hunt when it is suggested to fully characterize. Any other button cancels the analysis. Calling nexthunt now goes to the next hunt, and pressing spacebar will continually cycle through hunts until you find one that is satisfying. Pushing 2 during nexthunt will stop the string of recurring hunt displays. An important feature for judging hunts is calling repeat but with the opposite plane video to watch the hunt from both sides. Calling bouts_during_hunt lets you see the entire tail profile of the hunt and examine whether the bouts called look reasonable. They usually look fine. It can also show you why the clustering may have missed a deconverge bout, etc. 

Once you've chosen to analyze a para environment, you can watch the results of the calls using exp.paradata.watch_event(). You will notice that the fish occludes the paramecium at some points, which disconnects continuous records. You can call manual_join to join records, manual_fix to try to add single plane records to xyz records, and manual_remove to get rid of clearly incorrect calls (happens very rarely -- you are usually the one to introduce bad calls). You can call manual_match to join together an XY and XZ rec that were not previously paired. Once you have fully characterized the paramecia, include the hunt into a HuntDescriptor. 

For adding into the HuntDescriptor, just call hd.update_hunt_data(exp.current_hunt_ind.....)
The last entry into the hd is an endbout. If its simply the deconvergence bout, enter -1. Otherwise if the consumption of the para happens without a strike (which sometimes happens, then they swim off in a large deconvergence swim), use -2 etc. 

Once you are finished with your HuntDescriptor, call para_stimuli to generate a csv file with all initial stimuli information, hunted_para_descriptor to generate a csv file with all hunting bouts characterized relative to para location. You can also call huntbouts_wrapped to obtain the 
z and eye angle trajectories over each hunt, which is valuable for convincing people that hunt initiations do begin with eye vergence. 

'''




# The Experiment class wraps all methods one level up from finding fish and para related variables. It reveals how fish and para interact. Para can be mapped to the eyes of the fish and the heading angle of the fish. Behavior of paramecia during hunting epochs can be analyzed.


# Make a Pandas Dataframe for each hunt. Include the post-continuity window coordinates
# for every hunted paramecium. Put the initial conditions in a separate dictionary. 


class IdealFishData:
    pass


class RealFishControl:
    def __init__(self, exp):
        self.firstbout_para_intwin = 5
        self.fish_xyz = exp.ufish_origin
        filter_sd = 1
        self.pitch_all = np.radians(
            gaussian_filter(exp.fishdata.pitch,
                            filter_sd))
        self.yaw_all = np.radians(unit_to_angle(
            filter_uvec(
                ang_to_unit(exp.fishdata.headingangle), filter_sd)))
        self.fish_id = exp.directory[-8:]
        self.hunt_results = []
        self.hunt_firstframes = []
        self.hunt_interbouts = []
        self.initial_conditions = []
        self.hunt_dataframes = []
        self.para_xyz_per_hunt = []
        self.directory = exp.directory

# exp contains all relevant fish data. frames will come from hunted_para_descriptor, which will
# create realfishcontrol objects as it runs. 

    def find_initial_conditions(self):
        for firstframe in self.hunt_firstframes:
            self.initial_conditions.append([self.fish_xyz[firstframe],
                                            self.pitch_all[firstframe],
                                            self.yaw_all[firstframe]])

    def model_input(self, hunt_num):
        return {"Hunt Dataframe": self.hunt_dataframes[hunt_num],
                "Para XYZ": self.para_xyz_per_hunt[hunt_num],
                "Initial Conditions": self.initial_conditions[hunt_num],
                "Interbouts": self.hunt_interbouts[hunt_num],
                "First Bout Delay": self.firstbout_para_intwin}

    def exporter(self):
        self.find_initial_conditions()
        with open(
                self.directory + '/RealHuntData_' + self.fish_id + '.pkl',
                'wb') as file:
            pickle.dump(self, file)
        

# This class kind of got out of hand as I continued to add information
# At this point probably works better as a container for dictionaries.
# Each hunt should have a dictionary that states its properties.
# Remove method should simply search dictionaries for input hunt ID.
# When constructed, should take an exp so you don't have to add every time.
# This will yield a proper pcw and integration window
# Simply update any functions that take hds to take new class
            
class Hunt_Descriptor:
    def __init__(self, directory):
        self.hunt_ind_list = []
        self.para_id_list = []
        self.actions = []
        self.boutrange = []
        self.directory = directory
        self.infer_z = []
        self.interp_windows = []

    def exporter(self):
        with open(self.directory + '/hunt_descriptor.pkl', 'wb') as file:
            pickle.dump(self, file)

    def parse_interp_windows(self, exp, poi):
        cont_win = exp.paradata.pcw
        int_win = exp.integration_window
        iw = copy.deepcopy(exp.paradata.interp_indices)
        unique_wins = [k for k, a in itertools.groupby(iw)]
        inferred_windows_poi = [
            np.array(win[1]) - cont_win - int_win
            for win in unique_wins if win[0] == [poi]]
        if inferred_windows_poi:
            inferred_windows_poi = inferred_windows_poi[0]
        inferred_window_ranges_poi = [
            range(win[0], win[1]) for win in inferred_windows_poi]
        self.interp_windows.append(inferred_window_ranges_poi)
        
    def current(self):
        print self.hunt_ind_list
        print self.para_id_list
        print self.actions
        print self.boutrange
        print self.infer_z

    def remove_entry(self, ind):
        del self.hunt_ind_list[ind]
        del self.para_id_list[ind]
        del self.actions[ind]
        del self.boutrange[ind]
        del self.interp_windows[ind]
        del self.infer_z[ind]

    def update_hunt_data(self, p, a, br, exp, maxz):
        self.hunt_ind_list.append(copy.deepcopy(exp.current_hunt_ind))
        self.para_id_list.append(p)
        self.actions.append(a)
        self.boutrange.append(br)
        self.parse_interp_windows(exp, p)
        self.infer_z.append(maxz)
        self.exporter()


# 	1) Hunted Para
#       2) Strike Success = 1,
# 	   Strike Fail = 2,
# 	   Abort = 3,
# 	   Abort for Reorientation = 4,
# 	   Reorientation and successful strike = 5,
# 	   Reorientation and fail = 6,
# 	   Reorientation and abort = 7,
#       3) Range of bouts within hunt. Enter as tuple
#       4) enter myexp
#       5) enter True if you assigned max Z, false otherwise

        
# for each para env, you will get a coordinate matrix that is scalexscalexscale, representing
# 3270 x 2 pixels in all directions. the scale must be ODD!! this guarantees that it will have a center
# cube. make a coordinate matrix by adding each unit vector representing the fish basis to the scale / 2, scale /2 , scale / 2 coordinate.
# the unit vectors are scaled by 3270 / scale. coord - scale / 2 will give you the right scale factor. e.g. if the scale is 10, [0,0,0] will be
# -5, -5, -5. when you multiply the scaled unit vectors of the basis by these coords and add to ufish origin, you get a real x,y,z coord of tank position. if# all coords are less than 1888, give the coordinate of env_mat a 1 bump. if not, give a zero bump. never give a bump to negative x coordinates! (i.e. start # the loop at scale / 2 for x, 0 for y and z.


class ParaEnv:

    def __init__(self, index, directory):
        self.wrth = np.load(
            directory + '/wrth' + str(index).zfill(2) + '.npy')
        self.ufish = np.load('/Users/nightcrawler2/ufish.npy')
        self.ufish_origin = np.load('/Users/nightcrawler2/ufish_origin.npy')
        self.uperp = np.load('/Users/nightcrawler2/uperp.npy')
        self.para3D = np.load(
            directory + '/para3D' + str(index).zfill(2) + '.npy')
#        self.high_density_coord = np.load('high_density_xyz.npy')
        self.paravectors = []
        self.dotprod = []
        self.velocity_mags = []
        self.target = []
        self.bout_number = str(index)
        self.vector_spacing = 3

# Question is whether fish wait, follow, or move to the predicted location of para behind barriers. 
# Once you find the right metric for attention to a para, ask if ufish multiplied by any integer intersects 
# With the coordinates of a barrier.

    def exporter(self):
        with open('p_env' + self.bout_number + '.pkl', 'wb') as file:
            pickle.dump(self, file)

#Use ufish vectors here to see if the para is behind an occluder. If it is, give it a flag.             
    def barrier_occlusion(self):
        pass
        
    def find_paravectors(self, plot_para, *para_id):
        para_of_interest = []
        if para_id == ():
            para_of_interest = range(0, self.para3D.shape[0], 3)
        else:
            para_of_interest = [para_id[0]*3]
        for rec_rows in para_of_interest:
            win_length = self.vector_spacing
            rec_id = rec_rows / 3
            x = self.para3D[rec_rows]
            y = self.para3D[rec_rows+1]
            z = self.para3D[rec_rows+2]
            # x_diffs = [a[-1] - a[0] for a in sliding_window(win_length, x)]
            # y_diffs = [a[-1] - a[0] for a in sliding_window(win_length, y)]
            # z_diffs = [a[-1] - a[0] for a in sliding_window(win_length, z)]
            x_diffs = [a[-1] - a[0] for a in partition(win_length, x)]
            y_diffs = [a[-1] - a[0] for a in partition(win_length, y)]
            z_diffs = [a[-1] - a[0] for a in partition(win_length, z)]
            vel_vector = [
                np.array([deltax, deltay, deltaz])
                for deltax, deltay, deltaz in zip(x_diffs, y_diffs, z_diffs)
            ]
            mag_velocity = [np.sqrt(np.dot(vec, vec)) for vec in vel_vector]
            paravector_normalized = [vec / mag
                                     for vec, mag
                                     in zip(vel_vector, mag_velocity)]
# you could make sure that only changes greater than 1 are recorded.
            self.paravectors.append(
                [vel_vector, mag_velocity, paravector_normalized])
            dots = [np.dot(vec, vec_prev)
                    for vec_prev, vec
                    in sliding_window(2, paravector_normalized)]
            dots_nonan = [dott for dott in dots if not math.isnan(dott)]
            self.dotprod.append(dots)
            self.velocity_mags.append(mag_velocity)
            if plot_para:
                pl.ioff()
                if dots:
                    fig = pl.figure(figsize=(8, 8))
                    ax = fig.add_subplot(311)
                    ax2 = fig.add_subplot(312)
                    ax3 = fig.add_subplot(313)
                    ax.hist(dots_nonan, 50)
                    ax.set_ylim([0, 100])
                    ax.set_title('Para Record' + str(rec_id))
                    ax2.plot(dots)
                    ax2.set_xlim([0, self.para3D.shape[1]])
                    ax2.set_ylim([-1, 1])
                    ax2.set_title('Dot Product W/ Prev Vector Over Time')
                    ax3.set_xlim([0, self.para3D.shape[1]])
                    ax3.set_ylim([0, 4])
                    ax3.plot(mag_velocity)
                    ax3.set_title('Velocity Over Time')
                    pl.tight_layout()
                    pl.show()
        
# Para objects in wrth have the following attributes:
#  0: x
#  1: y 
#  2: z
#  3: ID
#  4: mag of vector to para from fish head
#  5: 3D angle via 3D dot product to para
#  6: azimuth angle in 2D
#  7: altitude angle in 2D 

    def p_ratios(self):
        firstframe_para = np.array([para[3] for para in self.wrth[0]])
        lastframe_para = np.array([para[3] for para in self.wrth[-1]])
        constant_para = np.intersect1d(firstframe_para, lastframe_para)
        percent_remaining = constant_para.shape[0] / firstframe_para.shape[0]
        targets = []
        for pr in constant_para:
            para_begin = [p for p in self.wrth[0] if p[3] == pr][0]
            para_end = [p for p in self.wrth[-1] if p[3] == pr][0]
            mag_minimized = para_begin[4] > para_end[4]
            az_minimized = abs(para_begin[6]) > abs(para_end[6])
            alt_minimized = abs(para_begin[7]) > abs(para_end[7])
            if mag_minimized and az_minimized and alt_minimized:
                targets.append(pr)
        self.target = targets

    def para_com_and_density(self):
        nanlist = [float('nan'), float('nan'), float('nan')]
        pstate1 = np.array(self.wrth[0])
        pstate2 = np.array(self.wrth[-1])
        if pstate1.any() and pstate2.any():
            com_xyz_start = np.mean(pstate1, axis=0)[0:3]
            com_xyz_end = np.mean(pstate2, axis=0)[0:3]
            return com_xyz_start, com_xyz_end
        else:
            return nanlist, nanlist


class BoutsAndFlags():

    def __init__(self, directory, bouts, flags):
        self.bouts = bouts
        self.flags = flags
        self.dirct = directory

    def exporter(self):
        with open(
                self.dirct + '/bouts.pkl', 'wb') as file:
            pickle.dump(self, file)

        
# This class will take in BoutsAndFlags objects for each fish then run TSNE on the combined dataset
# Want this to be able to take the boutsandflags from ALL fish. You want to add a dictionary with the fishids you want to cluster on.
# 

class DimensionalityReduce():

    def __init__(self,
                 bout_dict,
                 flag_dict, all_varbs_dict, directory, fish_id_dict):
        self.recluster = True
        self.directory = directory
        self.cluster_input = []
        self.num_dp = len(all_varbs_dict)
        self.all_varbs_dict = all_varbs_dict
        self.all_bouts = []
        self.all_flags = []
        self.dim_reduce_output = []
        self.cmem_pre_sub = []
        self.cluster_membership = []
        self.bout_dict = bout_dict
        self.num_bouts_per_fish = []
        self.flag_dict = flag_dict
        self.inv_fdict = {v: k for k, v in flag_dict.iteritems()}
        self.transition_matrix = np.array([])
        self.bouts_flags = [pickle.load(open(directory + '/bouts.pkl'))]
        self.cluster_count = 3
        self.hunt_cluster = []
        self.deconverge_cluster = []
        self.hunt_wins = []

# This will be for the future if you want to cluster all fish in the fish_id_dict. Just add the IDs from the dict to the flag data.

    def revert_cmem(self):
        self.cluster_membership = copy.deepcopy(self.cmem_pre_sub)

    def subcluster_hunts(self, sub_dict, orig_cluster, num_clusters):
        self.cmem_pre_sub = copy.deepcopy(self.cluster_membership)
        orig_cluster_indices = np.where(
            self.cluster_membership == orig_cluster)[0].tolist()
        cluster_model = SpectralClustering(n_clusters=num_clusters,
                                           affinity='nearest_neighbors',
                                           n_neighbors=10)
        keys = np.sort(np.array(sub_dict.keys()).astype(np.int))
        sub_cluster_input = []
        bt_array = np.array(self.all_bouts)
        std_array = [np.nanstd(
            bt_array[:, i:len(self.all_bouts[0])+1:self.num_dp])
                     for i in range(self.num_dp)]
        for b_ind, bout in enumerate(self.all_bouts):
            if b_ind in orig_cluster_indices:
                sub_bout = []
                b_partitioned = partition(self.num_dp, bout)
                for bout_frame in b_partitioned:
                    norm_frame = (np.array(bout_frame) / std_array)
                    sub_bout_frame = norm_frame[keys].tolist()
                    sub_bout += sub_bout_frame
                sub_cluster_input.append(sub_bout)
        cmem = cluster_model.fit_predict(np.array(sub_cluster_input))
        max_cluster_id = np.max(self.cluster_membership)
# What this does is leave the original cluster id alone for one subcluster,
# then add extra subclusters via adding more id's to cluster_membership
        for b_id, c_ind in enumerate(orig_cluster_indices):
            if cmem[b_id] != 0:
                self.cluster_membership[c_ind] = max_cluster_id + cmem[b_id]


            


        # what you will do is go through cluster_membership and assign a
        # 3 to clusters that come out 1, and keep at hunt_cluster the ones that come out 0.
        # then you can call cluster_summary and change the hunt_wins based on your observation. 
            
    def strike_abort_sep(self, term_cluster):
        cluster_indices = np.where(self.cluster_membership == term_cluster)[0]
            # now get all bouts from cluster indices in all_bouts
        flags_in_term_cluster = np.array(self.all_flags)[cluster_indices]
        interbout_index = int(dim.inv_fdict['Interbout'])
        delta_yaw_index = int(dim.inv_fdict['Total Yaw Change'])
        interbouts_in_cluster = [
            f[interbout_index] for f in flags_in_term_cluster]
        dyaw_in_cluster = [
            np.abs(f[delta_yaw_index]) for f in flags_in_term_cluster]
        ib_plot = sb.distplot(interbouts_in_cluster, bins=100)
        ib_plot.set_title('Interbouts in Term Cluster')
        pl.show()
        dy_plot = sb.distplot(dyaw_in_cluster, bins=100)
        dy_plot.set_title('Delta Yaw in Term Cluster')
        pl.show()
        return interbouts_in_cluster, dyaw_in_cluster

    def watch_cluster(self, exp, cluster, vidtype, term):
        if vidtype == 1:
            vid = imageio.get_reader(
                self.directory + '/top_contrasted.AVI', 'ffmpeg')
        elif vidtype == 0:
            vid = imageio.get_reader(
                self.directory + '/conts.AVI', 'ffmpeg')

        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        c_mem_counter = 0
        if term:
            ib, dy = self.strike_abort_sep(cluster)
        for bout_num, cluster_mem in enumerate(self.cluster_membership):
            if cluster_mem == cluster:
                print('Bout #: ' + str(bout_num))
                if term:
                    dyaw = dy[c_mem_counter]
                    c_mem_counter += 1
                    if dyaw > 10:
                        continue
                firstframe = exp.bout_frames[bout_num] - 10
                if firstframe <= 0:
                    continue
                lastframe = exp.bout_frames[bout_num] + 30
                for frame in range(firstframe, lastframe):
                    im = vid.get_data(frame)
                    im = cv2.resize(im, (700, 700))
                    if frame == lastframe - 1:
                        im = np.zeros([700, 700])
                    cv2.imshow('vid', im)
                    keypress = cv2.waitKey(delay=15)
                    if keypress == 50:
                        cv2.destroyAllWindows()
                        vid.close()
                        return

        cv2.destroyAllWindows()
        vid.close()
        return

    def extract_and_assignID(self, file_id):
        boutdata = pickle.load(open(file_id, 'rb'))
        num_bouts = len(boutdata.bouts)
        self.num_bouts_per_fish.append(num_bouts)
        new_flags = []
        for flag in boutdata.flags:
            flag.append(fish_id)
            new_flags.append(flag)
        boutdata.flags = new_flags
        return boutdata

    def exporter(self):
        with open(self.directory + '/dim_reduce.pkl', 'wb') as file:
            pickle.dump(self, file)

    def clear_huntwins(self):
        self.hunt_wins = []
        self.hunt_cluster = []
        self.deconverge_cluster = []

    def concatenate_records(self):
        for record in self.bouts_flags:
            self.all_bouts = self.all_bouts + record.bouts
            self.all_flags = self.all_flags + record.flags
        
# Looks like a typo, but since bout inversion mixes left and right, wanted to have a standard deviation 
# for "an eye". 
        bt_array = np.array(self.all_bouts)
        std_array = [np.nanstd(
            bt_array[:, i:len(self.all_bouts[0])+1:self.num_dp])
                     for i in range(self.num_dp)]
        print('Std_Array')
        print std_array
        norm_bouts = []
        bdict_keys = sorted([int(key) for key in self.bout_dict.keys()])
# this loop normalizes each value by the std_array and filters for the variables you want to cluster on
        for b in self.all_bouts:
            norm_bout = []
            b_partitioned = partition(self.num_dp, b)
            for b_frame in b_partitioned:
                norm_frame = (np.array(b_frame) / std_array)
                filt_norm = norm_frame[bdict_keys].tolist()
                norm_bout += filt_norm
            norm_bouts.append(norm_bout)
        self.cluster_input = norm_bouts
        print(str(len(self.all_bouts)) + " Bouts Detected")
        if self.recluster:
            cluster_model = SpectralClustering(n_clusters=self.cluster_count,
                                               affinity='nearest_neighbors',
                                               n_neighbors=10)
        c_flag = cluster_model.fit_predict(np.array(norm_bouts))
        self.cluster_membership = c_flag
        num_clusters = np.unique(self.cluster_membership).shape[0]
        transition_mat = np.zeros([num_clusters, num_clusters])
        for i, j in sliding_window(2, self.cluster_membership):
            transition_mat[i, j] += 1
        transition_mat /= self.cluster_membership.shape[0]
        self.transition_matrix = transition_mat
# Add 1 so there is no cluster 0
        new_flags = []
        for flag, cflag in zip(self.all_flags, self.cluster_membership):
            flag.append(cflag)
            new_flags.append(flag)
        self.all_flags = new_flags

    def transition_bar(self, cluster):
        palette = sb.color_palette('husl', 8)
        probs = self.transition_matrix[cluster]
        probs /= np.sum(probs)
        fig = pl.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.bar([ind for ind, p in enumerate(probs)],
               probs, align='center', color=palette[1])
        ax.set_title('Transition Probability, Cluster'+str(cluster))
        ax.set_xlim([-.5, self.transition_matrix.shape[0] - .5])
        ax.grid(False)
        pl.show()

    def find_hunts(self, init_inds, abort_inds):
        self.hunt_cluster = init_inds
        self.deconverge_cluster = abort_inds
        start_bouts_per_hunt = 10
        for ind, win in enumerate(
                sliding_window(start_bouts_per_hunt,
                               self.cluster_membership)):
            if win[0] in self.hunt_cluster:
                self.hunt_wins.append([ind, ind + start_bouts_per_hunt])

    def extend_hunt_window(self, exp, numbouts):
        self.hunt_wins[exp.current_hunt_ind][1] += numbouts
        self.exporter()

    def reset_hunt_window(self, exp):
        orig = self.hunt_wins[exp.current_hunt_ind][0]
        self.hunt_wins[exp.current_hunt_ind] = [orig, orig+10]

# one bug is deconverge_cluster should be cleared when find_hunts is called        
    def cap_hunt_window(self, exp):
        ind = exp.current_hunt_ind
        start_ind = self.hunt_wins[ind][0]
        original_end_ind = self.hunt_wins[ind][1]
        cluster_mem = self.cluster_membership[
            start_ind:original_end_ind+1].tolist()
        deconverge_inds = [
                i for i, v in enumerate(
                    cluster_mem) if v in self.deconverge_cluster]
        print('Candidate Deconvergences')
        print deconverge_inds
        di = raw_input('Enter Correct Deconvergence:  ')
        try:
            di = int(di)
        except ValueError:
            return
        candidate_window = [start_ind, start_ind + di]
        self.hunt_wins[ind] = candidate_window
        exp.watch_hunt(self, 1, 15, ind)
        response = raw_input('Cap Window?  ')
        if response == 'y':
            self.exporter()
        else:
            self.hunt_wins[ind] = [start_ind, original_end_ind]
        view_bouts = raw_input("View Bouts During Hunt?: ")
        if view_bouts == 'y':
            exp.watch_hunt(self, 0, 50, ind)
            bouts_during_hunt(exp.current_hunt_ind, dim, exp, True)

    def dim_reduction(self, dimension):
        print('running dimensionality reduction')
        np.set_printoptions(suppress=True)
        np.random.seed(1)
        model = SpectralEmbedding(n_components=dimension,
                                  affinity='nearest_neighbors',
                                  gamma=None,
                                  random_state=None,
                                  eigen_solver=None,
                                  n_neighbors=None,
                                  n_jobs=1)
        for b in self.all_bouts:
            if not np.isfinite(b).all():
                print('found nan or inf')
        dim_reduce_data = model.fit_transform(np.array(self.all_bouts))
        self.dim_reduce_output = dim_reduce_data

    def cluster_summary(self, *cluster_id):
        pl.ioff()
        print Counter(self.cluster_membership)
        num_dp = len(self.all_varbs_dict)
        palette = np.array(sb.color_palette("Set1", num_dp))
        cluster_ids = np.unique(self.cluster_membership)
        all_bouts = np.array(self.all_bouts)
        if cluster_id != ():
            cluster_ids = cluster_id
        for _id in cluster_ids:
            cluster_indices = np.where(self.cluster_membership == _id)[0]
            # now get all bouts from cluster indices in all_bouts
            bouts_in_cluster = all_bouts[cluster_indices]
# Could choose to cluster more here if you want
            data_points = {}
# initialize dictionary
            for i in range(num_dp):
                data_points[str(i)] = []
            for bout in bouts_in_cluster:
                for i in range(num_dp):
                    data_points[str(i)].append([list(p)[i]
                                                for p
                                                in partition(num_dp, bout)])
            num_cols = int(np.ceil(num_dp/3.0))
            num_rows = 3
            fig = pl.figure(figsize=(8, 8))
            bdict_keys = sorted(
                [int(key) for key in self.all_varbs_dict.keys()])
            for i in bdict_keys:
                ax = fig.add_subplot(num_rows, num_cols, i+1)
                ax.set_title(self.all_varbs_dict[str(i)])
                # if self.all_varbs_dict[str(i)] == 'Interbout_Back':
                #     sb.barplot(data=data_points[str(i)])
                #     continue

# Eventually make the "datapoint" keyword a dictionary of variable names
# you can also use the dictionary to access points for cluster plots
                sb.tsplot(data_points[str(i)], color=palette[i], ci=95)
                if self.all_varbs_dict[str(i)][0:3] == 'Eye':
                    ax.set_ylim([-60, 60])
                elif self.all_varbs_dict[str(i)] == 'Interbout_Back':
                    ax.set_ylim([0, 100])
                elif self.all_varbs_dict[str(i)][0:4] == 'Vect':
                    ax.set_ylim([0, 3])
                elif self.all_varbs_dict[str(i)] == 'Delta Z':
                    sb.tsplot([np.cumsum(dp)
                               for dp in data_points[str(i)]],
                              color='k',
                              ci=95)
                elif self.all_varbs_dict[str(i)] == 'Delta Yaw':
                    sb.tsplot([np.cumsum(dp)
                               for dp in data_points[str(i)]],
                              color='k',
                              ci=95)
                
            pl.suptitle('Cluster' + str(_id))
            pl.subplots_adjust(top=0.9, hspace=.4, wspace=.4)
            pl.savefig(self.directory + '/cluster' + str(_id) + '.tif')
            pl.show()

    def flags_in_cluster(self, cluster_id, flag_id):
        all_flags = np.array(self.all_flags)
        cluster_indices = np.where(
            self.cluster_membership == cluster_id)[0]
        flags_in_cluster = all_flags[cluster_indices]
# Could choose to cluster more here if you want
        flag_title = self.flag_dict[str(flag_id)]
        fl_plot = []
        for flag in flags_in_cluster:
            fl_plot.append(flag[flag_id])
        pl.hist(fl_plot, bins=10, color='r')
        pl.title(flag_title)
        pl.show()

    def cluster_plot(self, dimensions, flagnum):
        pl.ion()
        fish_id_flag = int(self.inv_fdict['Fish ID'])
        cluster_flag_id = int(self.inv_fdict['Cluster ID'])
        # fish_id_flag is the last entry in a flag set.
        palette = np.array(sb.color_palette("cubehelix", 10))
        flags = np.array([fl[flagnum] for fl in self.all_flags])
        if not flagnum == fish_id_flag and not flagnum == cluster_flag_id:
            flag_ranks, flagbins = pd.qcut(flags, 10,
                                           labels=False,
                                           retbins=True)
        else:
            flag_ranks = flags
        print(np.unique(flag_ranks))
        fig = pl.figure(figsize=(8, 8))
        if dimensions == 2:
            if flagnum == cluster_flag_id:
                palette = np.array(
                    sb.color_palette(
                        "cubehelix", np.max(self.cluster_membership)+1))
            ax = fig.add_subplot(121)
            ax.scatter(self.dim_reduce_output[:, 0],
                       self.dim_reduce_output[:, 1],
                       lw=0,
                       s=40,
                       c=palette[flag_ranks])
            ax.set_axis_off()
        
        elif dimensions == 3:
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(self.dim_reduce_output[:, 0],
                       self.dim_reduce_output[:, 1],
                       self.dim_reduce_output[:, 2],
                       lw=0,
                       s=40,
                       c=palette[flag_ranks])
            ax.set_axis_off()

        sm = cm.ScalarMappable(cmap=ListedColormap(palette),
                               norm=pl.Normalize(vmin=0,
                                                 vmax=int(palette.shape[0])))
        sm._A = []
        pl.colorbar(sm)
        ax2 = fig.add_subplot(122)
        n, bins, patches = ax2.hist(flags, bins=100)
#        new_patchlist = []
        if not flagnum == fish_id_flag and not flagnum == cluster_flag_id:
            for bn, patch in zip(bins, patches):
                bin_rank = [ind
                            for ind, fb
                            in enumerate(sliding_window(2, flagbins))
                            if bn >= fb[0] and bn < fb[1]]
                pl.setp(patch, 'facecolor', palette[bin_rank[0]])

        ax2.set_axis_bgcolor('w')
        ax2.yaxis.tick_right()
        pl.suptitle(self.flag_dict[str(flagnum)])
        pl.subplots_adjust(top=0.9)
        pl.show()
             
# ok so bins demark the beginning of each bin. assign a rank to each bin based on its location in flagbins


class Experiment():
    def __init__(self, cluster_length,
                 refractory_period, bout_dict, flag_dict, dirct):
        self.ufish = []
        self.ufish_origin = []
        self.uperp = []
        self.upar = []
        self.invert = True
        self.refract = refractory_period
        self.bout_dict = bout_dict
        self.flag_dict = flag_dict
        self.inv_bdict = {v: k for k, v in bout_dict.iteritems()}
        self.directory = dirct
        self.fishdata = pickle.load(open(
           self.directory + '/fishdata.pkl', 'rb'))
#        self.fluor_data = pickle.load(open('fluordata.pkl', 'rb'))
        self.set_types()
        self.filter_fishdata()
        self.delta_ha = []
        self.vectVcalc(2)
        self.fluor_data = []
        self.paradata = []
        self.framewindow = []
        self.barrierlocation = []
        self.bout_frames = []
        self.bout_durations = []
        self.rejected_bouts = []
        self.bout_data = []
        self.bout_flags = []
        self.minboutlength = cluster_length
        self.num_dp = len(bout_dict)
        self.para_win = 20
#just assures that 20 frames before bout is also continuous in order for para reconstructions to be accurate per bout. 
# This allows trajectory matching over x frames to improve correlation
        self.para_continuity_window = 600
        self.bout_of_interest = 0
        self.hunt_windows = []
        self.current_hunt_ind = 0
        self.integration_window = 30
        self.bout_az = []
        self.bout_alt = []
        self.bout_dist = []

# to do in bout detector: 
# HA must have no nans in 20 width window. 
# do bout detection for vel and tailangle here. 
 
    def set_types(self):
        self.fishdata.pitch = np.float64(self.fishdata.pitch)
        self.fishdata.phileft = np.float64(self.fishdata.phileft)
        self.fishdata.phiright = np.float64(self.fishdata.phiright)

    def watch(self, vid):
        self.paradata.watch_event(vid)

    def vectVcalc(self, num_dimensions):

        vectv = [0.0]
        xwin = sliding_window(3, self.fishdata.low_res_x)
        ywin = sliding_window(3, self.fishdata.low_res_y)
        zwin = sliding_window(3, self.fishdata.low_res_z)
        if num_dimensions == 3:
            vecwin = zip(xwin, ywin, zwin)
        elif num_dimensions == 2:
            vecwin = zip(xwin, ywin)
        for w in vecwin:
            w = np.array(w)
            vec1 = w[:, 0]
            vec2 = w[:, 1]
            vec3 = w[:, 2]
            if np.isfinite(vec1).all() and np.isfinite(vec2).all():
                vel = magvector(vec2-vec1)
            elif np.isfinite(vec1).all() and np.isfinite(vec3).all():
                vel = magvector(vec3-vec1) / 2
            else:
                vel = float('nan')
            vectv.append(vel)
# this is because windows stop before end
        vectv.append(vel)
        self.fishdata.vectV = np.array(vectv).astype(np.float64) / 16.0

    def filter_fishdata(self):
        self.fishdata.x = fill_in_nans(self.fishdata.x)
        self.fishdata.y = fill_in_nans(self.fishdata.y)
        self.fishdata.z = fill_in_nans(self.fishdata.z)
        self.fishdata.low_res_x = fill_in_nans(self.fishdata.low_res_x)
        self.fishdata.low_res_y = fill_in_nans(self.fishdata.low_res_y)
        self.fishdata.low_res_z = fill_in_nans(self.fishdata.low_res_z)
        self.fishdata.phiright = fill_in_nans(self.fishdata.phiright)
        self.fishdata.phileft = fill_in_nans(self.fishdata.phileft)
#        self.fishdata.vectV = fill_in_nans(self.fishdata.vectV)
        self.fishdata.pitch = fill_in_nans(self.fishdata.pitch)
        self.fishdata.headingangle = fill_in_nans(self.fishdata.headingangle)
        temp_tail = np.array(self.fishdata.tailangle)
        for i in range(len(temp_tail[0])):
            temp_tail[:, i] = fill_in_nans(temp_tail[:, i])
        self.fishdata.tailangle = temp_tail.tolist()

# This function will take in the ha_diff flag (total ha_diff) and invert ha, switch the eyes, and invert the tail angles.
# Use the itertoolz partition function with size num_dp over the entire bout data. 
# Inds 1-7 are tail, ha is 10, eyes are 11 and 12. 

# THIS FUNCTION AND FLAGS HAVE TO CHANGE BASED ON WHICH VARIABLES ARE ADDED TO THE CLUSTERING. YOU CANT LEAVE OUT HA B/C 
# IT CONTROLS BOUT INVERSION.

    def nexthunt(self, dim, vid):
        ret = self.watch_hunt(
            dim, vid, 15, self.current_hunt_ind + 1)
        if not ret:
            return self.nexthunt(dim, vid)

    def repeat(self, dim, vid):
        self.watch_hunt(
            dim, vid, 15, self.current_hunt_ind)

    def backone(self, dim):
        self.watch_hunt(
            dim, 1, 15, self.current_hunt_ind - 1)

    def bout_nanfilt_and_arrange(self, filter_walls):
        def get_var_index(var_name):
            try:
                v = int(self.inv_bdict[var_name])
            except KeyError:
                v = float('nan')
            return v

        def invert_bout(b_data):
            inverted_bout = []
            ey1 = get_var_index('Eye1 Angle')
            ey2 = get_var_index('Eye2 Angle')
            ha_ind = get_var_index('Delta Yaw')
            ts1 = get_var_index('Tail Segment 1')
            ts2 = get_var_index('Tail Segment 2')
            ts3 = get_var_index('Tail Segment 3')
            ts4 = get_var_index('Tail Segment 4')
            ts5 = get_var_index('Tail Segment 5')
            ts6 = get_var_index('Tail Segment 6')
            ts7 = get_var_index('Tail Segment 7')
            for part in partition(self.num_dp, b_data):
                part = list(part)
                if not math.isnan(ts1):
                    part[ts1] = -1*part[ts1]
                if not math.isnan(ts2):
                    part[ts2] = -1*part[ts2]
                if not math.isnan(ts3):
                    part[ts3] = -1*part[ts3]
                if not math.isnan(ts4):
                    part[ts4] = -1*part[ts4]
                if not math.isnan(ts5):
                    part[ts5] = -1*part[ts5]
                if not math.isnan(ts6):
                    part[ts6] = -1*part[ts6]
                if not math.isnan(ts7):
                    part[ts7] = -1*part[ts7]
                if not math.isnan(ey1):
                    part[ey1], part[ey2] = part[ey2], part[ey1]
                if not math.isnan(ha_ind):
                    part[ha_ind] = -1*part[ha_ind]
                inverted_bout += part
            return inverted_bout

        def nearwall(x_init, y_init, z_init, ha_init, pitch_init):
            wall_thresh = 50
            if x_init < wall_thresh and (90 < ha_init < 270):
                return True
            elif x_init > 1888-wall_thresh and (ha_init < 90 or ha_init > 270):
                return True
            elif y_init < wall_thresh and (ha_init > 180):
                return True
            elif y_init > 1888-wall_thresh and (ha_init < 180):
                return True
            elif z_init < wall_thresh and pitch_init < 0:
                return True
            elif z_init > 1888-wall_thresh and pitch_init > 0:
                return True
        
        all_bout_data = []
        filtered_bout_frames = []
        rejected_bouts = []
        filtered_bout_durations = []
        filtered_interbout = []
        bout_flags = []
        frametypes = np.load(self.directory + '/frametypes.npy')
# Frametypes contains an array of 1s for fluorescent frames and 0s for IR frames for the entire experiment. It describes the sequence
# of FL and IR frames in the raw cam0 and cam1 movies. all IR data is ONLY FOR the IR frames. You have to exclude IR data in bouts if it occurs
# directly after a FL frame because the internal timing of the bout will be off. Frametypes_IR is a new array for only IR frames. A 0 says it is an IR fram# that does not arrive after an FL frame while a 1 says it is an IR frame that occurs directly after an FL frame. Post_luor_inds are thus frames that should be excluded from bouts because they fall directly after an FL frame. 
        frametypes_ir = []
        post_fluor_inds = []
        prevframe = 0
        for frame in frametypes:
            if prevframe == 0:
                frametypes_ir.append(frame)
            prevframe = frame
        post_fluor_inds = [i for i, v in enumerate(frametypes_ir) if v == 1]
# enumerate fluor_inds to get the correct frame in fluor_data.gut_values
        bout_windows = [win for win in sliding_window(2, self.bout_frames)]
        ha_diffs = [0]
        z_diffs = [0]
        phileft_filt = gaussian_filter(self.fishdata.phileft, 1)
        phiright_filt = gaussian_filter(self.fishdata.phiright, 1)
        for h in sliding_window(2, self.fishdata.headingangle):
            if abs(h[1]-h[0]) < 180:
                ha_diffs.append(h[1]-h[0])
            elif h[1]-h[0] <= -180:
                diff = h[1] + (360-h[0])
                ha_diffs.append(diff)
            elif h[1]-h[0] >= 180:
                diff = -(360-h[1])+h[0]
                ha_diffs.append(diff)
            elif math.isnan(h[1]) or math.isnan(h[0]):
                ha_diffs.append(float('nan'))

        for z in sliding_window(2, self.fishdata.low_res_z):
            z_diffs.append(z[1]-z[0])

        for bindex, (bout, bout_duration) in enumerate(
                zip(bout_windows, self.bout_durations)):
            interbout_backwards = np.copy(
                bout[0] - bout_windows[bindex - 1][0])
            bout_vec = []
            ha_init = np.nanmean(self.fishdata.headingangle[bout[0]:bout[0]+5])
            x_init = np.nanmean(self.fishdata.x[bout[0]:bout[0]+5])
            y_init = np.nanmean(self.fishdata.y[bout[0]:bout[0]+5])
            z_init = np.nanmean(self.fishdata.z[bout[0]:bout[0]+5])
            pitch_init = np.nanmean(self.fishdata.pitch[bout[0]:bout[0]+5])
            
# x is 0,1888 left to right, y is 0,1888 bottom to top, 
            if nearwall(x_init,
                        y_init,
                        z_init,
                        ha_init,
                        pitch_init):
                rejected_bouts.append([bout[0], 'nearwall'])
# if filter_walls input to this function is false, bouts will still be candidates for clustering, but will be flagged in rejection. 
                if filter_walls:
                    continue
# THIS IS FOR CLUSTERING. FLAGS ARE COUNTED WITH BOUT DURATION AS THE ENDPOINT.             
            bout = (bout[0], bout[0] + self.minboutlength)
            full_window = (bout[0]-self.para_win, bout[1])
            if np.array(frametypes_ir[full_window[0]:full_window[1]]).any():
                rejected_bouts.append([bout[0], 'fluorframe'])
                continue
            
            if self.assure_nonans(full_window):
                filtered_bout_frames.append(bout[0])
                filtered_interbout.append(interbout_backwards)
                filtered_bout_durations.append(bout_duration)
                phil_start = phileft_filt[bout[0]]
                phir_start = phiright_filt[bout[0]]
                for ind in range(bout[0], bout[1]):
                    for key in range(len(self.bout_dict)):
                        if self.bout_dict[str(key)] == 'Pitch':
                            bout_vec.append(self.fishdata.pitch[ind])
                        elif self.bout_dict[str(key)] == 'Tail Segment 1':
                            bout_vec.append(self.fishdata.tailangle[ind][0])
                        elif self.bout_dict[str(key)] == 'Tail Segment 2':
                            bout_vec.append(self.fishdata.tailangle[ind][1])
                        elif self.bout_dict[str(key)] == 'Tail Segment 3':
                            bout_vec.append(self.fishdata.tailangle[ind][2])
                        elif self.bout_dict[str(key)] == 'Tail Segment 4':
                            bout_vec.append(self.fishdata.tailangle[ind][3])
                        elif self.bout_dict[str(key)] == 'Tail Segment 5':
                            bout_vec.append(self.fishdata.tailangle[ind][4])
                        elif self.bout_dict[str(key)] == 'Tail Segment 6':
                            bout_vec.append(self.fishdata.tailangle[ind][5])
                        elif self.bout_dict[str(key)] == 'Tail Segment 7':
                            bout_vec.append(self.fishdata.tailangle[ind][6])
                        elif self.bout_dict[str(key)] == 'Vector Velocity':
                            bout_vec.append(self.fishdata.vectV[ind])
                        elif self.bout_dict[str(key)] == 'Delta Z':
                            bout_vec.append(z_diffs[ind])
                        elif self.bout_dict[str(key)] == 'Delta Yaw':
                            bout_vec.append(ha_diffs[ind])
                        elif self.bout_dict[str(key)] == 'Eye1 Angle':
                            bout_vec.append(phileft_filt[ind] - phil_start)
                        elif self.bout_dict[str(key)] == 'Eye2 Angle':
                            bout_vec.append(phiright_filt[ind] - phir_start)
                        elif self.bout_dict[str(key)] == 'Eye Sum':
                            bout_vec.append(
                                phileft_filt[ind] -
                                phil_start + phiright_filt[ind] - phir_start)
                        elif self.bout_dict[str(key)] == 'Interbout_Back':
                            if bindex > 0:
                                interbout_from_previous = np.copy(
                                    ind - bout_windows[bindex - 1][0])
                            else:
                                interbout_from_previous = 0
                            bout_vec.append(interbout_from_previous)

# sets first ha_diff and xydiff to 0 given that there is no info about previous vals outside of window 
                
                dz = get_var_index('Delta Z')
                dh = get_var_index('Delta Yaw')
                if not math.isnan(dz) and math.isnan(bout_vec[dz]):
                    bout_vec[dz] = 0
                if not math.isnan(dh) and math.isnan(bout_vec[dh]):
                    bout_vec[dh] = 0
                delta_ha = np.sum(ha_diffs[bout[0]:bout[1]])
                self.delta_ha.append(delta_ha)
                if self.invert:
                    if delta_ha < 0:
                        bout_vec = invert_bout(bout_vec)
                all_bout_data.append(bout_vec)
            else:
                rejected_bouts.append([bout[0], 'nan in bout'])

        self.bout_data = all_bout_data
        self.bout_durations = filtered_bout_durations
        self.bout_frames = filtered_bout_frames
        self.rejected_bouts = rejected_bouts
        print len(self.bout_data)
# Now create flags for each bout.
#bout_number is the id of the bout, bout_frame is which frame it occurs in in the ir only movies. 

        for bout_number, bout_frame in enumerate(filtered_bout_frames):
            bout = [bout_frame, bout_frame + self.bout_durations[bout_number]]
            flags = []
            eye_sum = [a1 + a2
                       for a1, a2
                       in zip(self.fishdata.phileft[bout[0]:bout[1]],
                              self.fishdata.phiright[bout[0]:bout[1]])]
            end_z = np.median(
                self.fishdata.low_res_z[bout[1]:bout[1] + self.refract])
            start_z = np.median(
                self.fishdata.low_res_z[bout[0]-self.refract:bout[0]])
            total_z = end_z - start_z
            end_pitch = np.nanmedian(
                self.fishdata.pitch[bout[1]:bout[1] + self.refract])
            start_pitch = np.nanmedian(
                self.fishdata.pitch[bout[0]-self.refract:bout[0]])
            delta_pitch = end_pitch - start_pitch
            av_vel = np.mean(np.array(self.fishdata.vectV)[bout])
            ha_sum = np.sum(ha_diffs[bout[0]:bout[1]])
            if post_fluor_inds:
                closest_fl = min(post_fluor_inds,
                                 key=lambda x: abs(x-bout_frame))
                gutflag = self.fluor_data.gut_values[
                    post_fluor_inds.index(closest_fl)]
            else:
                gutflag = float('nan')


# have to get inter bout from raw bout data. but this has to be a time, not a frame diff.

            for key in sorted([int(k) for k in self.flag_dict.keys()]):
                if self.flag_dict[str(key)] == 'Bout ID':
                    flags.append(bout_number)
                elif self.flag_dict[str(key)] == 'Interbout':
                    flags.append(filtered_interbout[bout_number])
                elif self.flag_dict[str(key)] == 'Eye Angle Sum':
                    flags.append(np.mean(eye_sum))
                elif self.flag_dict[str(key)] == 'Average Velocity':
                    flags.append(av_vel)
                elif self.flag_dict[str(key)] == 'Total Z':
                    flags.append(total_z)
                elif self.flag_dict[str(key)] == 'Total Pitch Change':
                    flags.append(delta_pitch)
                elif self.flag_dict[str(key)] == 'Total Yaw Change':
                    flags.append(ha_sum)
                elif self.flag_dict[str(key)] == 'Fluorescence Level':
                    flags.append(gutflag)
                elif self.flag_dict[str(key)] == 'Bout Duration':
                    flags.append(self.bout_durations[bout_number])
            bout_flags.append(flags)
        self.bout_flags = bout_flags
        # a bout should be continuous over inter-bout
        # a bout should not have a fluor frame inside of it

#    def plot_flags(self, flagnum):
    # ADD TICK MARKS THAT INDICATE BIN BOUNDARIES. within tick marks use same colormap as spectral embedding            
# these two metrics rely on all other metrics in xy and xz to be non nan.
    def assure_nonans(self, bout):
        vv = self.fishdata.vectV
        ta = self.fishdata.tailangle
        pitch = self.fishdata.pitch
        prop1 = np.isfinite(ta[bout[0]:bout[1]+1]).all()
        prop2 = np.isfinite(pitch[bout[0]:bout[1]+1]).all()
        prop3 = np.isfinite(vv[bout[0]:bout[1]+1]).all()
        return prop1 and prop2 and prop3

    def bout_detector(self):
        def boutfilter_recur(boutstarts, boutends):
            for ind, interbout in enumerate(np.diff(boutstarts)):
                if interbout < self.refract:
                    del boutstarts[ind+1]
                    del boutends[ind+1]
                    break
                if interbout < boutends[ind] - boutstarts[ind]:
                    del boutstarts[ind+1]
                    del boutends[ind+1]
                    break
            if ind+1 < len(boutstarts)-1:
                boutstarts, boutends = boutfilter_recur(
                    boutstarts, boutends)
            return boutstarts, boutends

        threshstd = 5
        tailang = [tail[-1] for tail in self.fishdata.tailangle]
        ta_std = [np.abs(np.nanstd(tw)) for tw in sliding_window(5, tailang)]
        bts = scipy.signal.argrelmax(np.array(ta_std), order=3)[0]
        bts = [b for b in bts if ta_std[b] > threshstd]
        boutstarts = []
        boutends = []

# DO want to do boutfilter_recur here, but over a very small window (i.e. you don't want extreme overlaps)
        for b in bts:
            winlen = 20
            backwin = ta_std[b-winlen:b]
            backwin.reverse()
            backwin = np.array(backwin)
            forwardwin = np.array(ta_std[b:b+winlen])
        #thresh here will be noise + min calculation
            std_thresh = 3
            try:
                crossback = -np.where(backwin < std_thresh)[0][0] + b
            except IndexError:
                crossback = b - 5
            try:
                crossforward = np.where(forwardwin < std_thresh)[0][0] + b
            except IndexError:
                crossforward = b + 5
            boutstarts.append(crossback)
            boutends.append(crossforward)
        boutstarts, boutends = boutfilter_recur(boutstarts, boutends, 3)
        bout_durations = [
            be - bs for bs, be in zip(boutstarts, boutends)]
        
        self.bout_frames = boutstarts
        self.bout_durations = bout_durations
        fig, (ax1, ax2) = pl.subplots(1, 2, sharex=True, figsize=(7, 7))
        ax1.plot(bts, np.zeros(len(bts)), marker='.', color='b')
        ax2.plot(bts, np.zeros(len(bts)), marker='.', color='b')
        ax1.plot(
            boutstarts, np.zeros(len(boutstarts)), marker='.', color='m')
        ax2.plot(
            boutstarts, np.zeros(len(boutstarts)), marker='.', color='m')
        ax1.plot(boutends, np.zeros(len(boutends)), marker='.', color='k')
        ax2.plot(boutends, np.zeros(len(boutends)), marker='.', color='k')
        ax1.plot(ta_std)
        ax2.plot(tailang)
        pl.show()


    def bout_detector_old(self, plotornot, bout_type):

        def boutfilter_recur(boutlist, winlen):
            for ind, boutwin in enumerate(sliding_window(2, boutlist)):
                if boutwin[1] - boutwin[0] < winlen:
                    del boutlist[ind+1]
                    break
            if ind+1 < len(boutlist)-1:
                boutlist = boutfilter_recur(boutlist, winlen)
            return boutlist
            
        # Start orientation_based bout calls.
        std_threshold = 5

# consecutive splits lists into chunks depending on whether they are contiuously monotonic.
# e.g. [1,2,3,5,6] => [1,2,3],[5,6]
        def consecutive(data, stepsize=1):
            return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
        
        ha_winlength = 20
        yaw_all = unit_to_angle(
            filter_uvec(
                ang_to_unit(self.fishdata.headingangle), 1))
        ha_windows = sliding_window(ha_winlength,
                                    yaw_all)
        kernel = np.concatenate((np.full(ha_winlength / 2, -1),
                                 np.full(ha_winlength / 2, 1)), axis=0)
        orientation_track = []
        for hw in ha_windows:
            kernel_sum = np.abs(np.sum(np.array(hw) * kernel))
            orientation_track.append(kernel_sum)
        orientation_track = np.full(
            kernel.shape[0] / 2, np.nan).tolist() + orientation_track
        o_max_inds = scipy.signal.argrelmax(
            np.array(orientation_track))[0]
        bouts_orientation = [om
                             for om in o_max_inds
                             if orientation_track[om] > 100]
        pl.plot(bouts_orientation, np.zeros(len(bouts_orientation)),
                marker='.', ms=15, color='g')
        pl.plot(yaw_all, 'b')
        pl.plot(orientation_track, 'r')
        pl.title('Bouts from Orientation')
        tailanglesum = [tail[-1] for tail in self.fishdata.tailangle]
        tailangle_std = []
        var_inds = []
        var_win_len = 5
        for varind, tailwin in enumerate(
                sliding_window(var_win_len, tailanglesum)):
            tailangle_std.append(np.abs(np.nanstd(tailwin)))
        tailangle_std = np.array(tailangle_std)
        noise = np.nanmedian([tailangle_std[ind]
                              for ind in scipy.signal.argrelmin(
                                      tailangle_std, order=10)[0]])
        print('NOISE')
        print noise
        for ind, varwindow in enumerate(tailangle_std):
            if varwindow > std_threshold*noise:
                var_inds.append(ind)
        var_inds = [np.ceil(var_win_len / 2).astype(np.int) + v for v in var_inds]
        consecutive_slices = consecutive(var_inds)
# consecutive chunks the variance indices into continuous runs. allows you to keep only
# the first. 
# have to be at least 3 consecutive frames of high var windows for a bout (i.e. ~50 ms long)
        consecutive_thresh = 3

# this function will be used if you want to use 10 frame discrete windows        
        def assign_fixedwin_bouts(consecutive_slices, consecutive_thresh):
            consecutive_thresh = self.refract
            bouts_tail = []
            for cs in consecutive_slices:
                if cs.shape[0] > consecutive_thresh:
                    bouts_in_slice = range(
                        1 + (cs.shape[0] / self.minboutlength))
                    for b in bouts_in_slice:
                        bouts_tail.append(cs[b * self.minboutlength])
            return bouts_tail

        bouts_tail = [a[0]
                      for a in consecutive_slices
                      if a.shape[0] >= consecutive_thresh]

        self.bout_durations = [b.shape[0]
                               for b in consecutive_slices
                               if b.shape[0] >= consecutive_thresh]
        
        if bout_type == 'tail':
            self.bout_frames = bouts_tail
# #            self.bout_frames = boutfilter_recur(
# #               copy.deepcopy(bouts_tail), self.minboutlength)
#         elif bout_type == 'orientation':
#  #           self.bout_frames = boutfilter_recur(
# #                copy.deepcopy(bouts_orientation))
#             self.bout_durations = np.full(len(self.bout_frames), np.nan)
#         elif bout_type == 'combined':
#             self.bout_frames = boutfilter_recur(
#                 sorted(
#                     copy.deepcopy(
#                         bouts_tail) + copy.deepcopy(bouts_orientation)),
#                 self.minboutlength)
#             temp_bd = []
#             tailcounter = 0
#             for bf in self.bout_frames:
#                 if bf in bouts_tail:
#                     temp_bd.append(self.bout_durations[tailcounter])
#                     tailcounter += 1
#                 else:
#                     temp_bd.append(np.nan)
#             self.bout_durations = temp_bd

        print("Bouts from Bout Detector")
        print(len(self.bout_durations))

        if plotornot:
            fig2 = pl.figure()
            ax2 = fig2.add_subplot(111)
            ax2.plot(tailanglesum)
            ax2.plot(
                bouts_tail,
                np.zeros(len(bouts_tail)),
                marker='.',
                ms=15,
                color='g')
            ax2.plot(
                bouts_orientation,
                5*np.ones(len(bouts_orientation)),
                marker='.',
                ms=15,
                color='k')
            ax2.plot(
                self.bout_frames,
                10*np.ones(len(self.bout_frames)),
                marker='.',
                ms=15,
                color='m')
            ax2.plot(tailangle_std, color='r')
            pl.show()
            print('Checking Bout Frames')
            print(self.bout_frames[-20:])
            print(len(self.bout_frames))

# Above are all bout related functions. Here are para related functions. 

    def para_during_hunt(self, index, movies, hunt_wins):
        cv2.destroyAllWindows()
        showstats = False
        pl.ioff()
        directory = self.directory + '/'
        init_frame = self.bout_frames[hunt_wins[index][0]]
        abort_frame = self.bout_frames[hunt_wins[index][1]]
        integ_window = self.integration_window
        post_frames = self.integration_window
        window = [init_frame - self.para_continuity_window - integ_window,
                  abort_frame + post_frames]
        if window[0] < 0:
            return False
        self.paradata = return_paramaster_object(window[0],
                                                 window[1],
                                                 movies,
                                                 directory, showstats,
                                                 self.para_continuity_window)
# This establishes a setup where para_continuity_window is used for correlation, and integ_window frames before the first bout are kept for wrth. so the framewindow adds para_continuity_window to the 3D para coords so that you don't map 10 seconds backwards, but only 500 ms backwards.         
        self.framewindow = [window[0] + self.para_continuity_window, window[1]]
        self.map_bouts_to_heading(index, hunt_wins)
        self.map_para_to_heading(index)
        return True

    def watch_hunt(self, dim, cont_side, delay, h_ind):
        hunt_wins = dim.hunt_wins
        print('Hunt # ' + str(h_ind))
        print('Init Cluster: ' + str(
            dim.cluster_membership[hunt_wins[h_ind][0]]))
        self.current_hunt_ind = h_ind
        firstbout, lastbout = hunt_wins[h_ind]
        ind1 = self.bout_frames[firstbout]
        ind2 = self.bout_frames[lastbout+1]
        if np.mean(
                self.fishdata.z[ind1:ind1+10]) > 1600 and np.mean(
                    self.fishdata.pitch[ind1:ind1+10]) > 0:
            print("Hunting on Ceiling")
#            return False
        print("Starts At Frame " + str(ind1))
        print("Ends At Frame " + str(ind2))
        dirct = self.directory + '/'
        if cont_side == 0:
            vid = imageio.get_reader(dirct + 'conts.AVI', 'ffmpeg')
        elif cont_side == 1:
            vid = imageio.get_reader(dirct + 'top_contrasted.AVI', 'ffmpeg')
        elif cont_side == 2:
            vid = imageio.get_reader(dirct + 'side_contrasted.AVI', 'ffmpeg')
        else:
            print('unspecified stream')
            return False
#        cv2.namedWindow('vid', flags=cv2.WINDOW_NORMAL)
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        for i in range(ind1-30, ind2):
            im = vid.get_data(i)
            im = cv2.resize(im, (700, 700))
            cv2.imshow('vid', im)
            keypress = cv2.waitKey(delay)
            if keypress == 50:
                cv2.destroyAllWindows()
                vid.close()
                return True
            elif keypress == 32:
                cv2.destroyAllWindows()
                vid.close()
                return False
        cv2.destroyAllWindows()
        vid.close()

        cv2.namedWindow(
            'Enter 1 for Full Characterization',
            flags=cv2.WINDOW_NORMAL)
        cv2.moveWindow('Enter 1 for Full Characterization', 20, 20)
        key = cv2.waitKey(0)
        print key
        # This is equivalent to pressing 1.
        if key == 49:
            ret = self.para_during_hunt(h_ind, True, hunt_wins)
            cv2.destroyAllWindows()
            if ret:
                self.paradata.watch_event(0)
        cv2.destroyAllWindows()
        return True

# Eye1 is the eye on the side of the direction of the turn.
    def bout_stats(self, bout_ind, dim, global_bout):
        if not global_bout:
            bout_index = bout_ind + dim.hunt_wins[self.current_hunt_ind][0]
        else:
            bout_index = bout_ind
        bout = self.bout_data[bout_index]
        num_cols = int(np.ceil(self.num_dp/3.0))
        num_rows = 3
        fig = pl.figure(figsize=(8, 8))
        palette = np.array(sb.color_palette("Set1", self.num_dp))
        cmem = dim.cluster_membership[bout_index]
        print('Cluster Membership: ' + str(cmem))
        for i in range(self.num_dp):
            bout_partition = partition(self.num_dp, bout)
            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.set_title(self.bout_dict[str(i)])
            ax.plot([val[i] for val in bout_partition], color=palette[i])
            if self.bout_dict[str(i)][0:3] == 'Eye':
                ax.set_ylim([-30, 30])
            elif self.bout_dict[str(i)][0:4] == 'Vect':
                ax.set_ylim([0, 3])
            # elif self.bout_dict[str(i+1)][0:5] == 'Pitch':
            #     ax.set_ylim([-50, 50])
            elif self.bout_dict[str(i)] == 'Delta Z':
                bout_partition = partition(self.num_dp, bout)
                ax.plot(np.cumsum([val[i]
                                   for val in bout_partition]), color='k')
            elif self.bout_dict[str(i)] == 'Delta Yaw':
                bout_partition = partition(self.num_dp, bout)
                ax.plot(np.cumsum([val[i]
                                   for val in bout_partition]), color='k')
        pl.suptitle('Current Bout Data')
        pl.subplots_adjust(top=0.9, hspace=.4, wspace=.4)
        pl.show()

    def manual_fix(self, rec):
        fix = self.paradata.manual_fix(rec)
        if fix == 1:
            self.map_para_to_heading(self.current_hunt_ind)

    def manual_match(self):
        self.paradata.find_misses()
        ret = self.paradata.manual_match()
        if ret == 1:
            self.paradata.make_3D_para()
            self.paradata.label_para()
            self.map_para_to_heading(self.current_hunt_ind)


# up to you to assign proper xy record only. could be that xy rec is free or
# already part of an xyz record.
# if the latter, you will merge xyz records using
# manual join
# scenarios:  1) xy rec is alone. share timestamps, share lengths
#             2) xy rec is part of an xyzrec and para
#                and goes to top before or after it
#             3) xy rec is part of an xyzrec and goes to roof internally

# syntax is always myexp.paradata.all_xy[xyrec].timestamp or location
# up to you to figure out proper fill in timestamps according to 1,2,3 above

    def assign_max_z(self, xyrec, auto, zmax, *xzrec):
        if not auto:
            timestamp = raw_input("Enter Timestamp for New XZ: ")
            timestamp = int(eval(timestamp))
            rec_len = raw_input("Enter length of New XZ: ")
            rec_len = int(eval(rec_len))
        else:
            if xzrec == ():
                timestamp = self.paradata.all_xy[xyrec].timestamp
                rec_len = len(self.paradata.all_xy[xyrec].location)
            else:
                xzrec = xzrec[0]
                f_or_b = raw_input(
                    "Enter f for max_z ahead of known xzrec, b for behind: ")
                if f_or_b == 'f':
                    timestamp = self.paradata.all_xz[
                        xzrec].timestamp + len(
                            self.paradata.all_xz[xzrec].location)
                    rec_len = self.paradata.all_xy[
                        xyrec].timestamp + len(
                            self.paradata.all_xy[xyrec].location) - timestamp
                elif f_or_b == 'b':
                    timestamp = self.paradata.all_xy[xyrec].timestamp
                    rec_len = len(
                        self.paradata.all_xy[xyrec].location) - len(
                            self.paradata.all_xz[xzrec].location)
                else:
                    print("invalid entry")
                    return

        t_window = (int(timestamp), int(timestamp) + int(rec_len))
        self.paradata.assign_z(xyrec, t_window, zmax)
        self.paradata.watch_event(1)
        accept = raw_input("Accept Fix?: ")
        if accept == 'n':
            new_maxz = raw_input("Enter New Max Z: ")
            new_maxz = int(new_maxz)
            del self.paradata.all_xz[-1]
            del self.paradata.xyzrecords[-1]
            if xzrec != ():
                return self.assign_max_z(xyrec, auto, new_maxz, xzrec)
            else:
                return self.assign_max_z(xyrec, auto, new_maxz)
        elif accept == 'y':
            self.map_para_to_heading(self.current_hunt_ind)
            return

    def manual_remove(self):
        rec = raw_input('Enter Record to Remove: ')
        rec = int(rec)
        for xyz_pair in self.paradata.xyzrecords[rec]:
            xy_id = xyz_pair[0]
            xz_id = xyz_pair[1]
            xypara_obj = self.paradata.all_xy[xy_id]
            xzpara_obj = self.paradata.all_xz[xz_id]
            xy_coords = [(np.nan, np.nan) for i in range(
                self.paradata.framewindow[0], self.paradata.framewindow[1])]
            inv_y = [(x, 1888-y) for (x, y) in xypara_obj.location]
            xy_coords[xypara_obj.timestamp:xypara_obj.timestamp+len(
                xypara_obj.location)] = inv_y
            self.paradata.unpaired_xy.append((xy_id, xypara_obj, xy_coords))
            xz_coords = [(np.nan, np.nan) for i in range(
                self.paradata.framewindow[0], self.paradata.framewindow[1])]
            inv_z = [(x2, 1888-z) for (x2, z) in xzpara_obj.location]
            xz_coords[xzpara_obj.timestamp:xzpara_obj.timestamp+len(
                xzpara_obj.location)] = inv_z
            self.paradata.unpaired_xz.append((xz_id, xzpara_obj, xz_coords))
            del self.paradata.xyzrecords[rec]
        self.paradata.make_3D_para()
        self.paradata.clear_frames()
        self.paradata.label_para()
        self.map_para_to_heading(self.current_hunt_ind)

    def manual_join(self):
        rec1 = raw_input('Enter First Record:  ')
        rec2 = raw_input('Enter Second Record: ')
        self.paradata.manual_join(rec1, rec2)
        self.map_para_to_heading(self.current_hunt_ind)
        
    def create_unit_vectors(self):
        ufish_list = []
        uperp_list = []
        ufish_origin = []
        upar_list = []
        filter_sd = 1
        pitch_all = gaussian_filter(self.fishdata.pitch,
                                    filter_sd)
        pitch_all = np.radians(pitch_all)
        yaw_all = unit_to_angle(
            filter_uvec(
                ang_to_unit(self.fishdata.headingangle), 1))
        yaw_all = np.radians(yaw_all)
        x_all = gaussian_filter(self.fishdata.x, filter_sd)
        y_all = gaussian_filter(self.fishdata.y, filter_sd)
        z_all = gaussian_filter(self.fishdata.z, filter_sd)
        for frame in range(len(x_all)):
            yaw = yaw_all[frame]
            pitch = pitch_all[frame]
            x = x_all[frame]
            y = y_all[frame]
            z = z_all[frame]
            planepoint, ufish, u_par, u_perp = fishxyz_to_unitvecs(
                [x, y, z], yaw, pitch)
            ufish_origin.append(planepoint)
            ufish_list.append(ufish)
# this is a vector pointing out of the left eye when parallel to the body.
            upar_list.append(u_par)
            uperp_list.append(u_perp)
        self.ufish = ufish_list
        self.uperp = uperp_list
        self.upar = upar_list
        self.ufish_origin = ufish_origin
# this will yield a vector normal to the shearing plane of the fish.

    def map_bouts_to_heading(self, h_index, hunt_wins):
        firstbout, lastbout = hunt_wins[h_index]
        bout_frames = [self.bout_frames[i] for i in range(
            firstbout, lastbout+1)]
        bout_durations = [self.bout_durations[i] for i in range(
            firstbout, lastbout+1)]
        bout_az = []
        bout_alt = []
        bout_dist = []
        # make this a member variable b/c you also use it for para velocity and flags. did this. it's called parawin. make sure it is also used in flags. if its not, make sure you use the same number. 
        for b_dur, bf in zip(bout_durations, bout_frames):
            ufish = self.ufish[bf]
            upar = self.upar[bf]
            uperp = self.uperp[bf]
            origin_start = self.ufish_origin[bf]
#            origin_end = self.ufish_origin[bf+self.minboutlength]
            origin_end = self.ufish_origin[bf + b_dur]
            # here put new pmap to fish function
            azimuth, altitude, nb_mag, nb_wrt_heading, ang3d = p_map_to_fish(
                ufish, origin_start, uperp, upar, origin_end, 0)
            bout_dist.append(nb_mag)
            bout_az.append(azimuth)
            bout_alt.append(altitude)
        self.bout_dist = bout_dist
        self.bout_az = bout_az
        self.bout_alt = bout_alt
        
    def map_para_to_heading(self, h_index):
        para_wrt_heading = []
        wrth_xy = []
        for frame in range(self.framewindow[0], self.framewindow[1]):
            if frame % 10 == 0:
                print(frame)
            p_frame = frame - self.framewindow[
                0] + self.para_continuity_window
            eye_x = self.fishdata.x[frame]
            eye_y = self.fishdata.y[frame]
            temp_xypara = []
            # xy para coords will require you to start at p_frame
            for xyp in self.paradata.unpaired_xy:
                wrt_fish_xy = np.array(
                    [xyp[2][p_frame][0] - eye_x,
                     xyp[2][p_frame][1] - eye_y])
                az_xy = np.arctan(wrt_fish_xy[1] / wrt_fish_xy[0])
                mag_xy = magvector(wrt_fish_xy)
                temp_xypara.append([wrt_fish_xy[0],
                                    wrt_fish_xy[1],
                                    np.nan,
                                    xyp[0],
                                    mag_xy,
                                    np.nan,
                                    az_xy,
                                    np.nan])


#  0: x
#  1: y 
#  2: z
#  3: ID
#  4: mag of vector to para from fish head
#  5: 3D angle via 3D dot product to para
#  6: azimuth angle in 2D
#  7: altitude angle in 2D 
                            
            ufish = self.ufish[frame]
            u_par = self.upar[frame]
            u_perp = self.uperp[frame]
            ufish_origin = self.ufish_origin[frame]

# this is a vector pointing out of the left eye when parallel to the body.
# this will yield a vector normal to the shearing plane of the fish.
# shearing plane is used for rotation of eyes about the plane. this is for when you want to map the stimuli to the individual eyes. 
            # shearingplane = Plane(
            #     Point3D(ufish_origin[0], ufish_origin[1], ufish_origin[2]),
            #     normal_vector=(int(u_perp[0] * 100), int(u_perp[1] * 100), int(
            #         u_perp[2] * 100)))
# normal vector input to a Plane object must be an int. if you don't multiply by 100, decimals are meaningless and just get rounded to 1 by int()
            temp_plist_h = []
            for par_index in range(0,
                                   self.paradata.para3Dcoords.shape[0], 3):
                para_xyz = self.paradata.para3Dcoords[
                    par_index:par_index + 3, p_frame]
                azimuth, altitude, dist, nb_wrt_heading, ang3d = p_map_to_fish(
                    ufish, ufish_origin, u_perp, u_par, para_xyz, par_index)
                nb_wrt_heading.append(dist)
                nb_wrt_heading.append(ang3d)
                nb_wrt_heading.append(azimuth)
                nb_wrt_heading.append(altitude)
                temp_plist_h.append(nb_wrt_heading)
                
            para_wrt_heading.append(temp_plist_h)
            wrth_xy.append(temp_xypara)

        np.save(self.directory + '/para3D' + str(h_index).zfill(2) + '.npy',
                self.paradata.para3Dcoords)
        np.save(self.directory + '/wrth' + str(h_index).zfill(2) + '.npy',
                para_wrt_heading)
        np.save(self.directory + '/wrth_xy' + str(h_index).zfill(2) + '.npy',
                wrth_xy)
        np.save('/Users/nightcrawler2/ufish.npy',
                self.ufish[self.framewindow[0]:self.framewindow[1]])
        np.save('/Users/nightcrawler2/uperp.npy',
                self.uperp[self.framewindow[0]:self.framewindow[1]])
        np.save('/Users/nightcrawler2/ufish_origin.npy',
                self.ufish_origin[self.framewindow[0]:self.framewindow[1]])
        np.save('/Users/nightcrawler2/para_continuity_window.npy',
                np.array(self.para_continuity_window))


    def exporter(self):
        with open(self.directory + '/master.pkl', 'wb') as file:
            pickle.dump(self, file)

# arrays will tolerate 2 missing frames only.

# First go from spherical to xyz within the fish basis. Next multiply each xyz coord by the basis vectors. Add to previous XYZ coordinate.


def fishxyz_to_unitvecs(fish_xyz, yaw, pitch):
    planepoint = np.array([fish_xyz[0], fish_xyz[1], fish_xyz[2]])
    ufish = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) *
                      np.cos(pitch), np.sin(pitch)])
    u_par = np.array(
        [np.cos(yaw + np.pi / 2),
         np.sin(yaw + np.pi / 2), 0])
    u_perp = np.cross(ufish, u_par)
    return planepoint, ufish, u_par, u_perp
# this is a vector pointing out of the left eye when parallel to the body.
    

def sphericalbout_to_xyz(az, alt, dist, uf, u_perp, u_par):
    z_in_basis = dist*np.sin(alt)
    x_in_basis = dist*np.cos(alt)*np.cos(az)
    y_in_basis = dist*np.cos(alt)*np.sin(az)
    dx = uf * x_in_basis
    dy = -u_par * y_in_basis
    dz = u_perp * z_in_basis
    xyzvec = dx + dy + dz
    return xyzvec



# transforms para to new basis of fish
                

def p_map_to_fish(uf, uf_origin, u_prp, u_paral, p_xyz, p_index):
    wrt_fish = np.array(
        [p_xyz[0] - uf_origin[0],
         p_xyz[1] - uf_origin[1],
         p_xyz[2] - uf_origin[2]])
    new_basis_wrt_heading = [np.dot(wrt_fish, uf),
                             np.dot(wrt_fish, -u_paral),
                             np.dot(wrt_fish, u_prp),
                             p_index / 3]
    nb_mag = magvector(new_basis_wrt_heading)
    para_vector_unit = new_basis_wrt_heading[0:3] / nb_mag
    angle_to_para_3D = np.arccos(np.dot(para_vector_unit, uf))
    altitude = np.arcsin(para_vector_unit[2])
# hypotenuse is the cos of the height of the vector. y coord is the opp
    azimuth = np.arctan(para_vector_unit[1] / para_vector_unit[0])
    if para_vector_unit[0] < 0:
        if azimuth < 0:
            azimuth = np.pi + azimuth
        elif azimuth > 0:
            azimuth = -np.pi + azimuth
    return azimuth, altitude, nb_mag, new_basis_wrt_heading, angle_to_para_3D
    

# 1. Make sure this is correct.
# 2. Add a switch here for fill_all. If fill_all is True, simply replace a nan with the value before it.

# Linear interpolation over all nanwindows

def fill_in_nans(arr):
    nanstretch = False
    nan_start_ind = 0
    nan_end_ind = 0
    nan_windows = []
    if math.isnan(arr[0]):
        for i in arr:
            if not math.isnan(i):
                arr[0] = i
                break
    arr_out = np.copy(arr)
    for ind, j in enumerate(arr):
        if math.isnan(j) and not nanstretch and ind > 0:
            if not math.isnan(arr[ind-1]):
                nan_start_ind = ind
                nanstretch = True
        elif nanstretch and not math.isnan(j):
            nan_end_ind = ind
            nan_windows.append([nan_start_ind, nan_end_ind])
            nanstretch = False

    for win in nan_windows:
        arr_out[win[0]:win[1]] = np.linspace(
            arr[win[0]-1],
            arr[win[1]],
            win[1]-win[0]).astype(np.int)
    return arr_out


def fill_in_nans_prevcoord(arr):
    fill_2 = False
    fill_all = True
    rebuilt_array = []
    rebuilt_array.append(arr[0])
    windows = sliding_window(3, arr)
    for win in windows:
        finite = np.isfinite(win)
        if finite[0] and finite[2] and not finite[1]:
            if type(win[0]) == np.float64:
                rebuilt_array.append(np.mean([win[0], win[2]]))
            elif type(win[0]) == np.int:
                rebuilt_array.append(np.mean([win[0], win[2]])).astype(np.int)
        else:
            rebuilt_array.append(win[1])
    rebuilt_array.append(arr[-1])
    rb2_array = []
    windows2 = sliding_window(2, rebuilt_array)
    if fill_all:
        rb2_array.append(rebuilt_array[0])
        for win2 in windows2:
            if math.isnan(win2[1]):
                rb2_array.append(rb2_array[-1])
            else:
                rb2_array.append(win2[1])
        return rb2_array
    else:
        return rebuilt_array

    
def create_poirec(h_index, two_or_three, directory, para_id):
    wrth = []
    poi_wrth = []
    if two_or_three == 2:
        wrth = np.load(
            directory + '/wrth_xy' + str(h_index).zfill(2) + '.npy')
    elif two_or_three == 3:
        wrth = np.load(
            directory + '/wrth' + str(h_index).zfill(2) + '.npy')
    for all_visible_para in wrth:
        rec_found = False
        for prec in all_visible_para:
            if prec[3] == para_id:
                poi_wrth.append(prec)
                rec_found = True
        if not rec_found:
            poi_wrth.append(np.full((len(prec)), np.nan))
    return poi_wrth


def plot_hairball(hd, *actionlist):
    if actionlist == ():
        actionlist = hd.actions
    else:
        actionlist = actionlist[0]
    graph_3D = pl.figure(figsize=(8, 8))
    ax = graph_3D.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_xyz = 0
    for h, p, a in zip(hd.hunt_ind_list,
                       hd.para_id_list,
                       hd.actions):
        if a not in actionlist:
            continue
        xyz_coords = []
        poi_wrth = create_poirec(h, 3, hd.directory, p)
        for prec in poi_wrth:
            xyz_coords.append(prec[0:3])
        xyz_coords = np.array(xyz_coords)
        x, y, z = xyz_coords[:, 0], xyz_coords[:, 1], xyz_coords[:, 2]
        max_x = np.nanmax(np.abs(x))
        max_y = np.nanmax(np.abs(y))
        max_z = np.nanmax(np.abs(z))
        if max_x > max_xyz:
            max_xyz = max_x
        if max_y > max_xyz:
            max_xyz = max_y
        if max_z > max_xyz:
            max_xyz = max_z
        palette = np.array(sb.color_palette("coolwarm", x.shape[0]))
        for i in range(x.shape[0] - 1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=palette[i])

    ax.set_xlim(0, max_xyz)
    ax.set_ylim(max_xyz, -max_xyz)
    ax.set_zlim(-max_xyz, max_xyz)
    ax.set_title("3D WRTH")
    pl.show()
    

def pvec_wrapper(exp, hd):
    
    def concat_vecs(veclist):
        all_lengths = map(lambda a: len(a), veclist)
        all_vecs = reduce(lambda a, b: np.concatenate([a, b]), veclist)
        return all_lengths, all_vecs

    vecs = []
    vec_address = []
    nonan_indices = []
    p_id = 0
    for h in hd.hunt_ind_list:
        print h
        while True:
            try:
                v, no_nan_inds, spacing = para_vec_plot(exp, h, p_id, 0)
                nonan_indices.append(no_nan_inds)
            except IndexError:
                p_id = 0
                break
            vec_address.append([h, p_id])
            vecs.append(v)
            p_id += 1
#    all_l, all_v = concat_vecs(vecs)
    np.save('vec_address.npy', vec_address)
    np.save('input.npy', vecs)
    np.save('no_nan_inds.npy', nonan_indices)
    np.save('spacing.npy', spacing)
    return vecs, vec_address
        

def para_vec_plot(exp, h_id, p_id, animate):
    penv = ParaEnv(h_id, exp.directory)
    penv.find_paravectors(False)
    pvec = []
    non_nan_indices = []
    dp = []
    # what you might have to do here is define non-nan bounds. only
    #incorporate non-nan stretches. pomegranate doesn't take nans.
    p_index = 0
    for dot, vec, vel in zip(
            penv.dotprod[p_id],
            penv.paravectors[p_id][0][1:],
            penv.velocity_mags[p_id][1:]):
            if np.isfinite(vec).all():
                pvec.append(vec)
                non_nan_indices.append(p_index)
                dp.append(dot)
            p_index += 1
    
    if animate < 1:
        return pvec, non_nan_indices, penv.vector_spacing
    graph_3D = pl.figure(figsize=(16, 8))
    ax3d = graph_3D.add_subplot(121, projection='3d')
    dp_ax = graph_3D.add_subplot(122)
    dp_ax.set_ylim([-1, 1])
    dp_ax.set_xlim([0, len(dp)])
    dp_ax.set_title('Dot Product with Previous Vector')
    ax3d.set_title('Velocity Vectors')
    ax3d.set_xlabel('dx')
    ax3d.set_ylabel('dy')
    ax3d.set_zlabel('dz')
    cmap = pl.get_cmap('seismic')
    norm = Normalize(vmin=0, vmax=len(pvec))
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba_vals = scalarMap.to_rgba(range(len(pvec)))
#    print('Length Pvec')
#    print(len(pvec))
    #len paravectors is how many para are in the environment during the hunt. NOT the number of coordinates of a given para.
    pl.hold(True)
    pl.ioff()
    if animate == 1:
        for i in range(len(pvec)):
            print i
            print pvec[i]
            ax3d.plot([pvec[i][0]],
                      [pvec[i][1]],
                      [pvec[i][2]],
                      color=rgba_vals[i],
                      marker='.')
        scalarMap.set_array(range(len(penv.paravectors[p_id][0])))
        graph_3D.colorbar(scalarMap)
        dp_ax.plot(dp)
        pl.show()

    elif animate == 2:
        ax3d.set_xlim([-15, 15])
        ax3d.set_ylim([-15, 15])
        ax3d.set_zlim([-15, 15])
# dot product entry 1 is the dp of the first and second vec.
# it is a property of the two vecs, but you are plotting vec vs. time.
# if you stagger dot by 1, the dot is a property of the current vector. 
        def updater(num, plotlist):
            if num < 1:
                return plotlist
            x = [p[0] for p in pvec[0:num]]
            y = [p[1] for p in pvec[0:num]]
            z = [p[2] for p in pvec[0:num]]
            dp_x = range(num)
            dp_y = dp[0:num]
            if np.isfinite([x, y, z]).all():
                plotlist[0].set_data(x, y)
                plotlist[0].set_3d_properties(z)
                plotlist[1].set_data(dp_x, dp_y)
            return plotlist
        
        plot3d, = ax3d.plot(xs=[], ys=[], zs=[],
                            ls='-', color=[.3, .6, .8], linewidth=.75,
                            marker='o', ms=10, markevery=[-1])
        dp_plot, = dp_ax.plot([], [], linewidth=1.0)
        
        p3d_list = [plot3d, dp_plot]
        line_ani = anim.FuncAnimation(
            graph_3D,
            updater,
            len(pvec),
            fargs=[p3d_list],
            interval=200,
            repeat=True,
            blit=False)
#        line_ani.save('test.mp4')
        pl.show()
    return pvec, non_nan_indices, penv.vector_spacing

        


    
def para_state_plot(hd, exp):
    hunt_inds = hd.hunt_ind_list
    poi = hd.para_id_list
    dps = []
    vels = []
    c_pallete = np.array(sb.color_palette('husl', len(hunt_inds)))
    for hunt, p in zip(hunt_inds, poi):
        penv = ParaEnv(hunt, exp.directory)
        penv.find_paravectors(False, p)
        dps.append(np.copy(penv.dotprod))
        vels.append(np.copy(penv.velocity_mags))
    loopcount = 0
    avg_dp = []
    continuity_window = exp.para_continuity_window
    dvplot = pl.figure(figsize=(8, 8))
    ax_dp = dvplot.add_subplot(211)
    ax_vel = dvplot.add_subplot(212)
    for dp, vel in zip(dps, vels):
        avg_dp.append(dp[0][-100:])
        ax_dp.plot(dp[0][continuity_window:], color=c_pallete[loopcount])
        ax_vel.plot(vel[0][continuity_window:], color=c_pallete[loopcount])
        loopcount += 1
    ax_dp.set_title("Para Dot Products")
    ax_vel.set_title("Para Velocities")
    pl.show()
    if len(poi) > 1:
        sb.tsplot(avg_dp, ci=95)
        pl.show()
    if len(dps) == 1:
        return dps[0][0][continuity_window:], vels[0][0][continuity_window:]

    #     int_win = 35
    #     prt = 5
    #     az = [w[-2] for w in poi_wrth]
    #     alt = [w[-1] for w in poi_wrth]
    #     ax_az.plot(gaussian_filter(az, 1), color=c_pallete[loopcount])
    #     ax_alt.plot(gaussian_filter(alt, 1), color=c_pallete[loopcount])
    #    

    # ax_az.set_title("Delta Azimuth")
    # ax_alt.set_title("Delta Altitude")
    # ax_az.set_xlabel("Frames")
    # ax_alt.set_xlabel("Frames")
    # ax_az.set_ylabel("Radians")
    # ax_alt.set_ylabel("Radians")
    # pl.show()


def csv_data(headers, datavals, file_id, directory):
    with open(directory + '/' + file_id + '.csv', 'wb') as csvfile:
        output_data = csv.writer(csvfile)
        output_data.writerow(headers)
        for dt in datavals:
            output_data.writerow(dt)


def all_data_to_csv(directories):
    with open(
            '/Users/nightcrawler2/PreycapMaster/all_huntbouts.csv',
            'wb') as csvfile:
        output_data = csv.writer(csvfile)
        for d in directories:
            with open(
                    '/Users/nightcrawler2/PreycapMaster/'
                    + d + '/huntingbouts.csv',
                    'rb') as huntbouts:
                reader = csv.reader(huntbouts)
                for ind, row in enumerate(reader):
                    if ind != 0:
                        output_data.writerow(row)

    
            
# VERY EASY TO UPDATE THIS TO TAKE ALL 

# This function will tell you, for all hunts, the typical trajectory of eye convergence and Z traversal. 
            
def huntbouts_wrapped(hd, dim, exp, med_or_min, plotornot):
    zstack = []
    philstack = []
    phirstack = []
    for h in hd.hunt_ind_list:
        yaw, pitch, z, phil, phir = bouts_during_hunt(h, dim, exp, plotornot)
        zstack.append(z)
        philstack.append(phil)
        phirstack.append(phir)
    if med_or_min == 0:
        zstack_len = np.median(map(lambda x: len(x), zstack))
    elif med_or_min == 1:
        zstack_len = np.min(map(lambda x: len(x), zstack))
    zstack = [stack for stack in zstack if len(stack) >= zstack_len]
    zstarts = [a[:zstack_len] for a in zstack]
    zends = [zs[-zstack_len:] for zs in zstack]
    phil_starts = [p[0:60] for p in philstack]
    phil_ends = [p[-60:] for p in philstack]
    phir_starts = [p[0:60] for p in phirstack]
    phir_ends = [p[-60:] for p in phirstack]
    pl.figure()
    pl.title('Z Starts')
    sb.tsplot(zstarts, ci=95)
    pl.figure()
    pl.title('Z Ends')
    sb.tsplot(zends, ci=95)
    pl.figure()
    for ztraj in zstack:
        pl.plot(ztraj)
    pl.figure()
    sb.tsplot(phil_starts, ci=95, color='r')
    sb.tsplot(phir_starts, ci=95, color='g')
    pl.title('Hunt Initiations')
    pl.figure()
    sb.tsplot(phil_ends, ci=95, color='r')
    sb.tsplot(phir_ends, ci=95, color='g')
    pl.title('Hunt Ends')
    pl.show()


def every_huntbout(dim, exp, hd):
    for h_id in hd.hunt_ind_list:
        bouts_during_hunt(h_id, dim, exp, True)
    
    
def bouts_during_hunt(hunt_ind, dimred, exp, plotornot):
    integ_win = exp.integration_window
    firstind = dimred.hunt_wins[hunt_ind][0]
    secondind = dimred.hunt_wins[hunt_ind][1]    
    indrange = range(firstind, secondind+1, 1)
    #+1 so it includes the secondind
    print('Cluster Membership')
    print dim.cluster_membership[firstind:secondind+1]
    print('Bout Durations')
    print exp.bout_durations[firstind:secondind]
    start = exp.bout_frames[firstind]-integ_win
    end = exp.bout_frames[secondind]+integ_win
    # gives last bout 500ms to occur
    fig, ((ax1, ax2),
          (ax3, ax4)) = pl.subplots(
              2, 2,
              sharex=True,
              figsize=(7, 7))
    filt_phir = gaussian_filter(exp.fishdata.phiright, 2)[start:end]
    filt_phil = gaussian_filter(exp.fishdata.phileft, 2)[start:end]
    ax1.plot(filt_phir, color='g')
    ax1.plot(filt_phil, color='r')
    ax2.plot([t[-1] for t in exp.fishdata.tailangle[start:end]])
    bouts_tail = [exp.bout_frames[i]-start for i in indrange]
    bouts_tail_end = [
        exp.bout_frames[i]-start + exp.bout_durations[i] for i in indrange]
    print len(bouts_tail)
    ax2.plot(bouts_tail,
             np.zeros(len(bouts_tail)),
             marker='.',
             ms=10,
             color='m')
    ax2.plot(bouts_tail_end,
             np.zeros(len(bouts_tail_end)),
             marker='.',
             ms=10,
             color='c')
    for ind, typ in enumerate(dim.cluster_membership[firstind:secondind+1]):
        ax2.text(bouts_tail[ind], -.5, str(typ))
    pitch_during_hunt = exp.fishdata.pitch[start:end]
    yaw_during_hunt = exp.fishdata.headingangle[start:end]
    z_during_hunt = exp.fishdata.z[start:end]
    ax3.plot(yaw_during_hunt, color='m')
    ax3.plot(pitch_during_hunt, color='k')
    ax4.plot(z_during_hunt, color='b')
    if plotornot == 0:
        pl.clf()
        pl.close()
    elif plotornot == 1:
        pl.show()
    return pitch_during_hunt, yaw_during_hunt, z_during_hunt, filt_phil, filt_phir


def hunted_para_descriptor(dim, exp, hd):

    header = ['Hunt ID',
              'Bout Number',
              'Bout Az',
              'Bout Alt',
              'Bout Dist',
              'Bout Delta Pitch',
              'Bout Delta Yaw',
              'Para Az',
              'Para Alt',
              'Para Dist',
              'Para Az Velocity',
              'Para Alt Velocity',
              'Para Dist velocity',
              'Postbout Para Az',
              'Postbout Para Alt',
              'Postbout Para Dist',
              'Strike Or Abort',
              'Inferred'
              'Avg Para Velocity']
    int_win = exp.integration_window
    cont_win = exp.para_continuity_window
    pitch_flag = int(dim.inv_fdict['Total Pitch Change'])
    yaw_flag = int(dim.inv_fdict['Total Yaw Change'])
    bout_descriptor = []
    df_labels = ["Bout Az", "Bout Alt",
                 "Bout Dist", "Bout Delta Pitch", "Bout Delta Yaw"]
    realfish = RealFishControl(exp)

# going to make eb a bout range. eb will be a tuple. first entry is hunt start. second is hunt end.
# hunt end is a negative value. 
    
    for hi, hp, ac, br, iws, mz in zip(
            hd.hunt_ind_list,
            hd.para_id_list, hd.actions, hd.boutrange, hd.interp_windows,
            hd.infer_z):
        para3D = np.load(
            exp.directory + "/para3D" + str(hi).zfill(
                2) + ".npy")[
                    hp*3:hp*3 + 3][
                        :, cont_win+int_win-realfish.firstbout_para_intwin:]
        realfish.para_xyz_per_hunt.append(para3D)
        penv = ParaEnv(hi, exp.directory)
        penv.find_paravectors(False, hp)
        avg_vel = np.nanmean(penv.velocity_mags[0][
            exp.para_continuity_window / penv.vector_spacing:])
        hunt_df = pd.DataFrame(columns=df_labels)
        poi_wrth = create_poirec(hi, 3, exp.directory, hp)
        dist = [pr[4] for pr in poi_wrth]
        az = [pr[6] for pr in poi_wrth]
        alt = [pr[7] for pr in poi_wrth]
        filter_sd = 0
        filt_az = gaussian_filter(az, filter_sd)
        filt_alt = gaussian_filter(alt, filter_sd)
        filt_dist = gaussian_filter(dist, filter_sd)
        if len(filt_az) < 2 or len(filt_dist) < 2:
            continue
        delta_az = [b-a for a, b in sliding_window(2, filt_az)]
        delta_alt = [b-a for a, b in sliding_window(2, filt_alt)]
        delta_dist = [b-a for a, b in sliding_window(2, filt_dist)]
#instead of asking about sensory input, just asking what para is doing right before and right after bout
        hunt_bouts = range(dim.hunt_wins[hi][0],
                           dim.hunt_wins[hi][1]+1)
        hunt_bout_frames = [exp.bout_frames[i] for i in hunt_bouts]
        realfish.hunt_firstframes.append(hunt_bout_frames[0])
        # HERE YOU ALREADY HAVE THE INTERBOUT BACK FOR EACH BOUT. JUST SWITCH THEM TO
        # A CUMSUM STARTING WITH 3 OR SO. 
        realfish.hunt_interbouts.append(
            [0] + np.diff(hunt_bout_frames).tolist())
        realfish.hunt_results.append(ac)
        norm_bf = [hbf - hunt_bout_frames[0] for hbf in hunt_bout_frames]
        norm_bf = map(lambda(x): x+int_win, norm_bf)
        print('hunt_bout_frames')
        exp.map_bouts_to_heading(hi, dim.hunt_wins)
#        framewin = exp.minboutlength / 2
        framewin = exp.refract
# WANT TO ASSIGN FRAMEWIN TO BE THE REFRACTORY PERIOD.
# PUT THE REFRACTORY PERIOD INTO THE EXPERIMENT CLASS. USE IT IN BOUT_DETECTOR. 
        
        # these are normed to the hunting bout so that first bout is 0.
        endhunt = False
        for ind, bout in enumerate(hunt_bouts):
            print('true index')
            print ind
            print hunt_bouts
            if br[0] != 0:
                ind += br[0]
                bout += br[0]
                print('altered index')
                print ind

            norm_frame = norm_bf[ind]
            bout_dur = exp.bout_durations[bout]
            inferred_coordinate = 0
            if mz:
                inferred_coordinate = 2
            for infwin in iws:
                if np.intersect1d(
                        range(norm_frame-framewin, norm_frame),
                        infwin).any():
                    inferred_coordinate = 1
            #note that delta pitch and yaw are SINGULAR VALUES. not the same as the others. make norm_frame 3.
            # should be 3 because that is your REFRACTORY PERIOD. 
            delta_pitch = dim.all_flags[bout][pitch_flag]
            delta_yaw = dim.all_flags[bout][yaw_flag]
            para_az = np.nanmean(filt_az[norm_frame-framewin:norm_frame])
            para_alt = np.nanmean(filt_alt[norm_frame-framewin:norm_frame])
            para_dist = np.nanmean(filt_dist[norm_frame-framewin:norm_frame])
            para_daz = np.nanmean(delta_az[norm_frame-framewin:norm_frame])
            para_dalt = np.nanmean(delta_alt[norm_frame-framewin:norm_frame])
            para_ddist = np.nanmean(
                delta_dist[norm_frame-framewin:norm_frame])
            postbout_az = np.nanmean(
                filt_az[norm_frame+bout_dur:
                        norm_frame+bout_dur+framewin])
            postbout_alt = np.nanmean(
                filt_alt[norm_frame+bout_dur:
                         norm_frame+bout_dur+framewin])
            postbout_dist = np.nanmean(
                filt_dist[norm_frame+bout_dur:
                          norm_frame+bout_dur+framewin])

    # -1s will be strike as deconvergence. -2 to -huntlenght will be strike before deconvergence
    # will be strike that continues into hunting mode
            if br[1] < 0:
                if br[1] == -100:
                    if ind == len(hunt_bouts) - 1:
                        endhunt = True
                        last_bout = br[1]
                elif ind == len(hunt_bouts) + br[1]:
                    endhunt = True
                    last_bout = br[1]

            else:
                if ind == br[1]:
                    endhunt = True
                    last_bout = ind - len(hunt_bouts)
            bout_descriptor.append([hi,
                                    ind,
                                    exp.bout_az[ind],
                                    exp.bout_alt[ind],
                                    exp.bout_dist[ind],
                                    np.radians(delta_pitch),
                                    -1*np.radians(delta_yaw),
                                    para_az,
                                    para_alt,
                                    para_dist,
                                    para_daz,
                                    para_dalt,
                                    para_ddist,
                                    postbout_az,
                                    postbout_alt,
                                    postbout_dist,
                                    ac,
                                    inferred_coordinate,
                                    avg_vel])
#            if ind != -1:
            hunt_df.loc[ind] = [exp.bout_az[ind],
                                exp.bout_alt[ind],
                                exp.bout_dist[ind],
                                np.radians(delta_pitch),
                                -1*np.radians(delta_yaw)]
            if endhunt:
                bout_descriptor[-1][1] = last_bout
                realfish.hunt_dataframes.append(copy.deepcopy(hunt_df))
                break
    realfish.exporter()
    csv_data(header, bout_descriptor, 'huntingbouts', exp.directory)
                                    
                                                               
def para_stimuli(dim, exp, hd):

    include_kde = False
    cont_win = exp.para_continuity_window
    int_win = exp.integration_window
    fr_to_avg = 10

    # This function simply gathers up all the IDs of the para during the hunt.
    def make_distance_matrix(pmat3D):
        pdm = pmat3D[:, cont_win:cont_win+fr_to_avg]
        avg_xyz = np.nanmedian(pdm, axis=1)
        indiv_xyz = [xyz for xyz in partition(3, avg_xyz)]
        distmat = squareform(pdist(indiv_xyz))
        return distmat

    def make_angle_matrix(angle_array):
        aa_t = angle_array[:, None]
        distmat = np.abs(angle_array - aa_t)
        return distmat

    def k_nearest(distmat, para_id):
        distances = distmat[para_id]
        no_self = np.delete(distances, para_id)
        no_self_sorted = np.sort(no_self)
        closest_1 = no_self_sorted[0]
        closest_3 = np.nanmean(no_self_sorted[0:3])
        closest_5 = np.nanmean(no_self_sorted[0:5])
        return closest_1, closest_3, closest_5
    
    def find_integration_para(wrt):
        para_in_intwindow = []
        for vispara in wrt:
            for vp in vispara:
                # just put the id in each frame
                para_in_intwindow.append(vp[3])
        return np.unique(para_in_intwindow).astype(np.int)
                
    hunt_ind_list = hd.hunt_ind_list
    p_list = hd.para_id_list
    actions = hd.actions
    stim_list = []

    for h, hp, ac in zip(hunt_ind_list,
                         p_list, actions):
        print h
        # here you will create a bout frames list for the entire hunt. once you
        # get to the hunted para, 
        p3D = np.load(exp.directory + '/para3D' + str(h).zfill(2) + '.npy')
        distmat = make_distance_matrix(p3D)
        penv = ParaEnv(h, exp.directory)
        penv.find_paravectors(False)
        wrth = np.load(
            exp.directory + '/wrth' + str(h).zfill(2) + '.npy')

# HERE IS WHERE THE INTERVAL GETS SET.
# When this is a function, your flags here will include whether its an init or an abort. 

        if ac >= 5:
            continue
            # update this when you know how fish move to known para. then you can figure out when the new hunt starts
            # once you figure out when the new hunt starts, just wrth_int = wrth[huntstart:huntstart+int_win]

        else:
            wrth_int = wrth[0:int_win]
                
        p_in_intwindow = find_integration_para(wrth_int)
        temp_stim_list = []
        for vis_para in p_in_intwindow:
            k1, k3, k5 = k_nearest(distmat, vis_para)
            p_wrth = create_poirec(h, 3,
                                   exp.directory, vis_para)[0:int_win]
            dist = [pr[4] for pr in p_wrth]
            az = [pr[6] for pr in p_wrth]
            alt = [pr[7] for pr in p_wrth]
            filt_az = gaussian_filter(az, 1)
            filt_alt = gaussian_filter(alt, 1)
            filt_dist = gaussian_filter(dist, 1)
            if len(filt_az) < 2 or len(filt_dist) < 2:
                continue
            delta_az = np.nanmedian(
                [b-a for a, b in sliding_window(2, filt_az)])
            delta_alt = np.nanmedian(
                [b-a for a, b in sliding_window(2, filt_alt)])
            delta_dist = np.nanmedian(
                [b-a for a, b in sliding_window(2, filt_dist)])
            az_position = np.nanmedian(filt_az[0:fr_to_avg])
            alt_position = np.nanmedian(filt_alt[0:fr_to_avg])
            distance = np.nanmedian(filt_dist[0:fr_to_avg])
            hunted = 0
            # Here would like to make a hairball probably.
            if vis_para == hp:
                hunted = ac
            dp = penv.dotprod[vis_para][cont_win:cont_win+int_win]
            vel = penv.velocity_mags[vis_para][cont_win:cont_win+int_win]
            avg_dp = np.nanmedian(dp)
            avg_vel = np.nanmedian(vel)
            p_entry = [az_position,
                       alt_position,
                       distance,
                       delta_az,
                       delta_alt,
                       delta_dist,
                       avg_dp,
                       avg_vel,
                       hunted,
                       h,
                       k1,
                       k3,
                       k5,
                       vis_para]
            if not all(math.isnan(para_val) for para_val in p_entry[0:8]):
                temp_stim_list.append(p_entry)
# here will perform your KDEs and get para envirnoment descriptors.
# idea should be to take only the first 5 or 10  and last 5 or 10
# position readings as Init_az and Final_az. calculate marginals for
# az, alt, dist. calculate densest point for each pariwise KDE.
        
        # here do your pairwise az and alt

        env_az_init = np.array([p[0] for p in temp_stim_list])
        env_alt_init = np.array([p[1] for p in temp_stim_list])
        az_distmat = make_angle_matrix(env_az_init)
        alt_distmat = make_angle_matrix(env_alt_init)
        ts2 = []
        for p_id, tp in enumerate(temp_stim_list):
            az_k1, az_k3, az_k5 = k_nearest(az_distmat, p_id)
            alt_k1, alt_k3, alt_k5 = k_nearest(alt_distmat, p_id)
            ts2.append(tp + [az_k1, az_k3, az_k5,
                             alt_k1, alt_k3, alt_k5])
        temp_stim_list = ts2
        print len(temp_stim_list)
        print('Length Temp Stim')
        if include_kde:
            max_contours = []
            env_dist_init = np.array([p[2] for p in temp_stim_list])
            env_dist_final = np.array(
                [p[2] + int_win*p[5] for p in temp_stim_list])
            env_az_final = np.array([p[0] + int_win*p[3]
                                     for p in temp_stim_list])
            env_alt_final = np.array([p[1] + int_win*p[4]
                                      for p in temp_stim_list])
            if env_az_final.shape[0] <= 2 or env_az_init.shape[0] <= 2:
                environment_varbs = np.full(9, np.nan).tolist()
                temp_stim_list = [p + environment_varbs
                                  for p in temp_stim_list]
                stim_list += temp_stim_list
                continue
            az_alt_kde = sb.jointplot(env_az_init, env_alt_init, kind='kde')
            az_dist_kde = sb.jointplot(env_az_init, env_dist_init, kind='kde')
            alt_dist_kde = sb.jointplot(env_alt_init,
                                        env_dist_init, kind='kde')
            final_azalt_kde = sb.jointplot(env_az_final,
                                           env_alt_final, kind='kde')
            final_azdist_kde = sb.jointplot(env_az_final,
                                            env_dist_final, kind='kde')
            az_marg = az_alt_kde.ax_marg_x.get_lines()[0].get_data()
            alt_marg = az_alt_kde.ax_marg_y.get_lines()[0].get_data()
            dist_marg = az_dist_kde.ax_marg_y.get_lines()[0].get_data()
            az_marg_final = final_azalt_kde.ax_marg_x.get_lines(
            )[0].get_data()
            alt_marg_final = final_azalt_kde.ax_marg_y.get_lines(
            )[0].get_data()
            dist_marg_final = final_azdist_kde.ax_marg_y.get_lines(
            )[0].get_data()
            az_max = [az_marg[0][np.argmax(az_marg[1])], np.max(az_marg[1])]
            alt_max = [alt_marg[1][np.argmax(alt_marg[0])],
                       np.max(alt_marg[0])]
            dist_max = [dist_marg[1][np.argmax(dist_marg[0])],
                        np.max(dist_marg[0])]
            az_max_final = [az_marg_final[0][np.argmax(az_marg_final[1])],
                            np.max(az_marg_final[1])]
            alt_max_final = [alt_marg_final[1][np.argmax(alt_marg_final[0])],
                             np.max(alt_marg_final[0])]
            dist_max_final = [dist_marg_final[1][
                np.argmax(dist_marg_final[0])], np.max(dist_marg_final[0])]
            delta_az_max = az_max_final[0] - az_max[0]
            delta_alt_max = alt_max_final[0] - alt_max[0]
            delta_dist_max = dist_max_final[0] - dist_max[0]
            kdeps = [az_alt_kde.ax_joint,
                     az_dist_kde.ax_joint,
                     alt_dist_kde.ax_joint]

    # EACH KDE IS A SUM OF THE MARGINALS, BUT PAIRWISE. E.G. A 1D + 1D sum IS A 2D PLANE. A 1D+1D+1D is a 3D volume. 
            for kdep in kdeps:
                max_contours.append(find_density(kdep, [], -1, 0))
#        pl.close('all')
            pl.show()
            environment_varbs = [az_max[0],
                                 alt_max[0],
                                 dist_max[0],
                                 delta_az_max,
                                 delta_alt_max,
                                 delta_dist_max]
                 #            max_contours[0],
                 #            max_contours[1],
                 #            max_contours[2]]
            temp_stim_list = [p + environment_varbs for p in temp_stim_list]
            
        stim_list += temp_stim_list
    pstim_header = ['Az Coord',
                    'Alt Coord',
                    'Distance',
                    'Delta Az',
                    'Delta Alt',
                    'Delta Dist',
                    'Dot Product',
                    'Raw Velocity',
                    'Hunted Or Not',
                    'Hunt ID',
                    'Dist_K1',
                    'Dist_K3',
                    'Dist_K5',
                    'Para ID',
                    'Az_K1',
                    'Az_K3',
                    'Az_K5',
                    'Alt_K1',
                    'Alt_K3',
                    'Alt_K5',
                    'Env Az',
                    'Env Alt',
                    'Env Distance',
                    'Delta Az Env',
                    'Delta Alt Env',
                    'Delta Dist Env']
                    # 'Max Contour AltAz',
                    # 'Max Contour AzDist',
                    # 'Max Contour AltDist']
    csv_data(pstim_header, stim_list, 'stimuli', exp.directory)


def hd_import(dr):
    x = pickle.load(open(dr + '/hunt_descriptor.pkl', 'rb'))
    return x

    
def contour_diameter(contour):
    distances = pdist(contour.vertices)
    sq = squareform(distances)
    diameter = np.max(sq)
    return diameter


def vmag(vec1, vec2):
    diff_vec = [vec2[0]-vec1[0], vec2[1]-vec1[1]]
    mag = np.sqrt(np.dot(diff_vec, diff_vec))
    return mag


def find_density(kde_plot, local_maxes, itr, cd):
    for path in kde_plot.collections[itr].get_paths():
        x, y = path.vertices.mean(axis=0)
        contour_diam = contour_diameter(path)
        if len(local_maxes) > 0:
            if vmag([x, y], local_maxes[0]) < cd:
                continue
        local_maxes.append((x, y))
        kde_plot.plot(x, y, "wo")
        if len(local_maxes) > 1:
            return local_maxes
    if len(local_maxes) < 2:
        if itr > -4:
            return find_density(kde_plot, local_maxes, itr-1, contour_diam)
        else:
            return local_maxes


def filter_uvec(vecs, sd):
    filt_sd = sd
    npvecs = np.array(vecs)
    filt_vecs = np.copy(npvecs)
    for i in range(npvecs[0].shape[0]):
        filt_vecs[:, i] = gaussian_filter(npvecs[:, i], filt_sd)
    return filt_vecs


def ang_to_unit(angles):
    u_vecs = []
    for ang in angles:
        ang = np.radians(ang)
        uvec = [np.cos(ang), np.sin(ang)]
        u_vecs.append(uvec)
    u_vecs = np.array(u_vecs)
    return u_vecs


def unit_to_angle(units):
    angles = []
    for u in units:
        angle = np.arctan2(u[1], u[0])
        if angle < 0:
            angle = (2 * np.pi) + angle
        angles.append(angle)
    return np.degrees(angles)


def bout_header(v_dict, num_frames_in_bout):
    header = []
    for i in range(num_frames_in_bout):
        for j in range(len(v_dict)):
            entry = v_dict[str(j)] + '_' + str(i)
            header.append(entry)
    return header


def magvector(vector):
    mag = np.sqrt(np.dot(vector, vector))
    return mag



# This function plots all the possible bouts the fish can perform.
# It maps all bouts to Az, Alt, Dist coordinates with an interbout, pitch, and yaw.


def map_all_bouts(myexp, dim):
    # Don't forget delta yaw is negative with respect to Az changes. 
    bout_window = range(0, len(myexp.bout_frames))
    pitch_flag = int(dim.inv_fdict['Total Pitch Change'])
    yaw_flag = int(dim.inv_fdict['Total Yaw Change'])
    interbouts = [b-a for a, b in sliding_window(2, myexp.bout_frames)]
    myexp.map_bouts_to_heading(0, [[bout_window[0], bout_window[-1]]])
    delta_pitch = np.radians(
        [dim.all_flags[bout][pitch_flag]
         for bout in bout_window])
    delta_yaw = -1*np.radians([dim.all_flags[bout][yaw_flag]
                               for bout in bout_window])
    bout_descriptor = {'Bout Az': myexp.bout_az,
                       'Bout Alt': myexp.bout_alt,
                       'Bout Dist': myexp.bout_dist,
                       'Interbouts': interbouts,
                       'Delta Pitch': delta_pitch,
                       'Delta Yaw': delta_yaw}
    fig, axes = pl.subplots(1, len(bout_descriptor),
                            sharex=False,
                            sharey=False,
                            figsize=(8, 8))
    for ind, (title, entry) in enumerate(bout_descriptor.iteritems()):
        sb.distplot(entry, ax=axes[ind])
        axes[ind].set_title(title)

    graph_3D = pl.figure(figsize=(10, 10))
    ax3d = graph_3D.add_subplot(111, projection='3d')
    ax3d.set_title('3D Para Record')
    ax3d.set_xlim([-np.pi, np.pi])
    ax3d.set_ylim([-np.pi, np.pi])
    ax3d.set_zlim([0, 500])
    ax3d.set_xlabel('Bout Az')
    ax3d.set_ylabel('Bout Alt')
    ax3d.set_zlabel('Bout Dist')
    cmap = pl.get_cmap('seismic')
#    yaw_max = np.max(np.abs(delta_yaw))
#    norm = Normalize(-yaw_max, yaw_max)
    norm = Normalize(vmin=-1, vmax=1)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba_vals = scalarMap.to_rgba(delta_yaw)
    for i in range(len(myexp.bout_az) - 1):
        ax3d.plot([myexp.bout_az[i]],
                  [myexp.bout_alt[i]],
                  [myexp.bout_dist[i]],
                  color=rgba_vals[i],
                  marker='.',
                  ms=10*delta_pitch[i])
    scalarMap.set_array(delta_yaw)
    graph_3D.colorbar(scalarMap)
    pl.show()
    return bout_descriptor


def monotonic_max(winsize, array_in, maxthresh):
    mmax = []
    for ind, win in enumerate(sliding_window(winsize, array_in)):
        if not (np.array(win) > maxthresh).all():
            continue
        windiff = np.diff(win)
        if (windiff[0:(
                winsize / 2) + 1] >= 0).all() and (
                    windiff[winsize/2:] <= 0).all():
            mmax.append(ind)
            print win
            print windiff
    return np.array(mmax)

def fixmappings(exp, dim, hd):
    for hunt_id in hd.hunt_ind_list:
        init_frame = exp.bout_frames[dim.hunt_wins[hunt_id][0]]
        abort_frame = exp.bout_frames[dim.hunt_wins[hunt_id][1]]
        integ_window = exp.integration_window
        post_frames = exp.integration_window
        window = [init_frame - integ_window,
                  abort_frame + post_frames]
        exp.paradata = ParaMaster(window[0], window[1], exp.directory)
        exp.paradata.para3Dcoords = np.load(myexp.directory +
                                            '/para3D' +
                                            str(hunt_id).zfill(2) + '.npy')
        exp.framewindow = window
        exp.map_bouts_to_heading(hunt_id, dim.hunt_wins)
        exp.map_para_to_heading(hunt_id)
    hunted_para_descriptor(dim, exp, hd)
    para_stimuli(dim, exp, hd)


if __name__ == '__main__':

# 15 requires 250 consecutive milliseconds of data. this yields 497 bouts. 
    print('Running Master')


# This dictionary describes all variables to record and plot for each bout.     
    all_varbs_dict = {
        '0': 'Pitch',
        '1': 'Tail Segment 1',
        '2': 'Tail Segment 2',
        '3': 'Tail Segment 3',
        '4': 'Tail Segment 4',
        '5': 'Tail Segment 5',
        '6': 'Tail Segment 6',
        '7': 'Tail Segment 7',
        '8': 'Vector Velocity',
        '9': 'Delta Z',
        '10': 'Delta Yaw',
        '11': 'Eye1 Angle',
        '12': 'Eye2 Angle',
        '13': 'Eye Sum',
        '14': 'Interbout_Back'}

# This dictionary describes the variables to use for clustering.
    sub_dict = { '8':'Vector Velocity'}
        


    bout_dict = {
#        '0': 'Pitch',
        # '1': 'Tail Segment 1',
#         '2': 'Tail Segment 2',
        # '3': 'Tail Segment 3',
        # '4': 'Tail Segment 4',
        # '5': 'Tail Segment 5',
       # '6': 'Tail Segment 6',
#        '7': 'Tail Segment 7',
        # '8': 'Vector Velocity',
        # '9': 'Delta Z',
 #       '10': 'Delta Yaw',
#        '11': 'Eye1 Angle',
#        '12': 'Eye2 Angle'}
        '13': 'Eye Sum'} #,
#        '14': 'Interbout_Back'}

# This dictionary describes a flag set of variables that should be calculated for each bout that describes characteristics of the bout    

    flag_dict = {
        '0': 'Bout ID',
        '1': 'Interbout',
        '2': 'Eye Angle Sum',
        '3': 'Average Velocity',
        '4': 'Total Z',
        '5': 'Total Pitch Change',
        '6': 'Total Yaw Change',
        '7': 'Fluorescence Level',
        '8': 'Bout Duration',
        '9': 'Cluster ID'}
#        '9': 'Fish ID',
 


# Experiment arg is how many continuous frames are required for a bout call


# MAIN LINE: 


# Experiment class first 2 args define the minimum length of the bout and the amount of frames to go backwards for para calculation. 
# The bout_detector call finds bouts based on either orientation changes ('orientation') or on tail standard deviation changes ('tail')
# Nan_filt_and_arrange then finds whether the bouts detected are composed of Nan-Free fish variables 
# and the function (if flag is True) filters for bouts where fish is facing a tank edge. The bout variables are then stored in a continuous 
# bout array, matched with a flag array that describes summary statistics for each bout. A new BoutsandFlags object is then created
# whose only role is to contain the bouts and corresponding flags for each fish. 

    fish_id = '041618_1'
    drct = os.getcwd() + '/' + fish_id
    new_exp = True
    dimreduce = False
    
    if new_exp:
        # HERE IF YOU WANT TO CLUSTER MANY FISH IN THE FUTURE, MAKE A DICT OF FISH_IDs AND RUN THROUGH THIS LOOP. MAY WANT TO CLUSTER MORE FISH TO PULL OUT STRIKES VS ABORTS. 
        # if num_fish != 1:
        #     FISH ID DICTIONARY HERE, LOOP THROUGH
        #         myexp = Experiment(10, all_varbs_dict, flag_dict, drct)
        #         myexp.bout_detector(True, 'tail')
        #         myexp.bout_nanfilt_and_arrange(False)
        #         bouts_flags = BoutsAndFlags(fish_id,
        #                                     myexp.bout_data, myexp.bout_flags)
        #         bouts_flags.exporter()
#        else:
        
        myexp = Experiment(10, 3, all_varbs_dict, flag_dict, drct)
#        myexp.bout_detector(True, 'combined')
#        myexp.bout_detector(True, 'tail')
        myexp.bout_detector()
        myexp.bout_nanfilt_and_arrange(False)
        bouts_flags = BoutsAndFlags(drct,
                                    myexp.bout_data, myexp.bout_flags)
        bouts_flags.exporter()
        print("Creating Unit Vectors")
        myexp.create_unit_vectors()
        myexp.exporter()

    else:
        myexp = pickle.load(open(drct + '/master.pkl', 'rb'))

    if dimreduce:
        dim = DimensionalityReduce(bout_dict,
                                   flag_dict,
                                   all_varbs_dict,
                                   drct, {})
        dim.concatenate_records()
        dim.dim_reduction(2)
        dim.exporter()
    else:
        dim = pickle.load(open(drct + '/dim_reduce.pkl', 'rb'))

    clear_hunts = raw_input('New hunt windows?: ')
    if clear_hunts == 'y':
        dim.clear_huntwins()
        dim.find_hunts([1],[0])
        dim.exporter()
        hd = Hunt_Descriptor(drct)
    else:
        import_hd = raw_input('Import Hunt Descriptor?: ')
        if import_hd == 'y':
            hd = hd_import(myexp.directory)
        elif import_hd == 'n':
            hd = Hunt_Descriptor(drct)

