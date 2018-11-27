import os
import copy
import csv
import cv2
import pandas as pd
import imageio
import itertools
from astropy.convolution import Gaussian1DKernel, convolve
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering
import numpy as np
import pickle
import math
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from collections import Counter
import scipy.signal
from toolz.itertoolz import sliding_window, partition
# from sympy import Point3D, Plane
from scipy.spatial.distance import pdist, squareform
import warnings
from collections import deque
import matplotlib.cm as cm
from matplotlib import pyplot as pl
from matplotlib.colors import Normalize, ListedColormap
import seaborn as sb
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as anim
from phinalIR_cluster_wik import Variables
from phinalFL import Fluorescence_Analyzer
from pvidFINAL import Para, ParaMaster, return_paramaster_object
sb.set_style('darkgrid')
warnings.simplefilter('ignore')

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


class RealFishControl:
    def __init__(self, exp):
        self.firstbout_para_intwin = 5
        self.fish_xyz = exp.ufish_origin
        self.fish_orientation = exp.ufish
        self.pitch_all = exp.spherical_pitch
        self.yaw_all = exp.spherical_yaw
        self.fish_id = exp.directory[-8:]
        self.hunt_ids = []
        self.hunt_results = []
        self.hunt_frames = []
        self.hunt_interbouts = []
        self.huntbout_durations = []
        self.initial_conditions = []
        self.hunt_dataframes = []
        self.all_spherical_bouts = []
        self.all_spherical_huntbouts = []
        self.para_xyz_per_hunt = []
        self.directory = exp.directory

# exp contains all relevant fish data. frames will come from hunted_para_descriptor, which will
# create realfishcontrol objects as it runs. 

    def find_initial_conditions(self):
        for (firstframe, lastframe) in self.hunt_frames:
            self.initial_conditions.append([self.fish_xyz[firstframe],
                                            self.pitch_all[firstframe],
                                            self.yaw_all[firstframe]])

    def model_input(self, hunt_num):
        return {"Hunt Dataframe": self.hunt_dataframes[hunt_num],
                "Para XYZ": self.para_xyz_per_hunt[hunt_num],
                "Initial Conditions": self.initial_conditions[hunt_num],
                "Interbouts": self.hunt_interbouts[hunt_num],
                "Bout Durations": self.huntbout_durations[hunt_num],
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
        self.interp_windows = []
        self.decimated_vids = []
        self.dec_doubles = []

    def exporter(self):
        with open(self.directory + '/hunt_descriptor.pkl', 'wb') as file:
            pickle.dump(self, file)

    def check_for_doubles(self, ind):
        args_matching_id, = np.where(np.array(self.hunt_ind_list) == ind)
        if args_matching_id.shape[0] == 1:
            return False
        else:
            d1 = self.decimated_vids[args_matching_id[0]]
            d2 = self.decimated_vids[args_matching_id[1]]
            if d1 != d2:
                self.dec_doubles.append(ind)
                self.dec_doubles = np.unique(self.dec_doubles)
                return True
# This is a check for decimated vids vs. changes of mind            
            else:
                return False

# with this T, you choose a _d for this ind
# in h_p_d and a normal in pvec and para_stim.

            
    def parse_interp_windows(self, exp, poi):
        cont_win = exp.paradata.pcw
        inferred_window_ranges_poi = []
        iw = copy.deepcopy(exp.paradata.interp_indices)
        unique_wins = [k for k, a in itertools.groupby(iw)]
        inferred_windows_poi = [np.array(win[1]) - cont_win
                                for win in unique_wins if win[0] == poi]
        if inferred_windows_poi != []:
            inferred_windows_poi = np.concatenate(inferred_windows_poi)
            inferred_window_ranges_poi = [
                    range(win[0], win[1]) for win in inferred_windows_poi]
        self.interp_windows.append(inferred_window_ranges_poi)
        self.decimated_vids.append(exp.paradata.decimated_vids)

    def current(self):
        print self.hunt_ind_list
        print self.para_id_list
        print self.actions
        print self.boutrange
        print self.decimated_vids

    def remove_entry(self, h_id):
        ind = np.where(np.array(self.hunt_ind_list) == h_id)[0][0]
        del self.hunt_ind_list[ind]
        del self.para_id_list[ind]
        del self.actions[ind]
        del self.boutrange[ind]
        del self.interp_windows[ind]
        del self.decimated_vids[ind]

    def update_hunt_data(self, p, a, br, exp):
        self.hunt_ind_list.append(copy.deepcopy(exp.current_hunt_ind))
        self.para_id_list.append(p)
        self.actions.append(a)
        self.boutrange.append(br)
        self.parse_interp_windows(exp, p)
        self.exporter()


# 	1) Hunted Para. -1 if unknown, -2 if you can see it in non-br sub vid. 
#       2) Orientation to unknown target = 0
#          Strike Success = 1,
# 	   Strike Fail = 2,
# 	   Abort = 3,
# 	   Abort for Reorientation = 4,
# 	   Reorientation and successful strike = 5,
# 	   Reorientation and fail = 6,
# 	   Reorientation and abort = 7,
#       3) Range of bouts within hunt. Type list. (0,1) for unknown targets
#       4) enter myexp

# 

        
# for each para env, you will get a coordinate matrix that is scalexscalexscale, representing
# 3270 x 2 pixels in all directions. the scale must be ODD!! this guarantees that it will have a center
# cube. make a coordinate matrix by adding each unit vector representing the fish basis to the scale / 2, scale /2 , scale / 2 coordinate.
# the unit vectors are scaled by 3270 / scale. coord - scale / 2 will give you the right scale factor. e.g. if the scale is 10, [0,0,0] will be
# -5, -5, -5. when you multiply the scaled unit vectors of the basis by these coords and add to ufish origin, you get a real x,y,z coord of tank position. if# all coords are less than 1888, give the coordinate of env_mat a 1 bump. if not, give a zero bump. never give a bump to negative x coordinates! (i.e. start # the loop at scale / 2 for x, 0 for y and z.


class ParaEnv:

    def __init__(self, index, directory, filter_sd, dec, *para_coords):
        if para_coords == ():
            if dec:
                subscript = '_d'
            else:
                subscript = ''
            self.wrth = np.load(
                directory + '/wrth' + str(index).zfill(2) + subscript + '.npy')
            self.para_coords = np.load(
                directory + '/para3D' + str(
                    index).zfill(2) + subscript + '.npy')
            self.dimensions = 3
#        self.high_density_coord = np.load('high_density_xyz.npy')
        else:
            self.para_coords = para_coords[0][0]
            self.wrth = []
            self.dimensions = para_coords[0][1]
        self.paravectors = []
        self.dotprod = []
        self.velocity_mags = []
        self.target = []
        self.bout_number = str(index)
        self.vector_spacing = 3
        self.gkern_sd = filter_sd

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
        gkern = Gaussian1DKernel(self.gkern_sd)
        para_of_interest = []
        if para_id == ():
            if self.dimensions == 3:
                para_of_interest = range(0, self.para_coords.shape[0], 3)
            elif self.dimensions == 2:
                para_of_interest = range(0, self.para_coords.shape[0], 2)
        else:
            para_of_interest = [para_id[0]*3]
        for rec_rows in para_of_interest:
            win_length = self.vector_spacing
            rec_id = rec_rows / 3
            x = self.para_coords[rec_rows]
            x = convolve(x, gkern, preserve_nan=True)
            y = self.para_coords[rec_rows+1]
            y = convolve(y, gkern, preserve_nan=True)
# Note you have to cut the last val off the end of the convolution --
# the nans at the end screw up the kernel 
            
            x_diffs = [a[-1] - a[0] for a in partition(win_length, x)][:-1]
            y_diffs = [a[-1] - a[0] for a in partition(win_length, y)][:-1]
            if self.dimensions == 3:
                z = self.para_coords[rec_rows+2]
                z = convolve(z, gkern, preserve_nan=True)
                z_diffs = [a[-1] - a[0] for a in partition(win_length, z)][:-1]
                vel_vector = [
                    np.array([deltax, deltay, deltaz])
                    for deltax, deltay, deltaz in zip(x_diffs, y_diffs, z_diffs)
                ]
            elif self.dimensions == 2:
                vel_vector = [
                    np.array([deltax, deltay])
                    for deltax, deltay in zip(x_diffs, y_diffs)
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


class DimensionalityReduce():

    def __init__(self,
                 bout_dict,
                 flag_dict,
                 all_varbs_dict, fish_id_list, exp_bouts_list, exp_flags_list, bout_frames):
        self.fish_id_list = fish_id_list
        self.directory = os.getcwd()
        self.cluster_input = []
        self.num_dp = len(all_varbs_dict)
        self.all_varbs_dict = all_varbs_dict
        self.num_bouts_per_fish = [len(bts) for bts in exp_bouts_list]
        self.all_bouts = np.concatenate(exp_bouts_list, axis=0).tolist()
        self.all_flags = np.concatenate(exp_flags_list, axis=0).tolist()
        self.dim_reduce_output = []
        self.cmem_pre_sub = []
        self.cluster_membership = []
        self.bout_frames_by_fish = bout_frames
        self.cmem_by_fish = []
        self.bout_dict = bout_dict
        self.flag_dict = flag_dict
        self.inv_fdict = {v: k for k, v in flag_dict.iteritems()}
        self.transition_matrix = np.array([])
        self.cluster_count = 3

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

    def set_directory(self, exp):
        self.directory = exp.directory

    def strike_abort_sep(self, term_cluster):
        cluster_indices = np.where(self.cluster_membership == term_cluster)[0]
            # now get all bouts from cluster indices in all_bouts
        flags_in_term_cluster = np.array(self.all_flags)[cluster_indices]
        interbout_index = int(self.inv_fdict['Interbout'])
        delta_yaw_index = int(self.inv_fdict['Total Yaw Change'])
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

    def create_cmem_by_fish(self):
        self.cmem_by_fish = []
        cumsum_numbouts = np.r_[0, np.cumsum(self.num_bouts_per_fish)]
        self.cmem_by_fish = [self.cluster_membership[
            i[0]:i[1]] for i in sliding_window(2, cumsum_numbouts)]

    def watch_cluster(self, fish_id, clusters, vidtype, term):
        directory = os.getcwd() + '/' + fish_id
        cluster_ids_for_fish = self.cmem_by_fish[
            self.fish_id_list.index(fish_id)]
        bout_frames_for_fish = self.bout_frames_by_fish[
            self.fish_id_list.index(fish_id)]
        if vidtype == 1:
            vid = imageio.get_reader(
                directory + '/top_contrasted.AVI', 'ffmpeg')
        elif vidtype == 0:
            vid = imageio.get_reader(
                directory + '/conts.AVI', 'ffmpeg')
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        c_mem_counter = 0
        if term:
            ib, dy = self.strike_abort_sep(clusters)
        for bout_frame, cluster_mem in zip(bout_frames_for_fish,
                                           cluster_ids_for_fish):
            if cluster_mem in clusters:
                if term:
                    dyaw = dy[c_mem_counter]
                    c_mem_counter += 1
                    if dyaw > 10:
                        continue
                firstframe = bout_frame - 10
                if firstframe <= 0:
                    continue
                lastframe = bout_frame + 30
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

    def exporter(self):
        with open(self.directory + '/dim_reduce.pkl', 'wb') as file:
            pickle.dump(self, file)

    def redefine_clustering_params(self, new_bdict, num_clusters):
        self.cluster_count = num_clusters
        self.bout_dict = new_bdict
        self.prepare_records()

    def prepare_records(self):
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
        self.cluster_input = np.array(norm_bouts)
        print(str(len(self.all_bouts)) + " Bouts Detected")
        cluster_model = SpectralClustering(n_clusters=self.cluster_count,
                                           affinity='nearest_neighbors',
                                           n_neighbors=10)
        # i believe its only possible for a nan bout to be at the end
        # otherwise a bout won't be called at the beginning.  
        c_flag = cluster_model.fit_predict(self.cluster_input)
        self.cluster_membership = c_flag
        num_clusters = np.unique(self.cluster_membership).shape[0]
        new_flags = []
        for flag, cflag in zip(self.all_flags, self.cluster_membership):
            flag.append(cflag)
            new_flags.append(flag)
        self.all_flags = new_flags

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
        dim_reduce_data = model.fit_transform(self.cluster_input)
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
            pl.close()

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
        self.spherical_pitch = []
        self.spherical_yaw = []
#        self.spherical_dyaw = []
        self.invert = True
        self.refract = refractory_period
        self.bout_dict = bout_dict
        self.flag_dict = flag_dict
        self.inv_bdict = {v: k for k, v in bout_dict.iteritems()}
        self.directory = dirct
        self.fishdata = pickle.load(open(
           self.directory + '/fishdata.pkl', 'rb'))
        self.original_pitch = copy.deepcopy(self.fishdata.pitch)
#        self.fluor_data = pickle.load(open('fluordata.pkl', 'rb'))
        self.set_types()
        self.nans_in_original = count_nans(self.fishdata)
        self.pitch_fix()
        self.filter_fishdata()
        self.delta_ha = []
        self.vectVcalc(2)
        self.fluor_data = []
        self.paradata = []
        self.framewindow = []
        self.barrierlocation = []
        self.bout_frames = []
        self.bout_durations = []
        self.bout_durations_tail_only = []
        self.rejected_bouts = []
        self.bout_data = []
        self.bout_flags = []
        self.minboutlength = cluster_length
        self.num_dp = len(bout_dict)
        self.para_continuity_window = 600
        self.bout_of_interest = 0
        self.hunt_windows = []
        self.current_hunt_ind = 0
        self.integration_window = 10
        self.bout_az = []
        self.bout_alt = []
        self.bout_dist = []
        self.bout_dpitch = []
        self.bout_dyaw = []
        self.hunt_cluster = 0
        self.hunt_wins = []
        self.deconverge_cluster = 0
        self.cluster_membership = []

# to do in bout detector: 
# HA must have no nans in 20 width window. 
# do bout detection for vel and tailangle here.

    def pitch_fix(self):
        new_pitch = []
        for pitch, ha in zip(self.fishdata.pitch, self.fishdata.headingangle):
            if (80 < ha < 100) or (260 < ha < 280):
                newpit = np.degrees(np.arcsin(pitch / 90.0))
            else:
                newpit = pitch
            new_pitch.append(newpit)
        self.fishdata.pitch = new_pitch
 
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

    def nearwall(self, bout_frame, bout_dur, wall_thresh, ceiling_thresh):

        def calc_near_wall(x, y, z, ha, pitch):
            if x < wall_thresh and (90 < ha < 270):
                return True
            elif x > 1888-wall_thresh and (ha < 90 or ha > 270):
                return True
            elif y < wall_thresh and (ha > 180):
                return True
            elif y > 1888-wall_thresh and (ha < 180):
                return True
            elif z < ceiling_thresh and pitch < 0:
                return True
            elif z > 1888-ceiling_thresh and pitch > 0:
                return True
            else:
                return False
        
        ha_init = np.median(
            self.fishdata.headingangle[bout_frame-3:bout_frame])
        x_init = np.median(self.fishdata.x[bout_frame-3:bout_frame])
        y_init = np.median(self.fishdata.y[bout_frame-3:bout_frame])
        z_init = np.median(self.fishdata.z[bout_frame-3:bout_frame])
        pitch_init = np.median(self.fishdata.pitch[bout_frame-3:bout_frame])
        x_end = np.median(
            self.fishdata.x[bout_frame+bout_dur:bout_frame+bout_dur+3])
        y_end = np.median(
            self.fishdata.y[bout_frame+bout_dur:bout_frame+bout_dur+3])
        z_end = np.median(
            self.fishdata.z[bout_frame+bout_dur:bout_frame+bout_dur+3])
        pitch_end = np.median(
            self.fishdata.pitch[bout_frame+bout_dur:bout_frame+bout_dur+3])
        ha_end = np.median(
            self.fishdata.headingangle[
                bout_frame+bout_dur:bout_frame+bout_dur+3])
        return (calc_near_wall(x_init, y_init, z_init, ha_init, pitch_init)
                or calc_near_wall(x_end, y_end, z_end, ha_end, pitch_end))

    def find_hunts(self, init_inds, abort_inds):
        self.hunt_wins = []
        self.hunt_cluster = init_inds
        self.deconverge_cluster = abort_inds
        start_bouts_per_hunt = 10
        for i, cmem in enumerate(self.cluster_membership):
            if cmem in self.hunt_cluster:
                if i + start_bouts_per_hunt < len(self.cluster_membership) - 1:
                    self.hunt_wins.append([i, i + start_bouts_per_hunt])
                else:
                    self.hunt_wins.append([i, len(self.cluster_membership) - 2])
        self.exporter()
                
    def extend_hunt_window(self, numbouts):
        self.hunt_wins[self.current_hunt_ind][1] += numbouts
        np.save(self.directory + 'hunt_wins.npy', self.hunt_wins)

    def reset_hunt_window(self):
        orig = self.hunt_wins[self.current_hunt_ind][0]
        self.hunt_wins[self.current_hunt_ind] = [orig, orig+10]
        np.save(self.directory + 'hunt_wins.npy', self.hunt_wins)
        

# one bug is deconverge_cluster should be cleared when find_hunts is called        
    def cap_hunt_window(self):
        start_ind = self.hunt_wins[self.current_hunt_ind][0]
        original_end_ind = self.hunt_wins[self.current_hunt_ind][1]
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
        self.hunt_wins[self.current_hunt_ind] = [start_ind, start_ind + di]
        self.watch_hunt(1, 15, self.current_hunt_ind)
        response = raw_input('Cap Window?  ')
        if response == 'y':
            np.save(self.directory + 'hunt_wins.npy', self.hunt_wins)
        else:
            #            self.hunt_wins[ind] = [start_ind, original_end_ind]
            while(True):
                new_end = raw_input('Correct End: ')
                if new_end == 'd':
                    break
                else:
                    new_end = int(new_end)
                self.hunt_wins[self.current_hunt_ind] = [start_ind,
                                                         start_ind + new_end]
                self.watch_hunt(1, 15, self.current_hunt_ind)
        np.save(self.directory + 'hunt_wins.npy', self.hunt_wins)
        view_bouts = raw_input("View Bouts During Hunt?: ")
        if view_bouts == 'y':
            bouts_during_hunt(self.current_hunt_ind, dim, self, True)
            self.watch_hunt(0, 50, self.current_hunt_ind)

    def nexthunt(self):
        ret = self.watch_hunt(1, 15, self.current_hunt_ind + 1)
        if not ret:
            return self.nexthunt()

    def repeat(self, vid):
        self.watch_hunt(vid, 15, self.current_hunt_ind)

    def backone(self):
        self.watch_hunt(1, 15, self.current_hunt_ind - 1)

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
        z_diffs = [0]
        eye_sd = 3
        phileft_filt = gaussian_filter(self.fishdata.phileft, eye_sd)
        phiright_filt = gaussian_filter(self.fishdata.phiright, eye_sd)
        ha_diffs = calculate_delta_yaw(self.fishdata.headingangle)
        for z in sliding_window(2, self.fishdata.low_res_z):
            z_diffs.append(z[1]-z[0])

        for bindex, (bout, bout_duration) in enumerate(
                zip(bout_windows, self.bout_durations)):
            interbout_backwards = np.copy(
                bout[0] - bout_windows[bindex - 1][0])
            bout_vec = []
            # ha_init = np.nanmean(self.fishdata.headingangle[bout[0]:bout[0]+5])
            # x_init = np.nanmean(self.fishdata.x[bout[0]:bout[0]+5])
            # y_init = np.nanmean(self.fishdata.y[bout[0]:bout[0]+5])
            # z_init = np.nanmean(self.fishdata.z[bout[0]:bout[0]+5])
            # pitch_init = np.nanmean(self.fishdata.pitch[bout[0]:bout[0]+5])
            if self.nearwall(bout[0], bout_duration, 200, 100):
                rejected_bouts.append([bindex, 'nearwall'])
                if filter_walls:
                    continue

            bout = (bout[0], bout[0] + self.minboutlength)
            full_window = (bout[0]-self.integration_window, bout[1])
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
        if self.bout_frames[0] < 0:
            self.bout_data = self.bout_data[1:]
            self.bout_durations = self.bout_durations[1:]
            self.bout_frames = self.bout_frames[1:]
        self.rejected_bouts = rejected_bouts
        print len(self.bout_data)
# Now create flags for each bout.
#bout_number is the id of the bout, bout_frame is which frame it occurs in in the ir only movies. 

        for bout_number, bout_frame in enumerate(self.bout_frames):
            bout = [bout_frame, bout_frame + self.bout_durations[bout_number]]
            flags = []
            num_nans = np.sum(self.nans_in_original[bout[0]:bout[1]])
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
                elif self.flag_dict[str(key)] == 'Number of Nans':
                    flags.append(num_nans)
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

        plot_bouts = False
        tailang = [tail[-1] for tail in self.fishdata.tailangle]
        ta_std = gaussian_filter(
            [np.nanstd(tw) for tw in sliding_window(5, tailang)], 2).tolist()
        std_thresh = 4
        print("STD THRESH")
        print(std_thresh)
        bts = scipy.signal.argrelmax(np.array(ta_std), order=3)[0]
        bts = [b for b in bts if ta_std[b] > std_thresh]
        boutstarts = [bts[0]]
        boutends = [bts[0]+3]

# DO want to do boutfilter_recur here, but over a very small window (i.e. you don't want extreme overlaps)
        for b1, b2, b3 in sliding_window(3, bts):
            backwin = ta_std[b1:b2]
            backwin.reverse()
            backwin = np.array(backwin)
            forwardwin = np.array(ta_std[b2:b3])
        #thresh here will be noise + min calculation
            try:
                crossback = -np.where(backwin < std_thresh)[0][0] + b2
            except IndexError:
                crossback = b2 - 3
            if crossback <= boutends[-1]:
                crossback = boutends[-1] + 1
            try:
                crossforward = np.where(forwardwin < std_thresh)[0][0] + b2
            except IndexError:
                crossforward = b2 + 3
            boutstarts.append(crossback)
            boutends.append(crossforward)
#        boutstarts, boutends = boutfilter_recur(boutstarts, boutends)
        bout_durations = [
            be - bs for bs, be in zip(boutstarts, boutends)]
        
        self.bout_frames = boutstarts
        self.bout_durations = bout_durations
        if plot_bouts:
            fig, (ax1, ax2) = pl.subplots(1, 2, sharex=True, figsize=(6, 6))
            ax1.plot(bts, np.zeros(len(bts)), marker='.', color='b')
            ax2.plot(bts, np.zeros(len(bts)), marker='.', color='b')
            ax1.plot(
                boutstarts, np.zeros(len(boutstarts)), marker='.', color='c')
            ax2.plot(
                boutstarts, np.zeros(len(boutstarts)), marker='.', color='c')
            ax1.plot(boutends, np.zeros(len(boutends)), marker='.', color='m')
            ax2.plot(boutends, np.zeros(len(boutends)), marker='.', color='m')
            ax1.plot(ta_std)
            ax2.plot(tailang)
            pl.show()

    def para_during_hunt(self, index, movies, hunt_wins):
        cv2.destroyAllWindows()
        showstats = False
        pl.ioff()
        directory = self.directory + '/'
        init_frame = self.bout_frames[hunt_wins[index][0]]
        end_frame = self.bout_frames[hunt_wins[index][1]]
        integ_window = self.integration_window
        post_frames = self.bout_durations[hunt_wins[index][1]] + 1
        window = [init_frame - self.para_continuity_window - integ_window,
                  end_frame + post_frames]
        if window[0] < 0:
            return False
        self.paradata = return_paramaster_object(window[0],
                                                 window[1],
                                                 movies,
                                                 directory, showstats,
                                                 self.para_continuity_window)
# This establishes a setup where para_continuity_window is used for correlation, and integ_window frames before the first bout are kept for wrth. so the framewindow adds para_continuity_window to the 3D para coords so that you don't map 10 seconds backwards, but only 500 ms backwards.         
        self.framewindow = [window[0] + self.para_continuity_window, window[1]]
        # self.map_bouts_to_heading(index, hunt_wins)
        if self.paradata.startover:
            return False
        self.map_para_to_heading(index)
        return True

    def watch_hunt(self, cont_side, delay, h_ind):
        print('Hunt # ' + str(h_ind))
        print('Init Cluster: ' + str(
            self.cluster_membership[self.hunt_wins[h_ind][0]]))
        self.current_hunt_ind = h_ind
        firstbout, lastbout = self.hunt_wins[h_ind]
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
        elif cont_side == 3:
            vid = imageio.get_reader(dirct + 'tailcontvid.AVI', 'ffmpeg')
        elif cont_side == 4:
            vid = imageio.get_reader(dirct + 'sideconts.AVI', 'ffmpeg')
        elif cont_side == 5:
            vid = '/Volumes/WIKmovies/' + dirct[-9:-1] + '_cam0.AVI'
        elif cont_side == 6:
            vid = '/Volumes/WIKmovies/' + dirct[-9:-1] + '_cam1.AVI'
        else:
            print('unspecified stream')
            return False
#        cv2.namedWindow('vid', flags=cv2.WINDOW_NORMAL)
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
#        cv2.startWindowThread()
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
            elif keypress == 49:
                break
        cv2.destroyAllWindows()
        vid.close()
        cv2.namedWindow(
            'Enter 1 for Full Characterization',
            flags=cv2.WINDOW_NORMAL)
        cv2.moveWindow('Enter 1 for Full Characterization', 20, 20)
        key = cv2.waitKey(0)
        print key
        if key == 49:
            ret = self.para_during_hunt(h_ind, True, self.hunt_wins)
            cv2.destroyAllWindows()
            if ret:
                self.paradata.watch_event(1)
        cv2.destroyAllWindows()
        return True

# Eye1 is the eye on the side of the direction of the turn.
    def bout_stats(self, bout_ind, global_bout):
        if not global_bout:
            bout_index = bout_ind + self.hunt_wins[self.current_hunt_ind][0]
        else:
            bout_index = bout_ind
        bout = self.bout_data[bout_index]
        num_cols = int(np.ceil(self.num_dp/3.0))
        num_rows = 3
        fig = pl.figure(figsize=(8, 8))
        palette = np.array(sb.color_palette("Set1", self.num_dp))
        cmem = self.cluster_membership[bout_index]
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
        self.paradata.find_misses(1)
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

    def assign_z(self, xyrec, auto, zmax, *xzrec):
        if zmax == 0:
            fish_z = self.fishdata.z[
                self.paradata.framewindow[
                    0] + self.paradata.all_xy[
                        xyrec].timestamp + len(
                            self.paradata.all_xy[xyrec].location)]
            fish_z = 1888 - fish_z
            if xzrec == ():
                print("Fish mouth is at " + str(fish_z))
            elif xzrec != ():
                print(
                    "last z of para is " + str(
                        self.paradata.all_xz[
                            xzrec[0]].location[-1][1]))
            zmax = raw_input('Enter Z:  ')
            zmax = int(zmax)
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
                    "Enter f for assign_z ahead of known xzrec, b for behind: ")
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
        self.paradata.watch_event(2)
        accept = raw_input("Accept Fix?: ")
        if accept == 'n':
            new_maxz = raw_input("Enter New Max Z: ")
            new_maxz = int(new_maxz)
            del self.paradata.all_xz[-1]
            del self.paradata.xyzrecords[-1]
            if xzrec != ():
                return self.assign_z(xyrec, auto, new_maxz, xzrec)
            else:
                return self.assign_z(xyrec, auto, new_maxz)
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
                ang_to_unit(self.fishdata.headingangle), filter_sd))
        yaw_rad = np.radians(yaw_all)
        x_all = gaussian_filter(self.fishdata.x, filter_sd)
        y_all = gaussian_filter(self.fishdata.y, filter_sd)
        z_all = gaussian_filter(self.fishdata.z, filter_sd)
        for frame in range(len(x_all)):
            yaw = yaw_rad[frame]
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
        self.spherical_pitch = pitch_all
        self.spherical_yaw = yaw_rad
#        self.spherical_dyaw = dyaw_rad
# this will yield a vector normal to the shearing plane of the fish.

    def map_bout_to_heading(self, bout_id):
        bf = self.bout_frames[bout_id]
        post_ib = self.bout_frames[bout_id+1] - self.bout_frames[bout_id]
        bd = self.bout_durations[bout_id]
        delta_pitch = self.spherical_pitch[
                bf+bd] - self.spherical_pitch[bf]
        delta_yaw = calculate_delta_yaw_rad(
            self.spherical_yaw[bf], self.spherical_yaw[bf+bd])
        ufish = self.ufish[bf]
        upar = self.upar[bf]
        uperp = self.uperp[bf]
        origin_start = self.ufish_origin[bf]
        origin_end = self.ufish_origin[bf + bd]
        bout_az, bout_alt, bout_dist, nb_wrt_heading, ang3d = p_map_to_fish(
                ufish, origin_start, uperp, upar, origin_end, 0)
        return [bout_az, bout_alt,
                bout_dist, delta_pitch, delta_yaw, bd, post_ib]

    def watch_spherical_bouts(self, vidtype, sbouts):
        # Note you can just enter a subset of sbouts (or one) b/c of dict use
        if vidtype == 2:
            vid = imageio.get_reader(
                self.directory + '/side_contrasted.AVI', 'ffmpeg')
        elif vidtype == 1:
            vid = imageio.get_reader(
                self.directory + '/top_contrasted.AVI', 'ffmpeg')
        elif vidtype == 0:
            vid = imageio.get_reader(
                self.directory + '/conts.AVI', 'ffmpeg')
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        for s_bout in sbouts:
            firstframe = s_bout["Bout Frame"] - 10
            lastframe = s_bout["Bout Duration"] + s_bout["Bout Frame"] + 10
            if firstframe <= 0:
                    continue
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

    def spherical_huntbouts(self, fsbs, hd):
        shbs = []
        huntframes = [range(
            self.bout_frames[d[0]], self.bout_frames[d[1]] + 1)
                      for i, d in enumerate(
                              self.hunt_wins) if i in hd.hunt_ind_list]
        hf_concat = np.concatenate(huntframes)
        for sbout in fsbs:
            bf = sbout["Bout Frame"]
            if bf in hf_concat:
                shbs.append(sbout)
        return shbs

    def filtered_spherical_bouts(self, spherical_bouts):
        # filter for near walls and > 10% fish interpolation
        filt_sb = []
        for sbout in spherical_bouts:
            bf = sbout["Bout Frame"]
            bd = sbout["Bout Duration"]
            pct_interp = np.sum(self.nans_in_original[bf:bf+bd]) / bd
            if bf - 3 >= 0:
# more stringent than initial filtering b/c want to make sure bout call is excellent
                nw = self.nearwall(bf, bd, 200)
                if not nw and pct_interp <= .1:
                    filt_sb.append(sbout)
        return filt_sb

    def all_spherical_bouts(self, plot_or_not):
        # there is no interbout info for the last bout so ignore it.
        spherical_bouts = []
        empty_bouts = (self.bout_az == [])
        for b_num, b_frame in enumerate(self.bout_frames):
            if b_num == len(self.bout_frames) - 1:
                break
            s_bout = self.map_bout_to_heading(b_num)
            if empty_bouts:
                self.bout_az.append(s_bout[0])
                self.bout_alt.append(s_bout[1])
                self.bout_dist.append(s_bout[2])
                self.bout_dpitch.append(s_bout[3])
                self.bout_dyaw.append(-1*s_bout[4])
            spherical_bouts.append({'Bout Az': s_bout[0],
                                    'Bout Alt': s_bout[1],
                                    'Bout Dist': s_bout[2],
                                    'Interbout': s_bout[-1],
                                    'Bout Duration': s_bout[-2],
                                    'Delta Pitch': s_bout[3],
                                    'Delta Yaw': -1*s_bout[4],
                                    'Bout Frame': b_frame})
        if plot_or_not:
            b_dict = {'Bout Az': self.bout_az,
                      'Bout Alt': self.bout_alt,
                      'Bout Dist': self.bout_dist,
                      'Interbout': np.diff(self.bout_frames),
                      'Bout Duration': self.bout_durations,
                      'Delta Pitch': self.bout_dpitch,
                      'Delta Yaw': self.bout_dyaw}

            fig, axes = pl.subplots(1, len(spherical_bouts[0]),
                                    sharex=False,
                                    sharey=False,
                                    figsize=(8, 8))
            for ind, (title, entry) in enumerate(b_dict.iteritems()):
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
            rgba_vals = scalarMap.to_rgba(self.bout_dyaw)
            for i in range(len(self.bout_az) - 1):
                ax3d.plot([self.bout_az[i]],
                          [self.bout_alt[i]],
                          [self.bout_dist[i]],
                          color=rgba_vals[i],
                          marker='.',
                          ms=10*self.bout_dpitch[i])
            scalarMap.set_array(self.bout_dyaw)
            graph_3D.colorbar(scalarMap)
            pl.show()
        return spherical_bouts

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

        if self.paradata.decimated_vids:
            subscript = '_d'
        else:
            subscript = ''
        np.save(self.dirfectory + '/para3D' + str(
            h_index).zfill(2) + subscript + '.npy',
                self.paradata.para3Dcoords)
        np.save(self.directory + '/wrth' + str(
            h_index).zfill(2) + subscript + '.npy',
                para_wrt_heading)
        np.save(self.directory + '/wrth_xy' + str(
            h_index).zfill(2) + subscript + '.npy',
                wrth_xy)
        np.save('/Users/nightcrawler2/PreycapMaster/ufish.npy',
                self.ufish[self.framewindow[0]:self.framewindow[1]])
        np.save('/Users/nightcrawler2/PreycapMaster/uperp.npy',
                self.uperp[self.framewindow[0]:self.framewindow[1]])
        np.save('/Users/nightcrawler2/PreycapMaster/ufish_origin.npy',
                self.ufish_origin[self.framewindow[0]:self.framewindow[1]])
        np.save(
            '/Users/nightcrawler2/PreycapMaster/para_continuity_window.npy',
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
    

# tailangle will get nanified in any case where an xy variable is missed
# z will only get naned if z varbs aren't found. combining yields all nans

def count_nans(fishdata):
    nan_indices = []
    tail_seg = [x[0] for x in fishdata.tailangle]
    zcoord = [z for z in fishdata.z]
    for t, z in zip(tail_seg, zcoord):
        if math.isnan(t) or math.isnan(z):
            nan_indices.append(1)
        else:
            nan_indices.append(0)
    return nan_indices
        

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
    # this means nans at the end
    if nanstretch:
        end_nan_fill = np.ones(
            len(arr) - nan_start_ind) * arr[nan_start_ind - 1]
        arr_out[nan_start_ind:] = end_nan_fill
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

    
def create_poirec(h_index, two_or_three, directory, para_id, dec):
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
    

def pvec_wrapper(exp, hd, filter_sd):
    
    def concat_vecs(veclist):
        all_lengths = map(lambda a: len(a), veclist)
        all_vecs = reduce(lambda a, b: np.concatenate([a, b]), veclist)
        return all_lengths, all_vecs

    vecs = []
    vec_address = []
    nonan_indices = []
    p_id = 0
    for h, d in zip(hd.hunt_ind_list, hd.decimated_vids):
        print h
        if d:
            continue
        while True:
            try:
                v, no_nan_inds, spacing = para_vec_explorer(exp,
                                                            h, p_id,
                                                            0, filter_sd)
            except IndexError:
                p_id = 0
                break
            if v != []:
                nonan_indices.append(no_nan_inds)
                vec_address.append([h, p_id])
                vecs.append(v)
            p_id += 1
#    all_l, all_v = concat_vecs(vecs)
    np.save(exp.directory + '/para_vec_address.npy', vec_address)
    np.save(exp.directory + '/para_velocity_input.npy', vecs)
    np.save(exp.directory + '/no_nan_inds.npy', nonan_indices)
    np.save(exp.directory + '/spacing.npy', spacing)
    return vecs, vec_address


def para_vec_explorer(exp, h_id, p_id, animate, filter_sd):
    penv = ParaEnv(h_id, exp.directory, filter_sd, False)
    penv.find_paravectors(False)
    pvec = []
    non_nan_indices = []
    dp = []
    # what you might have to do here is define non-nan bounds. only
    #incorporate non-nan stretches. pomegranate doesn't take nans.
    p_index = 0

    # ** hunted_para_descriptor doesn't divide out vector spacing...
    # my metric says they must go 1.5 pixels per 3 frames to be moving.
    # this is extremely accurate. 
    # filt_velocity_mags = gaussian_filter(
    #     penv.velocity_mags[p_id][1:], 1) / penv.vector_spacing
    filt_velocity_mags = gaussian_filter(
        penv.velocity_mags[p_id][1:], 1)
    avg_vel = np.nanmedian(filt_velocity_mags)
#    top_percentile_vel = np.nanpercentile(filt_velocity_mags, 90)
#    print top_percentile_vel
    # accounts for a burst of velocity (i.e. swimming while otherwise still)
    # or a tumbler (avg vel > 1.5). immobile are all < 1.5. confirmed
    # many times by watching vids. 
 #   if top_percentile_vel > 3 or avg_vel > 1.5:
    if avg_vel > 2:
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
        penv = ParaEnv(hunt, exp.directory, 1, False)
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


# CALL THESE TWO FUNCTIONS AT THE END.             

def grab_all_spherical_bouts(fish_directories):
    sb_count_per_fish = []
    shb_count_per_fish = []
    all_spherical_bouts = []
    all_spherical_huntbouts = []
    for drct in fish_directories:
        try:
            rfo = pd.read_pickle(
                os.getcwd() + '/' + drct + '/RealHuntData_' + drct + '.pkl')
        except IOError:
            sb_count_per_fish.append(0)
            shb_count_per_fish.append(0)
            continue
        
        all_spherical_bouts += rfo.all_spherical_bouts
        all_spherical_huntbouts += rfo.all_spherical_huntbouts
        sb_count_per_fish.append(len(rfo.all_spherical_bouts))
        shb_count_per_fish.append(len(rfo.all_spherical_huntbouts))
    spherical_bout_dict = {"Spherical Bouts": all_spherical_bouts,
                           "Spherical HBs": all_spherical_huntbouts,
                           "SB Count": sb_count_per_fish,
                           "SHB Count": shb_count_per_fish,
                           "Fish": fish_directories}
    return spherical_bout_dict

            
def all_bout_data_to_csv(directories):
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
    firstind = exp.hunt_wins[hunt_ind][0]
    secondind = exp.hunt_wins[hunt_ind][1]    
    indrange = range(firstind, secondind+1, 1)
    #+1 so it includes the secondind
    print('Bout Ids')
    print range(firstind, secondind+1)
    print('Cluster Membership')
    print exp.cluster_membership[firstind:secondind+1]
    print('Bout Durations')
    print exp.bout_durations[firstind:secondind+1]
    print('Nans in Bouts')
    print [d[9] for d in exp.bout_flags[firstind:secondind+1]]
    filtV = gaussian_filter(exp.fishdata.vectV, 0)
    # yaw_all_filt = unit_to_angle(
    #     filter_uvec(
    #             ang_to_unit(exp.fishdata.headingangle), 1))
    # yaw_all = unit_to_angle(
    #     filter_uvec(
    #             ang_to_unit(exp.fishdata.headingangle), 0))

    start = exp.bout_frames[firstind]-integ_win
    end = exp.bout_frames[secondind]+integ_win
    framerange = range(start, end)
    # gives last bout 500ms to occur
    fig, ((ax1, ax2),
          (ax3, ax4)) = pl.subplots(
              2, 2,
              sharex=True,
              figsize=(6, 6))
    filt_phir = gaussian_filter(exp.fishdata.phiright, 2)[start:end]
    filt_phil = gaussian_filter(exp.fishdata.phileft, 2)[start:end]
    ax1.plot(framerange, filt_phir, color='g')
    ax1.plot(framerange, filt_phil, color='r')
    tailang = [t[-1] for t in exp.fishdata.tailangle]
    ax3.plot(framerange, tailang[start:end])
    ta_std = gaussian_filter([np.nanstd(tw) for tw in sliding_window(5, tailang)],2)
    ax3.plot(framerange, ta_std[start:end])
    vel_during_hunt = filtV[start:end]
    ax3.plot(framerange, vel_during_hunt)
    bouts_tail = [exp.bout_frames[i] for i in indrange]
    bouts_tail_end = [
        exp.bout_frames[i] + exp.bout_durations[i] for i in indrange]
    print len(bouts_tail)
    ax3.plot(bouts_tail,
             np.zeros(len(bouts_tail)),
             marker='.',
             ms=10,
             color='c')
    ax3.plot(bouts_tail_end,
             np.zeros(len(bouts_tail_end)),
             marker='.',
             ms=10,
             color='m')
    for ind, typ in enumerate(exp.cluster_membership[firstind:secondind+1]):
        ax3.text(bouts_tail[ind], -.5, str(typ))
    pitch_during_hunt = exp.spherical_pitch[start:end]
    yaw_during_hunt = exp.spherical_yaw[start:end]
#    yaw_filt_during_hunt = yaw_all_filt[start:end]
    z_during_hunt = gaussian_filter(exp.fishdata.z[start:end], 1)
    ax2.plot(framerange, yaw_during_hunt, color='m')
    ax2.plot(framerange, pitch_during_hunt, color='k')
    ax4.plot(framerange, z_during_hunt, color='b')
    pl.tight_layout()
    if plotornot == 0:
        pl.clf()
        pl.close()
    elif plotornot == 1:
        pl.savefig('bouts_during_hunt.pdf')
        pl.show()
    return pitch_during_hunt, yaw_during_hunt, z_during_hunt, filt_phil, filt_phir


def hunted_para_descriptor(exp, hd):

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
              'Para Dist Velocity',
              'Para Az Accel',
              'Para Alt Accel',
              'Para Dist Accel',
              'Postbout Para Az',
              'Postbout Para Alt',
              'Postbout Para Dist',
              'Strike Or Abort',
              'Inferred',
              'Avg Para Velocity',
              'Percent Nans in Bout']
    int_win = exp.integration_window
    cont_win = exp.para_continuity_window
    bout_descriptor = []
    df_labels = ["Bout Az", "Bout Alt",
                 "Bout Dist", "Bout Delta Pitch", "Bout Delta Yaw"]
    realfish = RealFishControl(exp)
    for hi, hp, ac, br, iws in zip(
            hd.hunt_ind_list,
            hd.para_id_list, hd.actions, hd.boutrange, hd.interp_windows):
        print('Hunt ID')
# this is a catch for not knowing the para
        if ac == 0 or hi in hd.dec_doubles:
            continue
        if hd.check_for_doubles(hi):
            subscript = '_d'
            penv_dec = True
        else:
            subscript = ''
            penv_dec = False
        para_xyz = np.load(
            exp.directory + "/para3D" + str(hi).zfill(
                2) + subscript + ".npy")[
                    hp*3:hp*3 + 3][
                        :, cont_win+int_win-realfish.firstbout_para_intwin:]
        penv = ParaEnv(hi, exp.directory, penv_dec, 1)
        penv.find_paravectors(False, hp)
# raw_para_vel = penv.velocity_mags[0][exp.para_continuity_window / penv.vector_spacing:]
# can index raw_para_vel with norm_bf / penv.vector_spacing.
        avg_vel = np.nanmean(penv.velocity_mags[0][
            exp.para_continuity_window / penv.vector_spacing:])
        hunt_df = pd.DataFrame(columns=df_labels)
        poi_wrth = create_poirec(hi, 3, exp.directory, hp)
        dist = [pr[4] for pr in poi_wrth]
        az = [pr[6] for pr in poi_wrth]
        alt = [pr[7] for pr in poi_wrth]
        # filtering here with 2 b/c velocities can be created by noise.
        # 2 suffices but scipy gaussian can't handle nans, and drops
        # para once it disappears too early. used astropy filter instead
        # which interpolates, but it interpolates to infinity. so have to
        # list comp the nans. 
        filter_sd = 2
        kernel = Gaussian1DKernel(filter_sd)
        filt_az = convolve(az, kernel, preserve_nan=True)
        # note i later found the preserve_nans key -- change if you want for
        # brevity. 
        filt_alt = convolve(alt, kernel, preserve_nan=True)
        filt_dist = convolve(dist, kernel, preserve_nan=True)
        if len(filt_az) < 2 or len(filt_dist) < 2:
            continue
        hunt_bouts = range(exp.hunt_wins[hi][0],
                           exp.hunt_wins[hi][1]+1)
        nans_in_bouts = [d[9] for d
                         in exp.bout_flags[
                             exp.hunt_wins[hi][0]:exp.hunt_wins[hi][1]+1]]
        hunt_bout_frames = [exp.bout_frames[i] for i in hunt_bouts]
        hunt_bout_durations = [exp.bout_durations[i] for i in hunt_bouts]
        percent_nans = np.array([na / float(bd) for na, bd
                                 in zip(nans_in_bouts, hunt_bout_durations)])
        norm_bf_raw = [hbf - hunt_bout_frames[0] for hbf in hunt_bout_frames]
        # int win is added because createpoirec cointains it. have to go beyond        # it to get relevant para coords. 
        norm_bf = map(lambda(x): x+int_win, norm_bf_raw)
        # these are normed to the hunting bout so that first bout is 0.
        endhunt = False
        for hb_ind, bout in enumerate(hunt_bouts):
            if br[0] != 0:
                hb_ind += br[0]
                bout += br[0]
                print('altered index')
                print hb_ind

            norm_frame = norm_bf[hb_ind]
            bout_dur = exp.bout_durations[bout]
            inferred_coordinate = 0
            for infwin in iws:
                if np.intersect1d(
                        range(
                            norm_frame-realfish.firstbout_para_intwin,
                            norm_frame),
                        infwin).any():
                    inferred_coordinate = 1
            delta_pitch = exp.bout_dpitch[bout]
            delta_yaw = exp.bout_dyaw[bout]
            para_az = filt_az[norm_frame]
            para_alt = filt_alt[norm_frame]
            para_dist = filt_dist[norm_frame]
            pxyz_temp = para_xyz[
                :, norm_bf_raw[
                    hb_ind]:norm_bf_raw[
                        hb_ind]+realfish.firstbout_para_intwin].T
            uf = exp.ufish[hunt_bout_frames[hb_ind]]
            ufish_origin = exp.ufish_origin[hunt_bout_frames[hb_ind]]
            upar = exp.upar[hunt_bout_frames[hb_ind]]
            uperp = exp.uperp[hunt_bout_frames[hb_ind]]
            pmap_returns = []
            for p_xyz in pxyz_temp:
                pmap_returns.append(p_map_to_fish(uf,
                                                  ufish_origin,
                                                  uperp, upar, p_xyz, 0))
            para_daz = gaussian_filter(np.diff(
                [x[0] for x in pmap_returns]), 1) / .015
            para_dalt = gaussian_filter(
                np.diff([x[1] for x in pmap_returns]), 1) / .015
            para_ddist = gaussian_filter(
                np.diff([x[2] for x in pmap_returns]), 1) / .015
            para_az_accel = np.diff(para_daz)
            para_alt_accel = np.diff(para_dalt)
            para_dist_accel = np.diff(para_ddist)
            # this exception is no longer required as I built the last bout duration into the para_during_hunt function
#            try:
            postbout_az = filt_az[norm_frame+bout_dur]
            postbout_alt = filt_alt[norm_frame+bout_dur]
            postbout_dist = filt_dist[norm_frame+bout_dur]
        # except IndexError:
            #     postbout_az = filt_az[-1]
            #     postbout_alt = filt_alt[-1]
            #     postbout_dist = filt_dist[-1]
            if br[1] < 0:
                if hb_ind == len(hunt_bouts) + br[1]:
                    endhunt = True
                    last_bout = br[1]
            else:
                if hb_ind == br[1]:
                    endhunt = True
                    last_bout = hb_ind - len(hunt_bouts)
            bout_descriptor.append([hi,
                                    hb_ind,
                                    exp.bout_az[bout],
                                    exp.bout_alt[bout],
                                    exp.bout_dist[bout],
                                    delta_pitch,
                                    delta_yaw,
                                    para_az,
                                    para_alt,
                                    para_dist,
                                    np.nanmean(para_daz),
                                    np.nanmean(para_dalt),
                                    np.nanmean(para_ddist),
                                    np.nanmean(para_az_accel),
                                    np.nanmean(para_alt_accel),
                                    np.nanmean(para_dist_accel),
                                    postbout_az,
                                    postbout_alt,
                                    postbout_dist,
                                    ac,
                                    inferred_coordinate,
                                    avg_vel,
                                    percent_nans[hb_ind]])
#            if ind != -1:
            hunt_df.loc[hb_ind] = [exp.bout_az[bout],
                                   exp.bout_alt[bout],
                                   exp.bout_dist[bout],
                                   delta_pitch,
                                   delta_yaw]
            if endhunt:
                bout_descriptor[-1][1] = last_bout
                para_velocity = bout_descriptor[-1][-2]
                # THIS IS WRONG. IF REC ENDS BEFORE DECONVERGE (i.e. w/ -2 index) THIS WILL GO BACK TOO FAR
                # INTO THE PREVIOUS PARA. 
                first_bout_in_hunt_index = (-1 * (hb_ind+1)) + br[0]
                print first_bout_in_hunt_index
                print("first_bout_in_hunt_index")
                para_interp_list = [
                    b[-3] for b in bout_descriptor[
                            first_bout_in_hunt_index:]] 
                prcnt_para_interp = np.sum(
                    para_interp_list).astype(float) / len(para_interp_list)
                pararec_present_at_outset = np.isfinite(
                    bout_descriptor[first_bout_in_hunt_index][7:10]).all()
                # anything less than 1.5 is stationary
                if para_velocity > 1.5 and pararec_present_at_outset and (
                        percent_nans < 1.0 / 2).all() and (
                            prcnt_para_interp <= 1.0 / 3) and 0 < ac < 4:
                    realfish.hunt_ids.append(hi)
                    realfish.hunt_frames.append(
                        (hunt_bout_frames[br[0]],
                         hunt_bout_frames[hb_ind] +
                         hunt_bout_durations[hb_ind]))
                    realfish.hunt_interbouts.append(
                        [0] +
                        np.diff(hunt_bout_frames[br[0]:hb_ind + 1]).tolist())
                    realfish.huntbout_durations.append(
                        hunt_bout_durations[br[0]:hb_ind+1])
                    realfish.hunt_results.append(ac)
#                    realfish.para_xyz_per_hunt.append(para_xyz)
                    realfish.para_xyz_per_hunt.append(
                        para_xyz[
                            :, norm_bf_raw[br[0]]:norm_bf_raw[hb_ind] + hunt_bout_durations[
                                    hb_ind] + realfish.firstbout_para_intwin])
                    realfish.hunt_dataframes.append(copy.deepcopy(hunt_df))
                    print pararec_present_at_outset
                    print prcnt_para_interp
                    print percent_nans
                    print para_velocity

                else:
                    print('rejected para or too much interp')
                    print pararec_present_at_outset
                    print prcnt_para_interp
                    print percent_nans
                    print para_velocity
                break
    sbouts = exp.all_spherical_bouts(False)
    fsb = exp.filtered_spherical_bouts(sbouts)
    shbs = exp.spherical_huntbouts(fsb, hd, dim)
    # nhbs = [f for f in fsb if f not in shbs]
    realfish.all_spherical_bouts = fsb
    realfish.all_spherical_huntbouts = shbs
#    velocity_kernel(exp, 'hunts', hd, dim)
#    yaw_kernel(exp, 'hunts', hd, dim)
    realfish.exporter()
    csv_data(header, bout_descriptor, 'huntingbouts', exp.directory)
    return realfish


def para_stimuli(exp, hd):

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
    decimation = hd.decimated_vids
    stim_list = []

    for h, hp, ac, d in zip(hunt_ind_list,
                            p_list, actions, decimation):
        print h
        # here you will create a bout frames list for the entire hunt. once you
        # get to the hunted para,
        if d:
            continue
        p3D = np.load(exp.directory + '/para3D' + str(h).zfill(2) + '.npy')
        distmat = make_distance_matrix(p3D)
        penv = ParaEnv(h, exp.directory, False, 1)
        penv.find_paravectors(False)
        wrth = np.load(
            exp.directory + '/wrth' + str(h).zfill(2) + '.npy')

# HERE IS WHERE THE INTERVAL GETS SET.
# When this is a function, your flags here will include whether its an init or an abort. 

        if ac >= 4:
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

# THIS FUNCTION REQUIRES UPDATING. EACH BOUT IS ALREADY IMPLICIT GIVEN CALCULATIONS OF UFISH, ETC.
# REDO THIS FUNCTION


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


def calculate_delta_yaw_rad(yaw_init, yaw_end):
    if np.abs(yaw_end-yaw_init) < np.pi:
        diff = yaw_end - yaw_init
    elif yaw_end-yaw_init <= - np.pi:
        diff = yaw_end + (2*np.pi-yaw_init)
    elif yaw_end-yaw_init >= np.pi:
        diff = -(2*np.pi-yaw_end)+yaw_init
    return diff


def calculate_delta_yaw(raw_yaw):
    delta_yaw = [0]
    for h in sliding_window(2, raw_yaw):
        if abs(h[1]-h[0]) < 180:
            delta_yaw.append(h[1]-h[0])
        elif h[1]-h[0] <= -180:
            diff = h[1] + (360-h[0])
            delta_yaw.append(diff)
        elif h[1]-h[0] >= 180:
            diff = -(360-h[1])+h[0]
            delta_yaw.append(diff)
        elif math.isnan(h[1]) or math.isnan(h[0]):
            delta_yaw.append(float('nan'))
    return delta_yaw



# before implementing this in full in the bout detector,
# add these extensions to the current bout_durations as
# indicated in the main line comments.(i.e. first
# create a new myexp and then add the bout extension). 
# then run hunted_para_descriptor which will output
# a new realfish object for the modeling.
# 

def yaw_kernel(exp, all_or_hb, hd):
    yaw_profiles = []
    yaw = exp.spherical_yaw
    bd = 7
    if all_or_hb == 'all':
        for bf in exp.bout_frames:
            ydiff = np.abs(yaw[bf+bd] - yaw[bf])
            if .1 < ydiff or ydiff > .5:
                continue
            norm_yaw = (np.array(yaw[bf:bf+bd]) - yaw[bf]) 
            if norm_yaw[5] < 0:
                norm_yaw *= -1
            yaw_profiles.append(norm_yaw)
    elif all_or_hb == 'hunts':
        h_inds = hd.hunt_ind_list
        huntbouts = np.concatenate(
            [range(exp.hunt_wins[h][0], exp.hunt_wins[h][1]+1)
             for h in h_inds], axis=0)
        huntframes = [exp.bout_frames[hb] for hb in huntbouts]
        for hf in huntframes:
            ydiff = np.abs(yaw[hf+bd] - yaw[hf])
            if .05 < ydiff or ydiff > .3:
                continue
            norm_yaw = (np.array(yaw[hf:hf+bd]) - yaw[hf]) 
            if norm_yaw[5] < 0:
                norm_yaw *= -1
            yaw_profiles.append(norm_yaw)
    final_yaw_profile = np.sum(yaw_profiles, axis=0) / len(yaw_profiles)
    # pl.plot(final_yaw_profile)
    # pl.show()
    if all_or_hb == 'hunts':
        np.save('hb_yaw_kernel.npy', np.diff(final_yaw_profile))
    elif all_or_hb == 'all':
        np.save('yaw_kernel.npy', np.diff(final_yaw_profile))
    return np.diff(final_yaw_profile)

    # if all_or_hb == 'hunts':
    #     np.save('hb_velocity_kernel.npy', final_vel_profile)
    # elif all_or_hb == 'all':
    #     np.save('velocity_kernel.npy', final_vel_profile)
    # return final_vel_profile
    

def velocity_kernel(exp, all_or_hb, hd):

    filtV = gaussian_filter(exp.fishdata.vectV, 0)
    vel_profiles = []
    if all_or_hb == 'all':
        for bf, bd in zip(exp.bout_frames, exp.bout_durations):
            vel_profiles.append(filtV[bf:bf+bd])
    elif all_or_hb == 'hunts':
        h_inds = hd.hunt_ind_list
        huntbouts = np.concatenate(
            [range(exp.hunt_wins[h][0], exp.hunt_wins[h][1]+1)
             for h in h_inds], axis=0)
        huntframes = [exp.bout_frames[hb] for hb in huntbouts]
        bd_hunts = [exp.bout_durations[hb] for hb in huntbouts]
        for hf, hd in zip(huntframes, bd_hunts):
            vel_profiles.append(filtV[hf:hf+hd])
    max_len_window = np.max([len(vp) for vp in vel_profiles])
    print max_len_window
    padded_profiles = [np.pad(
        vp, (
            0, max_len_window-len(vp)),
        mode='constant') for vp in vel_profiles]
    final_vel_profile = np.sum(padded_profiles, axis=0) / len(padded_profiles)
#    pl.plot(final_vel_profile)
#    pl.show()
    if all_or_hb == 'hunts':
        np.save('hb_velocity_kernel.npy', final_vel_profile)
    elif all_or_hb == 'all':
        np.save('velocity_kernel.npy', final_vel_profile)
    return final_vel_profile


def normalize_kernel(kernel, length):
    shortened_kernel = np.array(kernel[0:length])
    sum_kernel = np.sum(shortened_kernel)
    normed_kernel = shortened_kernel / sum_kernel
#    pl.plot(normed_kernel)
#    pl.show()
    return normed_kernel


def normalize_kernel_interp(kernel, length):
    l_ratio = float(length) / kernel.shape[0]
    cand = np.arange(0, l_ratio*kernel.shape[0], l_ratio)
    if cand.shape[0] > kernel.shape[0]:
        interp_range = cand[0:-1]
    else:
        interp_range = cand
    interp_func = interp1d(
        interp_range, kernel, 'linear', fill_value='extrapolate')
    new_kernel = np.array(
        [interp_func(i) for i in range(length)])
    normed_kernel = new_kernel / np.sum(new_kernel)
    return normed_kernel

def find_velocity_ends(exp, v_thresh, b_or_bplus1):
    bd_plus = []
    filt_v = gaussian_filter(exp.fishdata.vectV, 0)
    interbouts = np.diff(exp.bout_frames)
    thresh_vlist = []
    for bout_ind, (bf, bd) in enumerate(
            zip(exp.bout_frames, exp.bout_durations)):
        if bout_ind == interbouts.shape[0]:
            bd_plus.append(bd)
            break
#        v_win = filt_v[bf:bf+bd]
        v_win_end = bd + 5
        v_win = filt_v[bf:bf+v_win_end]
        v_max = np.max(v_win)
        v_argmax = np.argmax(v_win)
        v_start = filt_v[bf]
        v_start_boutplus1 = filt_v[exp.bout_frames[bout_ind+1]]
        if b_or_bplus1 == 0:
            dv = v_max - v_start
            thresh_v = (dv * v_thresh) + v_start
        elif b_or_bplus1 == 1:
            dv = v_max - v_start_boutplus1
            thresh_v = (dv * v_thresh) + v_start_boutplus1
        # i is the distance from bf to the threshV.
        # if i is less than the interbout,
        # add i - bd to the prev called bout duration

# has to be a catch here for when the if statement is never satisfied.
# in pure rotation, there will be no velocity. have to note this.
        v_thresh_found = False
        for i, v in enumerate(filt_v[bf:]):
            # can only go in here if v requirements are satisfied. 
            if v < thresh_v and i > v_argmax:
                v_thresh_found = True
                candidate_bd_extension = i - bd
                if candidate_bd_extension <= 0:
                    bd_plus.append(0)
                elif i < interbouts[bout_ind]:
                    bd_plus.append(candidate_bd_extension)
                else:
                    bd_plus.append(interbouts[bout_ind] - 1 - bd)
                thresh_vlist.append(thresh_v)
                break
        # this is a catch for a pure rotation's velocity actually being extremely low. 
        if not v_thresh_found:
            bd_plus.append(interbouts[bout_ind] - 1 - bd)
    return bd_plus, thresh_vlist


def validate_experiment(drct):
    fishdata = pickle.load(open(
           drct + '/fishdata.pkl', 'rb'))
    checkpoints = [1000, 11000, 21000, 31000]
    velocities_at_checkpoints = []
    xyvelocities_at_checkpoints = []
    # 3 second window
    len_pwindow = 180
    for ch in checkpoints:
        para_validate = return_paramaster_object(ch, ch+len_pwindow,
                                                 False, drct + '/',
                                                 False, 0)
        # 0s are dummy variables b/c no specific reconstruction specified
        penv3D = ParaEnv(0, 0, 1, False, (para_validate.para3Dcoords, 3))
        penv3D.find_paravectors(False)
        unpaired_xy = [xyr[2] for xyr in para_validate.unpaired_xy if
                       len(xyr[2]) == len_pwindow]
        p_xy_matrix = np.zeros([len(unpaired_xy) * 2, len_pwindow])
        for row_id, xyr in enumerate(unpaired_xy):
            x, y = zip(*xyr)
            p_xy_matrix[2*row_id] = np.array(x)
            p_xy_matrix[2*row_id + 1] = np.array(y)
        penvXY = ParaEnv(0, 0, 1, False, (p_xy_matrix, 2))
        penvXY.find_paravectors(False)
        checkpoint_velocities = []
        checkpoint_xyvelocities = []
        for pv3d in penv3D.velocity_mags:
            velocity_3Dmags = gaussian_filter(pv3d[1:], 1)
            checkpoint_velocities.append(np.nanmedian(velocity_3Dmags))
        for pv in penvXY.velocity_mags:
            velocity_XYmags = gaussian_filter(pv[1:], 1)
            checkpoint_xyvelocities.append(np.nanmedian(velocity_XYmags))
        velocities_at_checkpoints.append(checkpoint_velocities)
        xyvelocities_at_checkpoints.append(checkpoint_xyvelocities)
    f, (ax1, ax2, ax3) = pl.subplots(3)
    sb.pointplot(data=velocities_at_checkpoints, ci=95, ax=ax1)
    sb.pointplot(data=xyvelocities_at_checkpoints, ci=95, ax=ax2)
    ax1.set_title('Para Velocity')
    ax2.set_title('Para XY Velocity')
    filt_z = gaussian_filter(fishdata.low_res_z, 5)
    ax3.plot(filt_z)
    ax3.set_title('Avg Z = ' + str(np.nanmean(fishdata.low_res_z[8000:])))
    pl.tight_layout()
    pl.savefig(drct + '_validation.pdf')
    return velocities_at_checkpoints, xyvelocities_at_checkpoints, filt_z


# First you make calls based purely on tail variance.
# This defines the fact that a movement occurred.
# Tail movement is far more reliable for a movement
# b/c velocity of hunt bouts is so slow (w/ rotations)
# that you would miss them if you thresholded on velocity
# however, there IS a dynamic to the velocity once you've
# defined that a movement occurred. call find_velocity_end
# on the bouts. it will return a series of extensions to the
# current bout set. each extension must be added to the
# current bout_durations if and only if the total bout duration
# is less than the interbout. if it is not, make the bout
# duration one less than the interbout. 

# BE SURE TO RE-KICK OUT ANY OVERLAP
def plot_bout_calls(exp, extension, tvl, *hunt_win):

    if hunt_win != ():
        start = hunt_win[0][0]
        end = hunt_win[0][1]
        bout_frames = np.array(
            exp.bout_frames[start:end]) - exp.bout_frames[start]
        b_durs = np.array(exp.bout_durations[start:end])
        extension = extension[start:end]
        print("threshold velocity list")
        print tvl[start:end]
    else:
        start = 0
        end = -1
        bout_frames = np.array(exp.bout_frames) - exp.bout_frames[start]
        b_durs = np.array(exp.bout_durations)
        
    bdurs = b_durs + extension
    tailang = [
        tail[-1] for tail in exp.fishdata.tailangle[
            exp.bout_frames[start]:exp.bout_frames[end]]]
    ta_std = gaussian_filter([np.nanstd(tw) for tw in sliding_window(5, tailang)], 2)
    pl.plot(tailang)
    pl.plot(gaussian_filter(
        exp.fishdata.vectV[
            exp.bout_frames[start]:exp.bout_frames[end]], 1))
    pl.plot(ta_std)
    pl.plot(
        bout_frames, np.zeros(len(bout_frames)),
        marker='.', linestyle='None',
        color='c')
    pl.plot(
        [b+d for b, d in zip(
            bout_frames, bdurs)],
        np.zeros(len(bout_frames)),
        marker='.', linestyle='None', color='m')
    pl.plot(
        [b+d for b, d in zip(
            bout_frames, b_durs)],
        np.zeros(len(bout_frames)),
        marker='.', linestyle='None', color='k')
    pl.show()


def return_ta_std(drc_list):
    all_ta = []
    for drc in drc_list:
        print drc
        fishdata = pickle.load(open(
            drc + '/fishdata.pkl', 'rb'))
        temp_tail = np.array(fishdata.tailangle)
        for i in range(len(temp_tail[0])):
            temp_tail[:, i] = fill_in_nans(temp_tail[:, i])
        tailangle = temp_tail.tolist()
        tailang_lastseg = [ta[-1] for ta in tailangle]
        all_ta = all_ta + tailang_lastseg
    ta_std = gaussian_filter(
            [np.nanstd(tw) for tw in sliding_window(5, all_ta)], 2).tolist()
    threshold = 2*np.round(np.median(ta_std))
    return threshold


def exp_generation_and_clustering(fish_drct_list,
                                  all_varbs, cluster_varbs, flag_varbs, new_exps):
    all_bout_data = []
    all_flags = []
    all_bout_frames = []
    add_vel_ends = True
    for drct in fish_drct_list:
        if new_exps:
            fish_directory = os.getcwd() + '/' + drct
            myexp = Experiment(20, 3, all_varbs_dict,
                               flag_dict, fish_directory)
            myexp.bout_detector()
            myexp.bout_nanfilt_and_arrange(False)
            print("Creating Unit Vectors")
            myexp.create_unit_vectors()
            if add_vel_ends:
                vel_extensions, tvl = find_velocity_ends(myexp, .05, 1)
                myexp.bout_durations_tail_only = copy.deepcopy(
                    myexp.bout_durations)
                myexp.bout_durations = np.array(
                    myexp.bout_durations) + vel_extensions
            myexp.all_spherical_bouts(False)
            myexp.exporter()
        else:
            myexp = pickle.load(open(drct + '/master.pkl', 'rb'))
            try:
                myexp.hunt_wins = np.load(myexp.directory + '/hunt_wins.npy')
            except IOError:
                pass
        rejected_bout_inds = [b[0] for b in
                              myexp.rejected_bouts if b[1] == 'nearwall']
        bout_data_no_rejects = []
        bout_flag_no_rejects = []
        non_reject_frames = []
        for b_ind, (bout_d, bout_f) in enumerate(
                zip(myexp.bout_data, myexp.bout_flags)):
            if b_ind not in rejected_bout_inds:
                bout_data_no_rejects.append(bout_d)
                bout_flag_no_rejects.append(bout_f)
                non_reject_frames.append(myexp.bout_frames[b_ind])
        all_bout_data.append(bout_data_no_rejects)
        all_flags.append(bout_flag_no_rejects)
        all_bout_frames.append(non_reject_frames)
    myexp = []
    dim = DimensionalityReduce(cluster_varbs, flag_varbs, all_varbs,
                               fish_drct_list, all_bout_data,
                               all_flags, all_bout_frames)
    dim.prepare_records()
    dim.exporter()
#    dim.dim_reduction(2)
 #   dim.exporter()
    return dim


def cluster_call_wrapper(dim, fish_drct_list):
    for drct in fish_drct_list:
        myexp = pickle.load(open(drct + '/master.pkl', 'rb'))
        extract_cluster_calls_to_exp(dim, myexp)


def extract_cluster_calls_to_exp(dim, exp):
    rejected_bout_inds = [b[0] for b in
                          exp.rejected_bouts if b[1] == 'nearwall']
    cluster_mem = []
    fish_id = dim.fish_id_list.index(exp.directory[-8:])
    cluster_id_deque = deque(dim.cmem_by_fish[fish_id])
    for raw_bout_id in range(len(exp.bout_data)):
        if raw_bout_id in rejected_bout_inds:
            cluster_mem.append(-1)
        else:
            c = cluster_id_deque.popleft()
            cluster_mem.append(c)
    exp.cluster_membership = cluster_mem
    exp.exporter()


def finish_experiment(exp, hd):
    exp.exporter()
    hd.exporter()
    hunted_para_descriptor(exp, hd)
    para_stimuli(exp, hd)
    v, va = pvec_wrapper(exp, hd, 1)


if __name__ == '__main__':

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
    sub_dict = {'8': 'Vector Velocity'}

    tail_dict = {'2': 'Tail Segment 2',
                 '3': 'Tail Segment 3',
                 '4': 'Tail Segment 4',
                 '5': 'Tail Segment 5',
                 '6': 'Tail Segment 6',
                 '7': 'Tail Segment 7'}

    abort_dict = {'13': 'Eye Sum'}

    bdict_2 = {'11': 'Eye1 Angle',
               '12': 'Eye2 Angle'}


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
        '9': 'Number of Nans',
        '10': 'Cluster ID'}

#        '9': 'Fish ID',
 


# Experiment arg is how many continuous frames are required for a bout call


# MAIN LINE: 


# Experiment class first 2 args define the minimum length of the bout and the amount of frames to go backwards for para calculation. 
# The bout_detector call finds bouts based on either orientation changes ('orientation') or on tail standard deviation changes ('tail')
# Nan_filt_and_arrange then finds whether the bouts detected are composed of Nan-Free fish variables 
# and the function (if flag is True) filters for bouts where fish is facing a tank edge. The bout variables are then stored in a continuous 
# bout array, matched with a flag array that describes summary statistics for each bout. A new BoutsandFlags object is then created
# whose only role is to contain the bouts and corresponding flags for each fish. 

    fish_id = '090418_3'
    drct = os.getcwd() + '/' + fish_id
    import_exp = False
    import_dim = True
    if import_exp:
        myexp = pickle.load(open(drct + '/master.pkl', 'rb'))
        try:
            myexp.hunt_wins = np.load(drct + '/hunt_wins.npy')
        except IOError:
            pass
        try:
            hd = hd_import(drct)
        except IOError:
            hd = Hunt_Descriptor(drct)
    if import_dim:
        dim = pickle.load(open(os.getcwd() + '/dim_reduce.pkl', 'rb'))
#     sbouts = myexp.all_spherical_bouts()

    new_wik = ['090418_3', '090418_4', '090418_5', '090418_6',
               '090418_7', '090518_1', '090518_2', '090518_3', '090518_4',
               '090518_5', '090518_6', '090618_1', '090618_2', '090618_3',
               '090618_4', '090618_5', '090618_6', '090718_1', '090718_2',
               '090718_3', '090718_4', '090718_5', '090718_6', '091118_1',
               '091118_2', '091118_3', '091118_4', '091118_5', '091118_6',
               '091218_1', '091218_2', '091218_3', '091218_4', '091218_5',
               '091218_6', '091318_1', '091318_2', '091318_3', '091318_4',
               '091318_5', '091318_6', '091418_1', '091418_2', '091418_3',
               '091418_4', '091418_5', '091418_6']

    new_wik_subset = ['090418_3', '090418_4', '090418_5',
                      '090418_6', '090418_7']

#    dim = exp_generation_and_clustering(new_wik, all_varbs_dict,
#                                        bout_dict, flag_dict, False)
# pilot = ['070617_1', '070617_2','072717_1', '072717_2', '072717_5']

# vcp_list = []
# xvcp_list = []
# lrz_list = []
# for drct in new_wik:
#     vcp, xvcp, lrz = validate_experiment(drct)
#     vcp_list.append(vcp)
#     xvcp_list.append(xvcp)
#     lrz_list.append(lrz)


# f, (ax1, ax2, ax3) = pl.subplots(3)
# velocities_at_checkpoints = np.array(zip(*vcp_list))
# vac = np.array([np.concatenate(v) for v in velocities_at_checkpoints])
# xyvelocities_at_checkpoints = np.array(zip(*xvcp_list))
# xyvac = np.array([np.concatenate(v) for v in xyvelocities_at_checkpoints])
# sb.pointplot(data=vac, ci=95, ax=ax1)
# sb.pointplot(data=xyvac, ci=95, ax=ax2)
# ax1.set_title('Para Velocity')
# ax2.set_title('Para XY Velocity')
# sb.tsplot(data=lrz_list, ci=95, ax=ax3)
# ax3.set_title('Z Position')
# pl.tight_layout()
# pl.savefig('new_wik.pdf')


