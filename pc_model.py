import csv
import gc
import copy
import numpy as np
import pickle
import datetime as dt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib import pyplot as pl
from matplotlib import gridspec
import statsmodels.api as sm
import matplotlib.cm as cm
import seaborn as sb
import pandas as pd
import os
import math
from toolz.itertoolz import partition
from para_hmm_final import ParaMarkovModel
import collections
import functools
import pomegranate
import bayeslite as bl
from iventure.utils_bql import query
from iventure.utils_bql import subsample_table_columns
from collections import deque, Counter
from master import fishxyz_to_unitvecs, sphericalbout_to_xyz, p_map_to_fish, RealFishControl, normalize_kernel, normalize_kernel_interp
from toolz.itertoolz import sliding_window
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, ttest_ind, ttest_rel, wilcoxon, ttest_1samp, mannwhitneyu, pearsonr

# Notes: bayesdb, regardless of the seed you provide at opening,
# will yield the same result each time if you re-create the
# class. If you run marr2bayes on the same instance,
# a second time, it will produce a different result.


# make a nancount for each model, then nanfilter
# only show pvalues as list
# show means of each distribution as list

#note that algorithm should be solution to, where x is the strike mean:
# there is a family of solutions m, b of the form:

# mx + b = x
# (1 - m) * strike_mean = b

# strike_mean = b / (1 - m)




# small slope clearly best for getting para closest fastest, but
# question is whether para speed is matched to the choice. 


# however m is only reasonable if it is < 1, otherwise get blowup on edges? 
# show 95% confindence interval for fit to fish. say that the alg the
# models use is within it.

# Note that the regplot class uses statsmodels GLM instead of OLS.
# So your regplot fits will be slightly different. Consider relative benefits, and
# also consider robust. 

class Marr2Algorithms:
    def __init__(self, para_positions, strike_params, use_likelihood):
        self.para_positions = para_positions
        self.bouts_static = []
        self.bouts_bayes = []
        self.marr_bdb_file = bl.bayesdb_open('wik_bdb/bdb_hunts_marr.bdb')
        self.all_strikes_csv = pd.read_csv(
            'wik_bdb/all_huntbouts_rev_strikes_nob0_nostrikes.csv')
        self.marr_bdb_file.np_prng.seed()
        self.number_of_models_in_ensemble = len(
                self.marr_bdb_file.backends['cgpm']._engine(
                    self.marr_bdb_file, 1).states)
        self.use_likelihood = use_likelihood
        if not use_likelihood:
            self.marr_bdb_file.backends['cgpm'].set_multiprocess(False)
        else:
            self.marr_bdb_file.backends['cgpm'].set_multiprocess(True)
        self.strike_params = strike_params
        self.strike_means = strike_params[0]
        self.strike_std = strike_params[1]
        self.static_params = {'az_m': .54,
                              'alt_m_pos': .56,
                              'alt_m_neg': 1.1,
                              'dist_m': .88,
                              'az_b': 0,
                              'alt_b_pos': .14,
                              'alt_b_neg': .15,
                              'dist_b': -8}
        self.strike_ci = .05
        self.test_results = []
        self.fails = []
        self.averages = []
        self.prebout_para = []
        self.postbout_para = []
        self.glm_models = []

    def glm_fits(self):
        az_model_posvel = make_LM(
            self.all_strikes_csv['Para Az'][
                self.all_strikes_csv['Para Az Velocity'] > 0],
            self.all_strikes_csv['Postbout Para Az'][
                self.all_strikes_csv['Para Az Velocity'] > 0])
        az_model_negvel = make_LM(
            self.all_strikes_csv['Para Az'][
                self.all_strikes_csv['Para Az Velocity'] < 0],
            self.all_strikes_csv['Postbout Para Az'][
                self.all_strikes_csv['Para Az Velocity'] < 0])
        alt_model_posvel = make_LM(
            self.all_strikes_csv['Para Alt'][
                self.all_strikes_csv['Para Alt Velocity'] > 0],
            self.all_strikes_csv['Postbout Para Alt'][
                self.all_strikes_csv['Para Alt Velocity'] > 0])
        alt_model_negvel = make_LM(
            self.all_strikes_csv['Para Alt'][
                self.all_strikes_csv['Para Alt Velocity'] < 0],
            self.all_strikes_csv['Postbout Para Alt'][
                self.all_strikes_csv['Para Alt Velocity'] < 0])
        dist_model_posvel = make_LM(
            self.all_strikes_csv['Para Dist'][
                self.all_strikes_csv['Para Dist Velocity'] > 0],
            self.all_strikes_csv['Postbout Para Dist'][
                self.all_strikes_csv['Para Dist Velocity'] > 0])
        dist_model_negvel = make_LM(
            self.all_strikes_csv['Para Dist'][
                self.all_strikes_csv['Para Dist Velocity'] < 0],
            self.all_strikes_csv['Postbout Para Dist'][
                self.all_strikes_csv['Para Dist Velocity'] < 0])
        all_az = make_LM(self.all_strikes_csv['Para Az'],
                         self.all_strikes_csv['Postbout Para Az'])
        all_alt = make_LM(self.all_strikes_csv['Para Alt'],
                          self.all_strikes_csv['Postbout Para Alt'])
        all_dist = make_LM(self.all_strikes_csv['Para Dist'],
                          self.all_strikes_csv['Postbout Para Dist'])
        self.glm_models = [az_model_posvel, az_model_negvel, alt_model_posvel,
                           alt_model_negvel, dist_model_posvel,
                           dist_model_negvel, all_az, all_alt, all_dist]

    def firstbout_transformation(self, para_varbs):
        pb_az = .29 * para_varbs["Para Az"]
        pb_dist = .87 * para_varbs["Para Dist"] - 27.5
        pb_alt = .48 * para_varbs["Para Alt"] + .15
        return pb_az, pb_dist, pb_alt

    def result_compiler(self):

        max_bouts = 100

        def nanremove(arr):
            a = []
            for b in arr:
                if not np.isfinite(b):
                    b = max_bouts
                a.append(b)
            return np.array(a)

        fig, ax = pl.subplots(1, 1, figsize=(15, 15))
        all_results_w_nan = [self.bouts_static]
        for bayesbouts in self.bouts_bayes:
            all_results_w_nan.append(bayesbouts)
        all_results = [nanremove(b) for b in all_results_w_nan]
        wilcoxon_results = [wilcoxon(all_results[0], all_results[i+1])
                            for i in range(len(self.bouts_bayes))]
        sum_of_ranks = [np.sum(
            np.sign(
                np.array(
                    all_results[0]) - np.array(
                        all_results[i+1]))) for i in range(
                            len(self.bouts_bayes))]
        self.test_results = [wilcoxon_results, sum_of_ranks]
        self.averages = [np.percentile(f, [25, 50, 75]) for f in all_results]
        print('One Sample T on Medians')
        one_samp_t = ttest_1samp([av[1] for av in self.averages[1:]],
                                 self.averages[0][1])
        print one_samp_t
        mannwit = []
        for i in range(3):
            try:
                mwt = mannwhitneyu([av[i] for av in self.averages[1:]],
                                   [self.averages[0][i]])[1]
            except ValueError:
                mwt = np.inf
            mannwit.append(mwt)
        print mannwit
        sb.violinplot(data=all_results_w_nan, ax=ax)
        pl.show()

    def marr2algorithm(self, para_varbs, boutcounter):
        if boutcounter > 50:
            return np.nan
        if strike(para_varbs, self.strike_means, self.strike_std, self.strike_ci):
            return boutcounter + 1
        if boutcounter == 0:
            pb_az, pb_dist, pb_alt = self.firstbout_transformation(
                para_varbs)
        else:
            pb_az = self.static_params['az_m'] * para_varbs[
                "Para Az"] + self.static_params['az_b']
            pb_dist = self.static_params['dist_m'] * para_varbs[
                "Para Dist"] + self.static_params['dist_b']
            if para_varbs["Para Alt"] >= 0:
                pb_alt = self.static_params['alt_m_pos'] * para_varbs[
                    "Para Alt"] + self.static_params['alt_b_pos']
            else:
                pb_alt = self.static_params['alt_m_neg'] * para_varbs[
                    "Para Alt"] + self.static_params['alt_b_neg']

        para_varbs = {"Para Az": pb_az,
                      "Para Alt": pb_alt,
                      "Para Dist": pb_dist}
        return self.marr2algorithm(para_varbs, boutcounter+1)

    def marr2bayes(self, para_varbs, boutcounter):
        if boutcounter > 50:
            return np.nan
        if strike(para_varbs, self.strike_means, self.strike_std, self.strike_ci):
            return boutcounter + 1
        if boutcounter == 0:
            pb_az, pb_dist, pb_alt = self.firstbout_transformation(
                para_varbs)
        else:
            if not self.use_likelihood:
                df_sim = query(
                    self.marr_bdb_file,
                    '''
                        SIMULATE
                            "Postbout Para Az",
                            "Postbout Para Alt",
                            "Postbout Para Dist"
                        FROM bout_population
                        USING MODEL {model_index}
                        GIVEN
                            "Para Az" = {Para Az},
                            "Para Alt" = {Para Alt},
                            "Para Dist" = {Para Dist}
                       LIMIT 1
                   '''

                    .format(
                        model_index=np.random.randint(
                            0, self.number_of_models_in_ensemble),
                        **para_varbs
                    ))
            else:
                df_sim = query(
                    self.marr_bdb_file,
                    '''
                        SIMULATE
                           "Postbout Para Az",
                           "Postbout Para Alt",
                           "Postbout Para Dist"
                           FROM bout_population
                        GIVEN
                           "Para Az" = {Para Az},
                           "Para Alt" = {Para Alt},
                           "Para Dist" = {Para Dist}
                        LIMIT 1
                    '''

                    .format(**para_varbs)
                    )
        if boutcounter == 0:
            para_varbs = {"Para Az": pb_az,
                          "Para Alt": pb_alt,
                          "Para Dist": pb_dist}
        else:
            self.prebout_para.append(para_varbs)
            para_varbs = {"Para Az": df_sim["Postbout Para Az"][0],
                          "Para Alt": df_sim["Postbout Para Alt"][0],
                          "Para Dist": df_sim["Postbout Para Dist"][0]}
            self.postbout_para.append(para_varbs)
        return self.marr2bayes(para_varbs, boutcounter+1)

    def run_marr2_models(self, model_repetitions, *static_or_bayes):
        if static_or_bayes != ():
            static_or_bayes = static_or_bayes[0]
        if static_or_bayes == 'static' or static_or_bayes == ():
            self.bouts_static = map(lambda para_position:
                                    self.marr2algorithm(para_position, 0),
                                    self.para_positions)
        if static_or_bayes == 'bayes' or static_or_bayes == ():
            print('in bayes')
            for i in range(model_repetitions):
                self.bouts_bayes.append(map(lambda para_position:
                                        self.marr2bayes(para_position, 0),
                                        self.para_positions))

    def validate_bayes_model(self, update_static_model):
        slopes, yints = validate_model_transformations(self.prebout_para,
                                                       self.postbout_para)
        if update_static_model:
            self.static_params = {'az_m': slopes[0],
                                  'alt_m_pos': slopes[1],
                                  'alt_m_neg': slopes[2],
                                  'dist_m': slopes[3],
                                  'az_b': yints[0],
                                  'alt_b_pos': yints[1],
                                  'alt_b_neg': yints[2],
                                  'dist_b': yints[3]}
            self.bouts_static = map(lambda para_position:
                                    self.marr2algorithm(para_position, 0),
                                    self.para_positions)
            self.result_compiler()
        
        
class PreyCap_Simulation:
    def __init__(self, fishmodel, paramodel, simlen, simulate_para, *para_input):
        self.fishmodel = fishmodel
        self.paramodel = paramodel
        self.velocity_kernel = np.load('hb_velocity_kernel.npy')
        self.yaw_kernel = np.load('hb_yaw_kernel.npy')[0:-1]
#        self.velocity_kernel = gaussian_filter(self.velocity_kernel, 1)
        self.sim_length = simlen
        self.para_states = []
        self.model_para_xyz = []
        self.simulate_para = simulate_para
        # + 1 is b/c init condition is already in fish_xyz, pitch, yaw
        self.realframes_start = fishmodel.rfo.hunt_frames[
            fishmodel.hunt_ind][0] - fishmodel.rfo.firstbout_para_intwin
        self.realframes_end = fishmodel.rfo.hunt_frames[
            fishmodel.hunt_ind][1]
        if para_input == ():
            self.para_xyz = [fishmodel.real_hunt["Para XYZ"][0],
                             fishmodel.real_hunt["Para XYZ"][1],
                             fishmodel.real_hunt["Para XYZ"][2]]
            self.create_para_trajectory()
        else:
            self.para_xyz = para_input[0]
            self.create_para_trajectory()
            # want to generate new interbouts only once per simhunt
        self.fish_xyz = [fishmodel.real_hunt["Initial Conditions"][0]]
        self.fish_bases = []
        self.fish_pitch = [fishmodel.real_hunt["Initial Conditions"][1]]
        self.fish_yaw = [fishmodel.real_hunt["Initial Conditions"][2]]
        self.db_rows = []
        self.prebout_para = []
        self.postbout_para = []
        self.interbouts = self.fishmodel.interbouts
        self.bout_durations = self.fishmodel.bout_durations
        self.bout_energy = []
        self.bout_counter = 0
        self.interpolate = 'kernel'
        self.strikelist = []
        self.framecounter = 0
        self.hunt_result = 0
        self.last_pvarbs = []
        self.num_frames_out_of_view = 0
        self.pcoords = []
        self.para_initial_position = {}
        
    def write_bouts_to_csv(self, rowlist):
        header = ["Bout Az", "Bout Alt", "Bout Dist", "Bout Delta Pitch", "Bout Delta Yaw",
                  "Para Az", "Para Alt", "Para Dist", "Para Az Velocity", "Para Alt Velocity",
                  "Para Dist Velocity", "Postbout Para Az", "Postbout Para Alt", "Postbout Para Dist"]
        directory = os.getcwd() + "/Model_Runs/"
        date = str(
            dt.datetime.now().year) + str(
                dt.datetime.now().month) + str(dt.datetime.now().day)
        model_param = self.fishmodel.model_param
        fish_name = self.fishmodel.rfo.directory[-8:]
        file_name = reduce(
            lambda x1, x2: x1 + x2,
            [filestring.replace(' ', '') for filestring in [
                str(mp[0]) + str(mp[1]) for mp in model_param.items()]])
        # a is field for append
        file_id = file_name + date + fish_name + '.csv'
        if file_id not in os.listdir(directory):
            with open(directory + file_id, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in rowlist:
                    writer.writerow(row)
        else:
            with open(directory + file_id, 'a') as f:
                writer = csv.writer(f)
                for row in rowlist:
                    writer.writerow(row)

    def score_hunt(self):
        score_dict = {"Number of Bouts": self.bout_counter,
                      "Hunt Duration": self.framecounter * .016,
                      "Result": self.hunt_result,
                      "Para Position at Term": self.last_pvarbs,
                      "Energy Expended": np.mean(self.bout_energy),
                      "Frames Out Of View": self.num_frames_out_of_view}
        return score_dict

        
    def create_para_trajectory(self):

        def make_accel_vectors():
            spacing = int(np.load('spacing.npy'))
            vx = [p[-1] - p[0] for p in partition(spacing, self.para_xyz[0])]
            vy = [p[-1] - p[0] for p in partition(spacing, self.para_xyz[1])]
            vz = [p[-1] - p[0] for p in partition(spacing, self.para_xyz[2])]
            acc = np.diff([vx, vy, vz])
            accel_vectors = zip(acc[0], acc[1], acc[2])
            return accel_vectors
        
                        
        if self.simulate_para:
            p_start_position = np.array([
                self.para_xyz[0][0], self.para_xyz[1][0], self.para_xyz[2][0]])
            p_position_3 = np.array([
                self.para_xyz[0][3], self.para_xyz[1][3], self.para_xyz[2][3]])
            start_vector = p_position_3 - p_start_position
            para_initial_conditions = [p_start_position, start_vector]
            if self.para_states == []:
                px, py, pz, states, vmax = self.paramodel.generate_model_para(
                    para_initial_conditions[1],
                    para_initial_conditions[0], self.sim_length, -1, False)
            # resets para_xyz for generative model
            else:
                px, py, pz, states, vmax = self.paramodel.generate_model_para(
                    para_initial_conditions[1],
                    para_initial_conditions[0], self.sim_length, -1, False, self.para_states)
            self.para_xyz = [np.array(px), np.array(py), np.array(pz)]
            self.para_states = states

        else:
            accelerations = make_accel_vectors()
            self.para_states = self.paramodel.model.predict(accelerations)
            
    def run_simulation(self):

        # for real_coords model, just have to map the para, get the para_varbs back,
        # run the strike, update bouts at the correct times (i.e. framecounter + realframes_start)
        # from there, strike function knows the refractory period.

        # do this by simply adding the framecounter+realframes_start fish_xyz, pitch, and yaw to the sims versions 

        def project_para(p_coords, nback, proj_forward, curr_frame):
            def final_pos(pc):
                fp = (
                    ((pc[curr_frame] - pc[curr_frame-nback])
                     / float(nback)) * proj_forward) + pc[curr_frame]
                return fp
            p_xyz = [final_pos(p) for p in p_coords]                
            return p_xyz

        csv_row_list = []
        last_finite_pvarbs = {}
        hunt_result = 0
        framecounter = 0
        future_frame = 5
        # spacing is an integer describing the sampling of vectors by the para. i.e.       # draws a new accel vector every 'spacing' frames
        spacing = np.load('spacing.npy')
        # fish_basis is origin, x, y, z unit vecs
        if self.simulate_para:
            px = np.interp(
                np.linspace(0,
                            self.para_xyz[0].shape[0],
                            self.para_xyz[0].shape[0] * spacing),
                range(self.para_xyz[0].shape[0]),
                self.para_xyz[0])
            py = np.interp(
                np.linspace(0,
                            self.para_xyz[1].shape[0],
                            self.para_xyz[1].shape[0] * spacing),
                range(self.para_xyz[1].shape[0]),
                self.para_xyz[1])
            pz = np.interp(
                np.linspace(0,
                            self.para_xyz[2].shape[0],
                            self.para_xyz[2].shape[0] * spacing),
                range(self.para_xyz[2].shape[0]),
                self.para_xyz[2])
        else:
            px, py, pz = self.para_xyz

        first_postbout_frame = False
        fish_bout = np.array([])
        para_varbs = []
        while True:
            if first_postbout_frame:

                csv_row = fish_bout[0:5].tolist() + [para_varbs["Para Az"],
                                                     para_varbs["Para Alt"],
                                                     para_varbs["Para Dist"],
                                                     para_varbs["Para Az Velocity"], 
                                                     para_varbs["Para Alt Velocity"],
                                                     para_varbs["Para Dist Velocity"]]
                csv_row_list.append(csv_row)
            fish_basis = fishxyz_to_unitvecs(self.fish_xyz[-1],
                                             self.fish_yaw[-1],
                                             self.fish_pitch[-1])
            # fish_basis is a 4D vec with origin, ufish, upar, and uperp
            self.fish_bases.append(fish_basis[1])
            if framecounter == len(px):
                #print("hunt epoch complete w/out strike")
                hunt_result = 2
                break
            
            if self.fishmodel.linear_predict_para or self.fishmodel.boutbound_prediction:
                if self.fishmodel.boutbound_prediction:
                    if framecounter >= future_frame:
                        future_frame = framecounter + self.bout_durations[
                            self.bout_counter]
                    if not self.fishmodel.linear_predict_para:
                        try:
                            self.model_para_xyz = [px[future_frame],
                                                   py[future_frame],
                                                   pz[future_frame]]
                        except IndexError:
                            self.model_para_xyz = [px[-1], py[-1], pz[-1]]
                else:
                    self.model_para_xyz = project_para(
                        [px, py, pz],
                        self.fishmodel.rfo.firstbout_para_intwin,
                        self.bout_durations[self.bout_counter], framecounter)

            else:
                self.model_para_xyz = [px[framecounter],
                                       py[framecounter], pz[framecounter]]

            self.fishmodel.current_target_xyz = self.model_para_xyz
            
            para_spherical = p_map_to_fish(fish_basis[1],
                                           fish_basis[0],
                                           fish_basis[3],
                                           fish_basis[2],
                                           self.model_para_xyz,
                                           0)

            if first_postbout_frame and self.bout_counter > 1:
                #Here add either a first bout list or just ignore first transformation
                self.prebout_para.append(para_varbs)
            
            para_varbs = {"Para Az": para_spherical[0],
                          "Para Alt": para_spherical[1],
                          "Para Dist": para_spherical[2],
                          "Para Az Velocity": 0,
                          "Para Alt Velocity": 0,
                          "Para Dist Velocity": 0}

            if framecounter <= self.fishmodel.rfo.firstbout_para_intwin:
                if not math.isnan(para_varbs["Para Az"]):
                    self.para_initial_position = para_varbs

            if first_postbout_frame:
                # write row function
                if self.bout_counter > 1:
                    self.postbout_para.append(para_varbs)
                first_postbout_frame = False


# can get more fancy with this function -- could have an entire object
# that counts interesting facets of the hunt. e.g. how many frames
# was the az greater than the init az after the first bout?
# alt? dist? you know that successful hunts have linear drops. 
            if np.abs(para_varbs["Para Az"]) > np.pi / 2:
                self.num_frames_out_of_view += 1
            
            if framecounter % 20 == 0:
                pass
#                print framecounter
            if framecounter >= 2000:
 #               print("EVASIVE PARA!!!!!")
                hunt_result = 2
                self.para_xyz[0] = px
                self.para_xyz[1] = py
                self.para_xyz[2] = pz
                break
            
            if strike(para_varbs,
                      self.fishmodel.strike_means,
                      self.fishmodel.strike_std,
                      self.fishmodel.strike_ci):
  #              print("STRIKE!!!!!!!!")
                self.para_xyz[0] = px
                self.para_xyz[1] = py
                self.para_xyz[2] = pz
                hunt_result = 1
                self.bout_counter += 1
                # note that para may be "eaten" by model faster than in real life (i.e. it enters the strike zone)
                # 
                break

            ''' you will store these vals in a
            list so you can determine velocities and accelerations '''
                
            if framecounter in self.interbouts and not self.fishmodel.modchoice == "Real Coords":

                self.fishmodel.current_fish_xyz = self.fish_xyz[-1]
                self.fishmodel.current_fish_pitch = self.fish_pitch[-1]
                self.fishmodel.current_fish_yaw = self.fish_yaw[-1]
#                print para_varbs
                pxyz_temp = [[px[i], py[i], pz[i]] for i in range(
                    framecounter-self.fishmodel.rfo.firstbout_para_intwin, framecounter)]
                pmap_returns = []
                for p_xyz in pxyz_temp:
                    pmap_returns.append(p_map_to_fish(fish_basis[1],
                                                      fish_basis[0],
                                                      fish_basis[3], fish_basis[2], p_xyz, 0))
                para_varbs["Para Az Velocity"] = np.mean(gaussian_filter(
                    np.diff([x[0] for x in pmap_returns]), 1)) / .016
                para_varbs["Para Alt Velocity"] = np.mean(gaussian_filter(
                    np.diff([x[1] for x in pmap_returns]), 1)) / .016
                para_varbs["Para Dist Velocity"] = np.mean(gaussian_filter(
                    np.diff([x[2] for x in pmap_returns]), 1)) / .016
                if not np.isfinite(para_varbs.values()).all():
                    # here para record ends before fish catches.
                    # para can't be nan at beginning due to real fish filters in master.py
                    hunt_result = 3
                    break
                else:
                    last_finite_pvarbs = copy.deepcopy(para_varbs)
                fish_bout = self.fishmodel.model(para_varbs)
                dx, dy, dz = sphericalbout_to_xyz(fish_bout[0],
                                                  fish_bout[1],
                                                  fish_bout[2],
                                                  fish_basis[1],
                                                  fish_basis[3],
                                                  fish_basis[2])
                if self.interpolate != 'none':
                    if self.interpolate == 'kernel':
                        vkernel = normalize_kernel(
                            self.velocity_kernel,
                            self.bout_durations[self.bout_counter])
                        ykernel = normalize_kernel_interp(
                            self.yaw_kernel,
                            self.bout_durations[self.bout_counter])
                        x_prog = (np.cumsum(dx * vkernel) + self.fish_xyz[-1][0]).tolist()
                        y_prog = (np.cumsum(dy * vkernel) + self.fish_xyz[-1][1]).tolist()
                        z_prog = (np.cumsum(dz * vkernel) + self.fish_xyz[-1][2]).tolist()
                        yaw_prog = (np.cumsum(fish_bout[4] * ykernel) + self.fish_yaw[-1]).tolist()
                        pitch_prog = (np.cumsum(fish_bout[3] * vkernel) + self.fish_pitch[-1]).tolist()
                        
                    elif self.interpolate == 'linear':
                        x_prog = np.linspace(
                            self.fish_xyz[-1][0],
                            self.fish_xyz[-1][0] + dx,
                            self.bout_durations[self.bout_counter]).tolist()
                        y_prog = np.linspace(
                            self.fish_xyz[-1][1],
                            self.fish_xyz[-1][1] + dy,
                            self.bout_durations[self.bout_counter]).tolist()
                        z_prog = np.linspace(
                            self.fish_xyz[-1][2],
                            self.fish_xyz[-1][2] + dz,
                            self.bout_durations[self.bout_counter]).tolist()
                        yaw_prog = np.linspace(
                            self.fish_yaw[-1],
                            self.fish_yaw[-1] + fish_bout[4],
                            self.bout_durations[self.bout_counter]).tolist()
                        pitch_prog = np.linspace(
                            self.fish_pitch[-1],
                            self.fish_pitch[-1] + fish_bout[3],
                            self.bout_durations[self.bout_counter]).tolist()
                    bout_xyz = zip(x_prog, y_prog, z_prog)
                    bout_pitch = np.clip(pitch_prog, -np.pi, np.pi).tolist()
                    # assures fish can't frontflip or backflip over. 
                    bout_yaw = np.mod(yaw_prog, 2*np.pi).tolist()
                    # assures yaw coords stay 0 to 2pi w no negatives or > 2pi                    
                    self.fish_xyz += bout_xyz
                    self.fish_pitch += bout_pitch
                    self.fish_yaw += bout_yaw
                    uf_bout = []
                    fb_init = [self.fish_bases[-1]]
                    for x, y, z, p, yw in zip(x_prog[1:],
                                              y_prog[1:],
                                              z_prog[1:], pitch_prog[1:], yaw_prog[1:]):                        
                        fish_basis = fishxyz_to_unitvecs((x,y,z), yw, p)
                        uf_bout.append(fish_basis[1])
                    self.fish_bases += uf_bout
                    uf_bout = fb_init + uf_bout
                    self.bout_energy.append(calculate_bout_energy(bout_xyz, uf_bout, bout_yaw, bout_pitch))
                    framecounter += self.bout_durations[self.bout_counter]
                                            
                elif self.interpolate == 'none':
                    new_xyz = self.fish_xyz[-1] + np.array([dx, dy, dz])
                    self.fish_xyz.append(new_xyz)
                    self.fish_pitch.append(np.clip(self.fish_pitch[-1] + fish_bout[3], -np.pi, np.pi))
                    self.fish_yaw.append(np.mod(self.fish_yaw[-1] + fish_bout[4], 2*np.pi))
                    framecounter += 1
                first_postbout_frame = True
                self.bout_counter += 1
                
            else:
                if (self.fishmodel.modchoice == 'Real Coords') and (
                        framecounter >= self.fishmodel.rfo.firstbout_para_intwin):
                    # add 1 b/c of pre-population in model. worked all this out on paper
                    frame_map = framecounter + self.realframes_start + 1
                    if framecounter in self.interbouts:
                        if not np.isfinite(para_varbs.values()).all():
                            hunt_result = 3
                            break
                        else:
                            last_finite_pvarbs = copy.deepcopy(para_varbs)
                        bout_xyz = self.fishmodel.rfo.fish_xyz[frame_map:frame_map + self.bout_durations[self.bout_counter]]
                        bout_pitch = self.fishmodel.rfo.pitch_all[frame_map:frame_map + self.bout_durations[self.bout_counter]]
                        bout_yaw = self.fishmodel.rfo.yaw_all[frame_map:frame_map + self.bout_durations[self.bout_counter]]
                        fb_init = [self.fish_bases[-1]]
                        uf_bout = []
                        for xyz, p, yw in zip(bout_xyz[1:], bout_pitch[1:], bout_yaw[1:]):                        
                            fish_basis = fishxyz_to_unitvecs(xyz, yw, p)
                            uf_bout.append(fish_basis[1])
                        self.fish_xyz += bout_xyz
                        self.fish_pitch += bout_pitch.tolist()
                        self.fish_yaw += bout_yaw.tolist()
                        self.fish_bases += uf_bout
                        uf_bout = fb_init + uf_bout
                        self.bout_energy.append(calculate_bout_energy(
                            bout_xyz,
                            uf_bout, bout_yaw, bout_pitch))
                        framecounter += self.bout_durations[self.bout_counter]
                        first_postbout_frame = True
                        self.bout_counter += 1
                    else:
                        self.fish_xyz.append(self.fishmodel.rfo.fish_xyz[frame_map])
                        self.fish_pitch.append(self.fishmodel.rfo.pitch_all[frame_map])
                        self.fish_yaw.append(self.fishmodel.rfo.yaw_all[frame_map])
                        framecounter += 1
                else:    
                    self.fish_xyz.append(self.fish_xyz[-1])
                    self.fish_pitch.append(self.fish_pitch[-1])
                    self.fish_yaw.append(self.fish_yaw[-1])
                    framecounter += 1

        self.last_pvarbs = last_finite_pvarbs
        self.pcoords = [px[0:framecounter],
                        py[0:framecounter], pz[0:framecounter]]
        # if framecounter <= 1, fish catches it immediately and theres no bout
        if hunt_result == 1 and framecounter > 1:
            self.strikelist = np.zeros(framecounter-1).tolist() + [1]
        else:
            self.strikelist = np.zeros(framecounter)
        self.framecounter = framecounter
        self.hunt_result = hunt_result
        self.db_rows = csv_row_list
#        self.write_bouts_to_csv(csv_row_list)
    
class FishModel:
    def __init__(self, model_param, strike_params, rfo, hunt_ind, *spherical_bouts):
        self.rfo = rfo
        self.regmod = []
        self.hunt_ind = hunt_ind
        self.strike_ci = .05
        self.model_param = model_param
        self.modchoice = model_param["Model Type"]
        if spherical_bouts != ():
            if spherical_bouts[0] == 'All':
                self.spherical_bouts = rfo.all_spherical_bouts
            elif spherical_bouts[0] == 'Hunt':
                self.spherical_bouts = rfo.all_spherical_huntbouts
        self.bdb_file = []
        if self.modchoice == "Real Bouts":
            self.model = (lambda pv: self.real_fish_bouts(pv))
        elif self.modchoice == "Real Coords":
            self.model = (lambda pv: self.real_fish_coords(pv))
        elif self.modchoice in ["Independent Regression",
                                "Multiple Regression Velocity",
                                "Multiple Regression Position"]:
            self.model = (lambda pv: self.regression_model(pv))
        elif self.modchoice == "Bayes":
            print('loading bdb file')
            self.bdb_file = bl.bayesdb_open('wik_bdb/bdb_hunts_strikes.bdb')
            self.model = (lambda pv: self.bdb_model(pv))
        elif self.modchoice == "Ideal":
            self.model = (lambda pv: self.ideal_model(pv))
        elif self.modchoice == "Random":
            self.model = (lambda pv: self.random_model(pv))
        self.linear_predict_para = False
        self.boutbound_prediction = False
        self.strike_means = strike_params[0]
        self.strike_std = strike_params[1]
        self.strike_refractory_period = strike_params[2]
        self.real_hunt = self.rfo.model_input(hunt_ind)
        if model_param["Real or Sim"] == "Real":
            self.interbouts = self.real_hunt["Interbouts"]
            self.bout_durations = self.real_hunt["Bout Durations"]
        elif model_param["Real or Sim"] == "Sim":
            self.interbouts = []
            self.bout_durations = []
            self.generate_simulated_interbouts(1000)
        self.current_fish_xyz = []
        self.current_fish_yaw = 0
        self.current_fish_pitch = 0
        self.current_target_xyz = []
        self.interbouts = np.cumsum(
            self.interbouts) + self.real_hunt["First Bout Delay"]
        # note that the fish is given 5 frames before initializing a bout.
        # this is so the model has a proper velocity to work with. occurs
        # at the exact same time as the fish bouts in reality. para
        # trajectory is contwin + intwin - 5 to start. 
        self.num_bouts_generated = 0


    def generate_random_interbouts(self, num_bouts):
        all_ibs = np.floor(np.random.random(num_bouts) * 100)
        bout_indices = np.cumsum(all_ibs)
        return bout_indices

    def generate_simulated_interbouts(self, num_bouts):
        filt_ibs = []
        filt_bdur = []
        pct10ib, pct10bdur = np.percentile([(b["Interbout"],
                                             b["Bout Duration"]) for b in self.spherical_bouts],
                                           10, axis=0)
        pct90ib, pct90bdur = np.percentile([(b["Interbout"],
                                             b["Bout Duration"]) for b in self.spherical_bouts],
                                           90, axis=0)
        for b in self.spherical_bouts:
            if pct10ib < b["Interbout"] < pct90ib:
                if pct10bdur < b["Bout Duration"] < pct90bdur:
                    filt_ibs.append(b["Interbout"])
                    filt_bdur.append(b["Bout Duration"])
        rand_intlist = np.random.randint(0, len(filt_ibs), num_bouts)
        for r in rand_intlist:
            self.interbouts.append(filt_ibs[r])
            self.bout_durations.append(filt_bdur[r])

    def real_fish_coords(self, para_varbs):
        bout = np.zeros(5)
        return bout
    
    def real_fish_bouts(self, para_varbs):
        hunt_df = self.real_hunt[
            "Hunt Dataframe"].loc[self.num_bouts_generated]
        bout = np.array(
            [hunt_df["Bout Az"],
             hunt_df["Bout Alt"],
             hunt_df["Bout Dist"],
             hunt_df["Bout Delta Pitch"],
             -1*hunt_df["Bout Delta Yaw"]])
        self.num_bouts_generated += 1
        return bout

    def random_model(self, para_varbs):
        r_int = np.random.randint(0, len(self.spherical_bouts))
        rand_bout = self.spherical_bouts[r_int]
        bout = np.array(
            [rand_bout["Bout Az"],
             rand_bout["Bout Alt"],
             rand_bout["Bout Dist"],
             rand_bout["Delta Pitch"],
             -1*rand_bout["Delta Yaw"]])
        return bout

    def ideal_model(self, para_varbs):

        def score_para(p_results):
            # first ask if any para are IN strike zone
            # strikes = np.array([self.strike(p) for p in p_results])
            # if strikes.any():
            #     strike_args = np.argwhere(strikes)
            #     rand_strike = np.random.randint(strike_args.shape[0])
            #     bout_choice = strike_args[rand_strike][0]
            #     return bout_choice
            az = np.array([p['Para Az'] for p in p_results]) - self.strike_means[0]
            alt = np.array([p['Para Alt'] for p in p_results]) - self.strike_means[1]
            dist = np.array([p['Para Dist'] for p in p_results]) - self.strike_means[2]
            norm_az = az / np.std(az)
            norm_alt = alt / np.std(alt)
            norm_dist = dist / np.std(dist)
            score = np.abs(norm_az) + np.abs(norm_alt) + np.abs(norm_dist)
            bout_choice = np.argmin(score)
            return bout_choice
 
        # Use this model for all greedy models and simply manipulate the model_para_xyz        
        fish_basis = fishxyz_to_unitvecs(self.current_fish_xyz,
                                         self.current_fish_yaw,
                                         self.current_fish_pitch)
        para_results = []
        for bt in self.spherical_bouts:
            dx, dy, dz = sphericalbout_to_xyz(bt["Bout Az"],
                                              bt["Bout Alt"],
                                              bt["Bout Dist"],
                                              fish_basis[1],
                                              fish_basis[3],
                                              fish_basis[2])
            temp_yaw = -1*bt["Delta Yaw"] + self.current_fish_yaw
            temp_pitch = bt["Delta Pitch"] + self.current_fish_pitch
            temp_xyz = np.array(self.current_fish_xyz) + np.array([dx, dy, dz])
            temp_fish_basis = fishxyz_to_unitvecs(temp_xyz, 
                                                  temp_yaw, 
                                                  temp_pitch)
            postbout_para_spherical = p_map_to_fish(temp_fish_basis[1],
                                                    temp_fish_basis[0],
                                                    temp_fish_basis[3],
                                                    temp_fish_basis[2],
                                                    self.current_target_xyz,
                                                    0)
            pvarbs = {"Para Az": postbout_para_spherical[0],
                      "Para Alt": postbout_para_spherical[1],
                      "Para Dist": postbout_para_spherical[2]}
            para_results.append(pvarbs)
        best_bout = self.spherical_bouts[score_para(para_results)]

        bout_array = np.array([best_bout['Bout Az'],
                               best_bout['Bout Alt'],
                               best_bout['Bout Dist'], best_bout['Delta Pitch'], -1*best_bout['Delta Yaw']])
        return bout_array

    def regression_model(self, para_varbs):
        if self.num_bouts_generated == 0:
            regmod = self.regmod[0]
        else:
            regmod = self.regmod[1]
        if self.modchoice == "Independent Regression":
            p_input = [para_varbs["Para Az"], para_varbs["Para Alt"], para_varbs["Para Dist"],
                       para_varbs["Para Alt"], para_varbs["Para Az"]]
            bout = [rm.predict([1, p_input[pi]]) for pi, rm in enumerate(regmod)]
        else:
            if self.modchoice == "Multiple Regression Velocity":
                p_input = [para_varbs["Para Az"],
                           para_varbs["Para Alt"],
                           para_varbs["Para Dist"],
                           para_varbs["Para Az Velocity"], 
                           para_varbs["Para Alt Velocity"],
                           para_varbs["Para Dist Velocity"]]
            elif self.modchoice == "Multiple Regression Position":
                p_input = [para_varbs["Para Az"],
                           para_varbs["Para Alt"],
                           para_varbs["Para Dist"]]
            # this is for the constant y-intercept 
            p_input = [1] + p_input
            bout = [rm.predict(p_input) for rm in regmod]
        # invert bout delta yaw
        bout[-1] *= -1
        self.num_bouts_generated += 1
        return np.array(bout)

    # def regression_model(self, para_varbs):
    #     bout_az = (para_varbs['Para Az'] * 1.36) + .02
    #     bout_yaw = -1 * ((.46 * para_varbs['Para Az']) - .02)
    #     bout_alt = (1.5 * para_varbs['Para Alt']) + -.37
    #     bout_pitch = (.27 * para_varbs['Para Alt']) - .04
    #     bout_dist = (.09 * para_varbs['Para Dist']) + 29
    #     bout_array = np.array([bout_az,
    #                            bout_alt,
    #                            bout_dist, bout_pitch, bout_yaw])
    #     noise_array = np.ones(5)
    #     # noise_array = np.array(
    #     # [(np.random.random() * .4) + .8 for i in bout_array])
    #     bout = bout_array * noise_array
    #     return bout

    def bdb_model(self, para_varbs):
#        invert = True
        invert = False
#        sampling = 'median'
        sampling = 'sample'
        
        def invert_pvarbs(pvarbs):
            pvcopy = copy.deepcopy(pvarbs)
            invert_az = False
            invert_alt = False
            if pvcopy["Para Az"] < 0:
                pvcopy["Para Az"] *= -1
                pvcopy["Para Az Velocity"] *= -1
                invert_az = True
            if pvcopy["Para Alt"] < 0:
                pvcopy["Para Alt"] *= -1
                pvcopy["Para Alt Velocity"] *= -1
                invert_alt = True
            return pvcopy, invert_az, invert_alt

        def invert_bout(bout_dic, invert_az, invert_alt):
            bdic_copy = copy.deepcopy(bout_dic)
            if invert_az:
                bdic_copy["Bout Az"] *= -1
                bdic_copy["Bout Delta Yaw"] *= -1
            if invert_alt:
                bdic_copy["Bout Alt"] *= -1
                bdic_copy["Bout Delta Pitch"] *= -1
            return bdic_copy

        if invert:
            para_varbs, inv_az, inv_alt = invert_pvarbs(para_varbs)

        if sampling == 'sample':
            para_varbs.update({'row_limit': 1})
        else:
            para_varbs.update({'row_limit': 50})
            
        print para_varbs
        df_sim = query(self.bdb_file,
                       '''SIMULATE "Bout Az", "Bout Alt",
                       "Bout Dist", "Bout Delta Pitch",
                       "Bout Delta Yaw"
                       FROM bout_population
                       GIVEN "Para Az" = {Para Az},
                       "Para Alt" = {Para Alt},
                       "Para Dist" = {Para Dist},
                       "Para Az Velocity" = {Para Az Velocity},
                       "Para Alt Velocity" = {Para Alt Velocity},
                       "Para Dist velocity" = {Para Dist Velocity}
                       LIMIT {row_limit} '''.format(**para_varbs))
        # can add a USING MODEL flag if you want a
        # specific model before the LIMIT

        if sampling == 'median':
            bout_az = np.nanmedian(df_sim['Bout Az'])
            bout_alt = np.nanmedian(df_sim['Bout Alt'])
            bout_dist = np.nanmedian(df_sim['Bout Dist'])
            bout_pitch = np.nanmedian(df_sim['Bout Delta Pitch'])
            bout_yaw = -1*np.nanmedian(df_sim['Bout Delta Yaw'])

        elif sampling == 'mean':
            bout_az = np.nanmean(df_sim['Bout Az'])
            bout_alt = np.nanmean(df_sim['Bout Alt'])
            bout_dist = np.nanmean(df_sim['Bout Dist'])
            bout_pitch = np.nanmean(df_sim['Bout Delta Pitch'])
            bout_yaw = -1*np.nanmean(df_sim['Bout Delta Yaw'])

        elif sampling == 'sample':
            bout_az = df_sim['Bout Az'][0]
            bout_alt = df_sim['Bout Alt'][0]
            bout_dist = df_sim['Bout Dist'][0]
            bout_pitch = df_sim['Bout Delta Pitch'][0]
            bout_yaw = -1*df_sim['Bout Delta Yaw'][0]

        b_dict = {"Bout Az": bout_az,
                  "Bout Alt": bout_alt,
                  "Bout Dist": bout_dist,
                  "Bout Delta Yaw": bout_yaw,
                  "Bout Delta Pitch": bout_pitch}

        del df_sim
        gc.collect()

        if invert:
            b_dict = invert_bout(b_dict, inv_az, inv_alt)
        bout = np.array([b_dict["Bout Az"],
                         b_dict["Bout Alt"],
                         b_dict["Bout Dist"],
                         b_dict["Bout Delta Pitch"],
                         b_dict["Bout Delta Yaw"]])
        if not np.isfinite(bout).all():
            print('Found nan in bout')
            return self.bdb_model(para_varbs)

        print bout
        return bout


def characterize_strikes(hb_data):
    strike_characteristics = []
    for i in range(len(hb_data["Bout Number"])):
        if hb_data["Bout Number"][i] == -1:
            if hb_data["Strike Or Abort"][i] == 1:
                strike = np.array([hb_data["Para Az"][i],
                                   hb_data["Para Alt"][i],
                                   hb_data["Para Dist"][i]])
                if np.isfinite(strike).all():
                    strike_characteristics.append(strike)
    # mu_az, std_az = norm.fit(strike_characteristics[0])
    # mu_alt, std_alt = norm.fit(strike_characteristics[1])
    # mu_d, std_d = norm.fit(strike_characteristics[2])
    # avg_strike_position = np.array([mu_az, mu_alt, mu_d])
    # std = np.array([std_az, std_alt, std_d])
    avg_strike_position = np.mean(strike_characteristics, axis=0)
    std = np.std(strike_characteristics, axis=0)
    return [avg_strike_position, std]


def make_LM(y, x):
    # if type(x) == 'list':
    # else:
    x = sm.add_constant(x)
#    mod = sm.RLM(y, x, M=sm.robust.norms.HuberT())
    mod = sm.GLM(y, x)
    fit_mod = mod.fit()
    return fit_mod


def make_independent_regression_model(hunt_db):
    print('Building Independent Reg Model')
    fit_models = [make_LM(hunt_db["Bout Az"], hunt_db["Para Az"]),
                  make_LM(hunt_db["Bout Alt"], hunt_db["Para Alt"]),
                  make_LM(hunt_db["Bout Dist"], hunt_db["Para Dist"]),
                  make_LM(hunt_db["Bout Delta Pitch"], hunt_db["Para Alt"]),
                  make_LM(hunt_db["Bout Delta Yaw"], hunt_db["Para Az"])]
#    reg_lambdas = [lambda pfeature: fm.predict(pfeature) for fm in fit_models]
    return fit_models


def make_multiple_regression_model(hunt_db, vel_or_position):
    print('Building Multiple Reg Model')
    if vel_or_position == 'velocity':
        para_features = ["Para Az",
                         "Para Alt",
                         "Para Dist", 
                         "Para Az Velocity", "Para Alt Velocity", "Para Dist Velocity"]
    elif vel_or_position == 'position':
        para_features = ["Para Az",
                         "Para Alt", "Para Dist"]
    bout_features = ["Bout Az", "Bout Alt", "Bout Dist", "Bout Delta Pitch", "Bout Delta Yaw"]
    # mult_reg_mods = map(lambda bf: sm.RLM(
    #     hunt_db[bf],
    #     sm.add_constant(hunt_db[para_features]), M=sm.robust.norms.HuberT()), bout_features)
    mult_reg_mods = map(lambda bf: sm.GLM(
        hunt_db[bf],
        sm.add_constant(hunt_db[para_features])), bout_features)

    fit_mult_reg_mods = [m.fit() for m in mult_reg_mods]
    return fit_mult_reg_mods


def realhunt_allframes(rfo, hunt_id):
    delay = rfo.firstbout_para_intwin
    h_ind = np.argwhere(np.array(rfo.hunt_ids) == hunt_id)[0][0]
    frames = np.array(rfo.hunt_frames[h_ind]) - delay
    mod_input = rfo.model_input(h_ind)
    para_xyz = [mod_input["Para XYZ"][0],
                mod_input["Para XYZ"][1],
                mod_input["Para XYZ"][2]]
    np.save('ufish.npy', rfo.fish_orientation[frames[0]:frames[1]])
    np.save('ufish_origin.npy', rfo.fish_xyz[frames[0]:frames[1]])
    np.save('3D_paracoords.npy', para_xyz)

    
def make_vr_movies(sim1, sim2):
    hunt_id = sim1.fishmodel.rfo.hunt_ids[sim1.fishmodel.hunt_ind]
    realhunt_allframes(sim1.fishmodel.rfo, hunt_id)
    np.save('para_simulation1.npy', sim1.pcoords)
    np.save('origin_model1.npy', sim1.fish_xyz[:-1])
    np.save('uf_model1.npy', sim1.fish_bases[:-1])                
    np.save('strikelist1.npy', sim1.strikelist)
    np.save('para_simulation2.npy', sim2.pcoords)
    np.save('origin_model2.npy', sim2.fish_xyz[:-1])
    np.save('uf_model2.npy', sim2.fish_bases[:-1])                
    np.save('strikelist2.npy', sim2.strikelist)

    
def calculate_bout_energy(bout_xyz,
                          uf_vector, bout_yaw, bout_pitch):
    bout_duration = len(bout_xyz)
    head_to_com_distance = 50
    com_xyz_init = np.array(bout_xyz[0]) - (head_to_com_distance * uf_vector[0])
    com_xyz_end = np.array(bout_xyz[-1]) - (head_to_com_distance * uf_vector[-1])
    velocity_com = np.linalg.norm(com_xyz_end - com_xyz_init) / (bout_duration * .016)
    angular_yaw_vel = yaw_fix([bout_yaw[-1] - bout_yaw[0]])[0] / (bout_duration * .016)
    angular_pitch_vel = bout_pitch[-1] - bout_pitch[0] / (bout_duration * .016)
    # 1 mg from avella et al. 2012, dpf10. wet mass. dry mass is ~.2mg
    fish_mass = .001
    moment_of_inertia = fish_mass * (head_to_com_distance)**2
    total_bout_energy = .5*((velocity_com**2 * fish_mass) + (
            angular_yaw_vel**2 * moment_of_inertia) + (
                angular_pitch_vel**2 * moment_of_inertia))
    return total_bout_energy


def calculate_bout_energy_nonlin(bout_xyz,
                                 uf_vector, bout_yaw, bout_pitch):
    bout_xyz = np.array(bout_xyz)
    head_to_com_distance = 50
    com_xyz = [xyz - (head_to_com_distance * uf) for xyz, uf in zip(bout_xyz, uf_vector)]
    velocity_com = np.array([np.linalg.norm(velvec) for velvec in np.diff(com_xyz, axis=0)])
    angular_yaw_vel = np.array(yaw_fix(np.diff(bout_yaw)))
    angular_pitch_vel = np.diff(bout_pitch)
    # 1 mg from avella et al. 2012, dpf10. wet mass. dry mass is ~.2mg
    fish_mass = .001
    moment_of_inertia = fish_mass * (head_to_com_distance)**2
    total_bout_energy = np.sum(.5*(
        (velocity_com**2 * fish_mass) + (
            angular_yaw_vel**2 * moment_of_inertia) + (
                angular_pitch_vel**2 * moment_of_inertia)))
    return total_bout_energy


# first half vs second half, quartiles (split into 4)
# am currently stopping plotting when one model eats. maybe don't do this.
# if you plot both entirely, should end up in same place
# also when scoring, note whether success or fail (i.e. result of hunt 1, 2, 3)
def score_model_similarity(mod1_index, mod2_index,
                           simlist_by_huntid, num_partitions):

    def avg_of_partitions(arr):

        # this isn't quite right. its not iterating through the partitions,
        # it stores a partition for each sub_array and then the whole partition comes out
        # and is taken the mean of. want to take the mean of each piece. 
        means_by_partition = []        
        arr_partitioned = [partition(len(xs) / num_partitions, xs) for xs in arr]
        extracted_partitions = [[p for p in part] for part in arr_partitioned]
        for n in range(num_partitions):
            partition_means = [np.nanmean(
                prt[n]) for prt in extracted_partitions if prt != []]
            means_by_partition.append(partition_means)
        return means_by_partition
    
    scores = []
    prebout_para1 = []
    prebout_para2 = []
    postbout_para1 = []
    postbout_para2 = []
    for sims in simlist_by_huntid:
        sim1 = sims[mod1_index]
        sim2 = sims[mod2_index]
        # these are now ordered pairs of pre and post para positions
        prebout_para1 += sim1.prebout_para
        postbout_para1 += sim1.postbout_para
        prebout_para2 += sim2.prebout_para
        postbout_para2 += sim2.postbout_para
        scores.append(score_trajectory_similarity(
            False, sim1, sim2))
    scores = np.array(scores)
    xyz_sim = avg_of_partitions(scores[:, 0])
    pitch_scores = avg_of_partitions(scores[:, 1])
    yaw_scores = avg_of_partitions(scores[:, 2])
    fig, ax = pl.subplots(2, 1, figsize=(15, 15))
    ax[0].set_title('XYZ Similarity')
    ax[1].set_title('Pitch and Yaw Similarity')
    sb.violinplot(data=xyz_sim, ax=ax[0])
    sb.violinplot(data=pitch_scores + yaw_scores, ax=ax[1])
    pl.show()
    print('XYZ Quartiles')
    print([np.percentile(xs, [25, 50, 75]).tolist() for xs in xyz_sim])
    print('Pitch Quartiles')
    print([np.percentile(ps, [25, 50, 75]).tolist() for ps in pitch_scores])
    print('Yaw Quartiles')
    print([np.percentile(ys, [25, 50, 75]).tolist() for ys in yaw_scores])
    return [fig, xyz_sim, pitch_scores, yaw_scores,
            [prebout_para1, postbout_para1], [prebout_para2, postbout_para2]]


def extract_all_spherical_bouts(drct_list):
    all_spherical_bouts = []
    all_spherical_huntbouts = []
    for fish_id in drct_list:
        rfo = pd.read_pickle(
            os.getcwd() + '/' + fish_id + '/RealHuntData_' + fish_id + '.pkl')
        all_spherical_bouts += rfo.all_spherical_bouts
        all_spherical_huntbouts += rfo.all_spherical_huntbouts
    return all_spherical_bouts, all_spherical_huntbouts


def model_wrapper(drct_list, strike_params, para_model,
                  model_params, hunt_db, actions, *combine_spherical):
    para_initial_positions = []
    sim_list_allfish = []
    if combine_spherical != ():
        sb_allfish, shb_allfish = extract_all_spherical_bouts(drct_list)
    for dc_ind, fish_id in enumerate(drct_list):
        rfo = pd.read_pickle(
            os.getcwd() + '/' + fish_id + '/RealHuntData_' + fish_id + '.pkl')
        if combine_spherical != ():
            rfo.all_spherical_bouts = sb_allfish
            rfo.all_spherical_huntbouts = shb_allfish
        simlist, ppos_list = execute_models(rfo, strike_params,
                                            para_model, model_params, hb, actions)
        sim_list_allfish += simlist
        if dc_ind == 0:
            ms = score_summary(simlist, model_params, 0)
        else:
            ms_2 = score_summary(simlist, model_params, 0)
            ms = [a + b for a, b in zip(ms, ms_2)]
        para_initial_positions += ppos_list
    sim_list_by_model = [sim_list_allfish[i::len(model_params)]
                         for i in range(len(model_params))]
    sim_list_by_hunt = [p for p in partition(len(model_params),
                                             sim_list_allfish)]
    return ms, para_initial_positions, sim_list_by_model, sim_list_by_hunt


def execute_models(rfo, strike_params, para_model,
                   model_params, hunt_db, actions, *hunt_id):
    print("Executing Models")
    sim_list = []
    para_init_positions = []
    prev_interbouts = []
    sequence_length = 10000
    p_xyz = []
    pause_wrapper = False
    # this will iterate through hunt_ids and through types of Fishmodel.
    # function takes a real_fish_object and creates a Simulation for each hunt id and each model.
    # have to have the Sim output a -1 or something if the fish never catches the para
    if hunt_id == ():
        hunt_ids = rfo.hunt_ids
    else:
        hunt_ids = hunt_id
#        pause_wrapper = True
    for h_ind, (hid, action) in enumerate(zip(hunt_ids, rfo.hunt_results)):
        if hunt_id != ():
            h_ind = np.argwhere(np.array(rfo.hunt_ids) == hunt_id[0])[0][0]
            action = rfo.hunt_results[h_ind]
        sim_created = False
        # Make a variable that copies prevoius interbouts and durations so that each sim has the same ibs and durations
        if action not in actions:
            continue
        for mi, model_run in enumerate(model_params):
#            print("Running " + model_run["Model Type"] + " on hunt " + str(hid))
            try:
                fish = FishModel(model_run, strike_params, rfo, h_ind, model_run["Spherical Bouts"])
            except KeyError:
                fish = FishModel(model_run, strike_params, rfo, h_ind)
            try:
                if model_run["Extrapolate Para"]:
                    fish.linear_predict_para = True
            except KeyError:
                pass
            try: 
                if model_run["Willie Mays"]:
                    fish.boutbound_prediction = True
            except KeyError:
                pass

            if model_run["Model Type"] == "Independent Regression":
                fish.regmod = [ir_bout0, independent_regression_model]
            elif model_run["Model Type"] == "Multiple Regression Velocity":
                fish.regmod = [mv_bout0, multiple_regression_model_velocity]
            elif model_run["Model Type"] == "Multiple Regression Position":
                fish.regmod = [mp_bout0, multiple_regression_model_position]
            else:
                fish.regmod = []
            try: 
                fish.strike_ci = model_run["Strike CI"]
            except KeyError:
                pass
            if model_run["Real or Sim"] == "Real":
                sim = PreyCap_Simulation(
                    fish,
                    para_model,
                    sequence_length,
                    False)
            # else accounts for Sim entry or entry of para states    
            else:
                # this allows same para sim to be run on all models for each hunt
                # if you want to run multiple sims per hunt, just submit the same dictionary
                # list with no real hunts in it.           
                if not sim_created:
                    sim = PreyCap_Simulation(
                        fish,
                        para_model,
                        sequence_length,
                        True)
                    if type(model_run["Real or Sim"]) == list:
                        sim.para_states = model_run["Real or Sim"]
                    sim_created = True 
                    
                else:
                    fish.interbouts = prev_interbouts
                    fish.bout_durations = prev_boutdurations
                    sim = PreyCap_Simulation(
                        fish,
                        para_model,
                        sequence_length,
                        False,
                        p_xyz)

            sim.run_simulation()
            p_xyz = copy.deepcopy(sim.para_xyz)
            prev_interbouts = copy.deepcopy(sim.fishmodel.interbouts)
            prev_boutdurations = copy.deepcopy(sim.fishmodel.bout_durations)
            if mi == 0:
                para_init_positions.append(copy.deepcopy(sim.para_initial_position))
            sim_list.append(sim)
            if model_run["Model Type"] == "Bayes":
                fish.bdb_file.close()
            if pause_wrapper:            
                r = raw_input('Press Enter to Continue')
    return sim_list, para_init_positions


def yaw_fix(dy_input):
    dyaw = []
    for dy in dy_input:
        if np.abs(dy) < np.pi:
            dyaw.append(dy)
        else:
            if dy < 0:
                dyaw.append(-2*np.pi + np.abs(dy))
            else:
                dyaw.append(2*np.pi - dy)
    return np.array(dyaw)


def score_summary(simlist, modlist, by_hunt_or_model):
    num_models = len(modlist)
    hunt_scores = [[] for i in range(num_models)]
    for ind, sim in enumerate(simlist):
        score = sim.score_hunt()
        hunt_scores[ind % num_models].append(score)
    return hunt_scores


def plot_query(model_summary, para_positions,
               strike_params, plot_type, *filter_result):

    def result_filter(modsum, result):
        results_lists = score_query("Result", modsum)
        args_matching_result = [np.where(
            np.array(rs) == result)[0] for rs in results_lists]
        return args_matching_result

    res_args = []
    if filter_result != ():
        filter_result = filter_result[0]
        res_args = result_filter(model_summary, filter_result)
    else:
        res_args = [np.arange(len(ms)) for ms in model_summary]
    var_keys = model_summary[0][0].keys()
    var_keys.remove("Para Position at Term")
    barfig, barax = pl.subplots(1, 5, figsize=[16, 5])
#    marralgs = Marr2Algorithms(para_positions, strike_params)
#    marralgs.run_marr2_models()
    #note this should change if real fish is not arg 0. just make sure it
    #always is.
 #   marr_static_boutcount = np.array(marralgs.bouts_static)[res_args[0]]
#    marr_bayes_boutcount = np.array(marralgs.bouts_bayes)[res_args[0]]

    for var_ind, variable in enumerate(var_keys):
        mapped_values = [np.array(mv)[res_args[i]] for i, mv in enumerate(
            score_query(variable, model_summary))]
        if variable == "Result":
            mapped_values = [np.array(np.count_nonzero(a==1)) for a in mapped_values]
            # marr_static_results = np.sum(
            #     [1 for ma in marr_static_boutcount if not math.isnan(ma)])
            # marr_bayes_results = np.sum(
            #     [1 for ma in marr_bayes_boutcount if not math.isnan(ma)])
            # mapped_values.append(np.array(marr_static_results))
            # mapped_values.append(np.array(marr_bayes_results))
        if variable == "Number of Bouts":
            pass
            # so mapped values has 4 entries, one for each model
            # mapped_values.append(np.array(marr_static_boutcount))
            # mapped_values.append(np.array(marr_bayes_boutcount))
        barax[var_ind].set_title(variable)
        print([mv.shape for mv in mapped_values])
        if plot_type == 'bar' or variable == "Result":
            sb.barplot(data=mapped_values, ax=barax[var_ind])
        elif plot_type == 'violin':
            sb.violinplot(data=mapped_values, ax=barax[var_ind])
        elif plot_type == 'box':
            sb.boxplot(data=mapped_values, ax=barax[var_ind], whis=[5, 95],
                       showfliers=False)

#            sb.boxplot(data=mapped_values, ax=barax[var_ind])
    return barfig


def score_query(variable, ms, *func):

    # want to change this so it takes a result.
    # want to query on results where fish catches and compare to when other models dont
    # could also query on whenever any model succeeds

    f = lambda y: map(lambda x: x[variable], y)
    if func != ():
        func = func[0]
        return map(func, map(f, ms))
    else:
        return map(f, ms)


def find_refractory_periods(rfo):
    strike_refractory_list = [b["Interbouts"][-1] - b["Bout Durations"][-2]
                              for b in [rfo.model_input(i)
                                        for i in range(len(rfo.hunt_ids))]]
    all_refractory_zip = [zip(b["Bout Durations"], b["Interbouts"][1:]) for b in [rfo.model_input(i)
                                                                                  for i in range(len(rfo.hunt_ids))]]
    refractory_all = [[v[1] - v[0] for v in zip_ib_db] for zip_ib_db in all_refractory_zip]
    median_strike_refract = np.median(strike_refractory_list)
    median_hb_refract = np.median(np.concatenate(refractory_all))
    return median_strike_refract, median_hb_refract


def slice_dataframe_for_regmodels(hunt_db):
    slice_bn_neg = hunt_db['Bout Number'] >= 0
    hdb_noneg = hunt_db[slice_bn_neg]
    slice_fails = hdb_noneg['Strike Or Abort'] < 3
    return hdb_noneg[slice_fails]

def slice_dataframe_for_bout0(hunt_db):
    slice_bout_0 = hunt_db['Bout Number'] == 0
    hdb_bout0 = hunt_db[slice_bout_0]
    slice_fails = hdb_bout0['Strike Or Abort'] < 3
    return hdb_bout0[slice_fails]


def strike(p, strike_means, strike_std, confidence_interval):
    prob_az = norm.cdf(p["Para Az"], strike_means[0], strike_std[0])
    prob_alt = norm.cdf(p["Para Alt"], strike_means[1], strike_std[1])
    if p["Para Az"] > strike_means[0]:
        prob_az = 1 - prob_az
    if p["Para Alt"] > strike_means[1]:
        prob_alt = 1 - prob_alt
    # max prob when multiplying all cdfs together is .5**3. normalize. 
    normalize_prob = prob_alt * prob_az * (1 / .25)
    if p["Para Dist"] < (
            strike_means[2] + 2*strike_std[
                2]) and normalize_prob > confidence_interval:
        return True
    else:
        return False

def static_strike(p, strike_means, strike_std):
    num_stds = 1.5
    in_az_win = (p["Para Az"] < strike_means[0] + num_stds * strike_std[0] and
                 p["Para Az"] > strike_means[0] - num_stds * strike_std[0])
    in_alt_win = (p["Para Alt"] < strike_means[1] + num_stds * strike_std[1] and
                 p["Para Alt"] > strike_means[1] - num_stds * strike_std[1])
    in_dist_win = p["Para Dist"] <= strike_means[2] + num_stds * strike_std[2]
    if in_az_win and in_alt_win and in_dist_win:
        return True
    else:
        return False


def strike_prob(p, strike_means, strike_std, strike_ci):
    prob_az = norm.cdf(p["Para Az"], strike_means[0], strike_std[0])
    prob_alt = norm.cdf(p["Para Alt"], strike_means[1], strike_std[1])
    prob_dist = norm.cdf(p["Para Dist"], strike_means[2], strike_std[2])
    if p["Para Az"] > strike_means[0]:
        prob_az = 1 - prob_az
    if p["Para Alt"] > strike_means[1]:
        prob_alt = 1 - prob_alt
    if p["Para Dist"] > strike_means[2]:
        prob_dist = 1 - prob_dist

    # max prob when multiplying all cdfs together is .5**3. normalize. 
    normalize_prob = prob_alt * prob_az * prob_dist * (1 / .125)
    if normalize_prob > strike_ci:
        return True
    else:
        return False

    
def score_trajectory_similarity(plot_or_not, sim1, sim2):
    # call this with flag that para is REAL (b/c have real fish record to compare to)
    # if flag is sim, avoid the real parts or ignore them.

    # sims both contain the bout durations and start times. do not record values
    # for scoring inside the bouts.
    bout_times = sim1.interbouts
    bout_durations = sim1.bout_durations
    avoid_indices = np.concatenate(
        [np.arange(ib, ib+bdur) for ib, bdur in zip(
            bout_times, bout_durations)])
    make_vr_movies(sim1, sim2)
    m1_yaw = np.delete(np.array(sim1.fish_yaw), avoid_indices)
    m1_pitch = np.delete(np.array(sim1.fish_pitch), avoid_indices)
    m1_xyz = np.delete(
        np.array([np.array(xyz) for xyz in sim1.fish_xyz]), avoid_indices)
    m2_yaw = np.delete(np.array(sim2.fish_yaw), avoid_indices)
    m2_pitch = np.delete(np.array(sim2.fish_pitch), avoid_indices)
    m2_xyz = np.delete(np.array(
        [np.array(xyz) for xyz in sim2.fish_xyz]), avoid_indices)
    mag_diff_xyz = np.array(
        [np.linalg.norm(r-m) for r, m in zip(m1_xyz, m2_xyz)])
    diff_pitch = np.abs([b-a for a, b in zip(m1_pitch, m2_pitch)])
    dy_raw = [b-a for a, b in zip(m1_yaw, m2_yaw)]
    diff_yaw = np.abs(yaw_fix(dy_raw))
    if plot_or_not:
        # zip keeps the scales the same
        rx = [r[0] for r, m in zip(m1_xyz, m2_xyz)]
        ry = [r[1] for r, m in zip(m1_xyz, m2_xyz)]
        rz = [r[2] for r, m in zip(m1_xyz, m2_xyz)]
        mx = [m[0] for r, m in zip(m1_xyz, m2_xyz)]
        my = [m[1] for r, m in zip(m1_xyz, m2_xyz)]
        mz = [m[2] for r, m in zip(m1_xyz, m2_xyz)]
        m1_yaw = m1_yaw[0:len(rx)]
        m1_pitch = m1_pitch[0:len(rx)]
        m2_yaw = m2_yaw[0:len(rx)]
        m2_pitch = m2_pitch[0:len(rx)]
        cmap1 = pl.get_cmap('cool')
        norm1 = Normalize(vmin=0, vmax=len(rx))
        scalarMap1 = cm.ScalarMappable(norm=norm1, cmap=cmap1)
        rgba_vals1 = scalarMap1.to_rgba(range(len(rx)))
        cmap2 = pl.get_cmap('coolwarm')
        norm2 = Normalize(vmin=0, vmax=len(mx))
        scalarMap2 = cm.ScalarMappable(norm=norm2, cmap=cmap2)
        rgba_vals2 = scalarMap2.to_rgba(range(len(mx)))
        bout_times = sim1.interbouts[sim1.interbouts < len(rx)]
        bout_ends = sim1.bout_durations + sim1.interbouts
        bout_ends = bout_ends[bout_ends < len(rx)]
        fig = pl.figure(figsize=(15, 15))
        fig.suptitle(
            'Transparent and Triangles = Model 1, Solid and Circles = Model 2')
        gs = gridspec.GridSpec(4, 4)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[2:, 0:2])
        ax3 = fig.add_subplot(gs[2:, 2:])
        ax4 = fig.add_subplot(gs[0:2, 2:])
        ax1.plot(rx, 'r', linewidth=1)
        ax1.plot(mx, 'r', alpha=.5, linewidth=1)
        ax1.plot(ry, 'g', linewidth=1)
        ax1.plot(my, 'g', alpha=.5, linewidth=1)
        ax1.plot(rz, 'b', linewidth=1)
        ax1.plot(mz, 'b', alpha=.5, linewidth=1)
        ax1.plot(bout_times, [rx[bt] for bt in bout_times], 'c', marker='.', linestyle='None')
        ax1.plot(bout_times, [ry[bt] for bt in bout_times], 'c', marker='.', linestyle='None')
        ax1.plot(bout_times, [rz[bt] for bt in bout_times], 'c', marker='.', linestyle='None')
        ax1.plot(bout_ends, [rx[bt] for bt in bout_ends], 'm', marker='.', linestyle='None')
        ax1.plot(bout_ends, [ry[bt] for bt in bout_ends], 'm', marker='.', linestyle='None')
        ax1.plot(bout_ends, [rz[bt] for bt in bout_ends], 'm', marker='.', linestyle='None')
        ax1.set_title('XYZ vs T')
        ax4.plot(m1_yaw, 'r', linewidth=1)
        ax4.plot(m2_yaw, 'r', linewidth=1, alpha=.5)
        ax4.plot(m1_pitch, 'y', linewidth=1)
        ax4.plot(m2_pitch, 'y', linewidth=1, alpha=.5)
        ax4.plot(bout_times, [m2_yaw[bt] for bt in bout_times], 'c', marker='.', linestyle='None')
        ax4.plot(bout_times, [m2_pitch[bt] for bt in bout_times], 'c', marker='.', linestyle='None')
        ax4.plot(bout_ends, [m2_yaw[bt] for bt in bout_ends], 'm', marker='.', linestyle='None')
        ax4.plot(bout_ends, [m2_pitch[bt] for bt in bout_ends], 'm', marker='.', linestyle='None')
        ax4.set_title('Yaw (Red), Pitch (Yellow)')
        ax2.plot(rx, ry, '.5', linewidth=.5)
        ax2.plot(mx, my, '.5', linewidth=.5)
        ax3.plot(rx, rz, '.5', linewidth=.5)
        ax3.plot(mx, mz, '.5', linewidth=.5)
        for i in range(len(rx)):
            ax2.plot(rx[i], ry[i], color=rgba_vals1[i], marker='.', ms=4)
            ax3.plot(rx[i], rz[i], color=rgba_vals1[i], marker='.', ms=4)
        for j in range(len(mx)):
            ax2.plot(mx[j], my[j], color=rgba_vals2[j], marker='.', ms=4)
            ax3.plot(mx[j], mz[j], color=rgba_vals2[j], marker='.', ms=4)
#        ax2.set_xlim([rx[0]-500, rx[0] + 500])
        # ax3.set_xlim([0, 1800])
        # ax2.set_ylim([0, 1800])
        # ax3.set_ylim([0, 1800])
        ax2.set_title('XY')
        ax3.set_title('XZ')
#        ax3.axis('equal')
#        ax2.axis('equal')
        ax2.set_xlim([0, 1800])
        ax3.set_xlim([0, 1800])
        ax2.set_ylim([0, 1800])
        ax3.set_ylim([0, 1800])
        pl.savefig('mod_comparision.pdf')
        pl.subplots_adjust(top=.9, wspace=.8)
        pl.tight_layout()
        pl.show()
#        pl.close()

    return mag_diff_xyz, diff_pitch, diff_yaw



# consider if abs value of yaw and pitch is the correct choice. 

def compare_para_positions_across_models(simlist_by_hunt, mod1, mod2,
                                         pre_or_post, use_abs, varb, *para_coords):
    
    fig, ax = pl.subplots(1, 1)
    colors = sb.color_palette()
    pv1 = []
    pv2 = []
    for siml in simlist_by_hunt:
        if pre_or_post == 0:
            sim_pvs = np.array(
                [(p[0][varb], p[1][varb]) for p in zip(
                    siml[mod1].prebout_para,
                    siml[mod2].prebout_para) if np.isfinite(
                        (p[0][varb], p[1][varb])).all()])
        elif pre_or_post == 1:
            sim_pvs = np.array(
                [(p[0][varb], p[1][varb]) for p in zip(
                    siml[mod1].postbout_para,
                    siml[mod2].postbout_para) if np.isfinite(
                        (p[0][varb], p[1][varb])).all()])
        if sim_pvs.tolist() != []:
            pv1 += sim_pvs[:, 0].tolist()
            pv2 += sim_pvs[:, 1].tolist()

    # this allows you to add coords directly from the marralgs runs
    if para_coords != ():
        if len(para_coords[0]) > 2:
            pv2 = [p[varb] for p in para_coords[0] if np.isfinite(p[varb])]
        else:
            pv1 = [p[varb] for p in para_coords[0][0] if np.isfinite(p[varb])]
            pv2 = [p[varb] for p in para_coords[0][1] if np.isfinite(p[varb])]

    if use_abs:
        pv1 = np.abs(pv1)
        pv2 = np.abs(pv2)

    if varb == "Para Dist":
        ax.set_xlim([0, 1200])
    else:
        ax.set_xlim([-2, 2])

    sb.distplot(pv1,
                fit_kws={"color": colors[0]},
                fit=norm, kde=False, color=colors[0],
                hist_kws={"alpha": .6}, ax=ax)
    sb.distplot(pv2,
                fit_kws={"color": colors[1]},
                fit=norm, kde=False, color=colors[1],
                hist_kws={"alpha": .6}, ax=ax)
    pl.show()
    print("Mean, STD Model 1")
    print np.mean(pv1)
    print np.std(pv1)
    print("Mean, STD Model 2")
    print np.mean(pv2)
    print np.std(pv2)
    return pv1, pv2, ax

        




def validate_model_transformations(prebout_para, postbout_para):

    varbs = ['Para Az', 'Para Alt', 'Para Alt', 'Para Dist']
    fig, ax = pl.subplots(4, 1, figsize=(15, 10))
    cpal = sb.color_palette()
    slopes = []
    yints = []
    for i, pv in enumerate(varbs):
        if i == 1 or i == 2:
            if i == 1:
                p_pairs = [(pre[pv], post[pv]) for pre,
                           post in zip(prebout_para, postbout_para) if pre[pv] > 0]
            elif i == 2:
                p_pairs = [(pre[pv], post[pv]) for pre,
                           post in zip(prebout_para, postbout_para) if pre[pv] < 0]
            p_pairs = [p for p in p_pairs if np.isfinite(p).all()]
        else:
            p_pairs = [(p[0][pv], p[1][pv]) for p in zip(
                prebout_para,
                postbout_para) if np.isfinite((p[0][pv], p[1][pv])).all()]
        v1 = [p[0] for p in p_pairs]
        v2 = [p[1] for p in p_pairs]
        sb.regplot(v1,
                   v2,
                   fit_reg=True, n_boot=100,
                   scatter_kws={'alpha': 0.2},
                   robust=False, color=cpal[i], ax=ax[i])
        rx, ry = ax[i].get_lines()[0].get_data()
        r_slope = np.around((ry[1] - ry[0])/(rx[1] - rx[0]), 2)
        slopes.append(r_slope)
        r_yint = np.around(ry[1] - r_slope*rx[1], 3)
        yints.append(r_yint)
        reg_fit = np.around(pearsonr(v1, v2)[0], 2)
        if pv == 'Para Dist':
            ax[i].set_ylim([0, 1200])
            ax[i].set_xlim([0, 1200])
            ax[i].text(100, 1000, '  ' +
                       str(r_slope) + 'x + ' + str(
                           r_yint) + ', ' + '$r^{2}$ = ' + str(reg_fit**2),
                       color=cpal[i], fontsize=16)

        else:
            ax[i].set_xlim([-2.5, 2.5])
            ax[i].set_ylim([-2.5, 2.5])
            ax[i].text(-2, 2, '  ' +
                       str(r_slope) + 'x + ' + str(
                           r_yint) + ', ' + '$r^{2}$ = ' + str(reg_fit**2),
                       color=cpal[i], fontsize=16)
    return slopes, yints




    

# write out algorithms as lisp programs

# make a plot for the algorithm of alt / postbout alt.

# do 1 fish every 2 days until whole dataset is done. at least 1 para per 3 minutes.

# lasso and ridge for regression


if __name__ == "__main__":
    csv_file = 'wik_bdb/all_huntbouts_rev.csv'
    hb = pd.read_csv(csv_file)
    # CAN FILTER BASED ON HUNT RESULT IF YOU WANT.
    regmodel_input = slice_dataframe_for_regmodels(hb)
    bout0regmodel_input = slice_dataframe_for_bout0(hb)
    independent_regression_model = make_independent_regression_model(
        regmodel_input)
    ir_bout0 = make_independent_regression_model(bout0regmodel_input)
    multiple_regression_model_velocity = make_multiple_regression_model(
        regmodel_input,
        'velocity')
    mv_bout0 = make_multiple_regression_model(
        bout0regmodel_input,
        'velocity')
    multiple_regression_model_position = make_multiple_regression_model(
        regmodel_input,
        'position')
    mp_bout0 = make_multiple_regression_model(
        bout0regmodel_input,
        'position')

    para_model = pickle.load(open(os.getcwd() + '/pmm.pkl', 'rb'))
    np.random.seed()
    sequence_length = 10000
    strike_params = characterize_strikes(hb)
    str_refractory = 0
    strike_params.append(str_refractory)

    # Dict fields are  "Model Type" : Real, Regression, Bayes, Ideal, Random. This is the fish model
    #                  "Real or Sim": Real, Sim, which generates a para or uses real para coords
    #                  "Willie Mays": Enter number of frames ahead to predict
    #                  "Extrapolate Para": ''
    #                  "Spherical Bouts": Enter the list of possible bouts here
    #
    # Hunt Results are: 1 = Success, 2 = Fail, 3 = Para Rec nans before end of hunt
    
    modlist = [{"Model Type": "Independent Regression", "Real or Sim": "Real"},
               {"Model Type": "Independent Regression", "Real or Sim": "Real", "Extrapolate Para":True},
               {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "All"},
               {"Model Type": "Independent Regression", "Real or Sim": "Real", "Willie Mays":True}]

    modlist4 = [{"Model Type": "Real Coords", "Real or Sim": "Real"},
                {"Model Type": "Independent Regression", "Real or Sim": "Real"},
                {"Model Type": "Independent Regression", "Real or Sim": "Real", "Extrapolate Para": True},
                {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "All"},
                {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "All", "Extrapolate Para":True}]

    # modlist2 = [{"Model Type": "Real Coords", "Real or Sim": "Real"},
    #             {"Model Type": "Independent Regression", "Real or Sim": "Real"},
    #             {"Model Type": "Multiple Regression Position", "Real or Sim": "Real"}]


    modlist2 = [{"Model Type": "Real Coords", "Real or Sim": "Real"},
                {"Model Type": "Multiple Regression Position", "Real or Sim": "Real"},
                {"Model Type": "Multiple Regression Velocity", "Real or Sim": "Real"}, 
                {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "All"},
                {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "All", "Extrapolate Para":True},
                {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "Hunt"},
                {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "Hunt", "Extrapolate Para":True}]
#                {"Model Type": "Random", "Real or Sim": "Real", "Spherical Bouts": "All"},
#                {"Model Type": "Random", "Real or Sim": "Real", "Spherical Bouts": "Hunt"}]

    modlist_ci = [{"Model Type": "Real Coords", "Real or Sim": "Real"},
                  {"Model Type": "Real Coords", "Real or Sim": "Real", "Strike CI": .32},
                  {"Model Type": "Multiple Regression Position", "Real or Sim": "Real"},
                  {"Model Type": "Multiple Regression Velocity", "Real or Sim": "Real"},
                  {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "Hunt"},
                  {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "Hunt", "Extrapolate Para": True},
                  {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "Hunt", "Strike CI": .32},
                  {"Model Type": "Ideal", "Real or Sim": "Real", "Spherical Bouts": "Hunt", "Extrapolate Para": True, "Strike CI": .32}]





               

    modlist_bayes = [{"Model Type": "Bayes", "Real or Sim": "Real"}]
    actions = [1, 2]


    # this is the complete dataset
    
    new_wik = ['090418_3', '090418_4', '090418_5', '090418_7',
               '090518_1', '090518_2', '090518_3', '090518_4', '090518_5', '090518_6',
               '090618_1', '090618_2', '090618_3', '090618_4', '090618_5', '090618_6',
               '090718_1', '090718_2', '090718_3', '090718_4', '090718_5', '090718_6',
               '091118_1', '091118_2', '091118_3', '091118_4', '091118_5', '091118_6',
               '091218_1', '091218_2', '091218_3', '091218_4', '091218_5', '091218_6',
               '091318_1', '091318_2', '091318_3', '091318_4',  '091318_5','091318_6',
               '091418_1', '091418_2', '091418_3', '091418_4', '091418_5', '091418_6']


    # this is the original dataset
    
    new_wik_subset = ['091418_1', '091418_2', '091418_3', '091418_4', '091418_6',
                      '091318_1', '091318_2', '091318_3', '091318_4', '091318_5', '091318_6',
                      '091218_1', '091218_2', '091218_3', '091218_4', '091218_5', '091218_6',
                      '091118_1', '091118_2', '091118_3', '091118_4', '091118_5',
                      '090418_3', '090418_4']


    ms, p_inits, simlist_by_model, simlist_by_hunt = model_wrapper(
        new_wik, strike_params,
        para_model, modlist_ci, hb, actions, 1)
    np.save('ms.npy', ms)
    np.save('p_inits.npy', p_inits)
    np.save('strike_params.npy', strike_params)
#    np.save('simlist_by_hunt.npy', simlist_by_hunt)
    sq = score_query("Result", ms, lambda x: Counter(x))
    print sq


    

    
#    simlist, bpm = model_wrapper(real_fish_object, strike_params, para_model, modlist4)


    # for i in real_fish_object.hunt_ids:
    #     simlist, bpm = view_and_sim_hunt(real_fish_object, strike_params, para_model, modlist4, i)
    #     print(sim.score_model(True))

    
#    score_and_view(simlist, 16, 17)
#    total_bouts = [np.sum([b[0] for b in bp]) for bp in bpm]
 #   result_counts_per_model = [Counter([b[-1] for b in bp]) for bp in bpm]
#    results_per_hunt = single_hunt_results(rfo, bpm, 6, modlist4)


    
    # for i in range(len(rfo.hunt_ids) * 2):
    #     if i % 2 == 0:
    #         score_and_view(simlist, i, i+1)

#    score_and_view(simlist, 2, 3)
    
    




