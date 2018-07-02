import csv
import copy
import numpy as np
import pickle
from matplotlib import pyplot as pl
import seaborn as sb
import pandas as pd
import os
import math
from toolz.itertoolz import partition
from para_hmm_final import ParaMarkovModel
import pomegranate
import bayeslite as bl
from iventure.utils_bql import query
from iventure.utils_bql import subsample_table_columns
from collections import deque
from master import fishxyz_to_unitvecs, sphericalbout_to_xyz, p_map_to_fish, RealFishControl

# Next steps: make sure bayesDB runs look like real bouts. see what regression does too. 


class PreyCap_Simulation:
    def __init__(self, fishmodel, paramodel, simlen, simulate_para, *para_input):
        self.fishmodel = fishmodel
        self.paramodel = paramodel
        self.sim_length = simlen
        self.para_states = []
        self.model_para_xyz = []
        if para_input == ():
            self.para_xyz = [fishmodel.real_hunt["Para XYZ"][0],
                             fishmodel.real_hunt["Para XYZ"][1],
                             fishmodel.real_hunt["Para XYZ"][2]]
        else:
            self.para_xyz = para_input
        self.para_spherical = []
        self.fish_xyz = [fishmodel.real_hunt["Initial Conditions"][0]]
        self.fish_bases = []
        self.fish_pitch = [fishmodel.real_hunt["Initial Conditions"][1]]
        self.fish_yaw = [fishmodel.real_hunt["Initial Conditions"][2]]
        self.simulate_para = simulate_para
        self.interbouts= self.fishmodel.interbouts
        self.bout_durations = self.fishmodel.bout_durations
        self.interpolate = True
        self.create_para_trajectory()
        
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
            p_position_5 = np.array([
                self.para_xyz[0][5], self.para_xyz[1][5], self.para_xyz[2][5]])
            start_vector = p_position_5 - p_start_position
            para_initial_conditions = [p_start_position, start_vector]
            px, py, pz, states, vmax = self.paramodel.generate_para(
                para_initial_conditions[1],
                para_initial_conditions[0], self.sim_length)
            # resets para_xyz for generative model
            self.para_xyz = [np.array(px), np.array(py), np.array(pz)]
            self.para_states = states

        else:
            accelerations = make_accel_vectors()
            self.para_states = self.paramodel.model.predict(accelerations)
            
    def run_simulation(self):

        def project_para(p_coords, nback, proj_forward, curr_frame):
            def final_pos(pc):
                fp = (
                    ((pc[curr_frame] - pc[curr_frame-nback])
                     / float(nback)) * proj_forward) + pc[curr_frame]
                return fp
            p_xyz = [final_pos(p) for p in p_coords]                
            return 
        
        
        hunt_result = 0
        framecounter = 0
        bout_counter = 0
        # spacing is an integer describing the sampling of vectors by the para. i.e.       # draws a new accel vector every 'spacing' frames
        spacing = np.load('spacing.npy')
        vel_history = 6
        # fish_basis is origin, x, y, z unit vecs
        print("Para Shape")
        print self.para_xyz[0].shape[0]
        p_az_hist = deque(maxlen=vel_history)
        p_alt_hist = deque(maxlen=vel_history)
        p_dist_hist = deque(maxlen=vel_history)
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

        while True:
            print framecounter
            fish_basis = fishxyz_to_unitvecs(self.fish_xyz[-1],
                                             self.fish_yaw[-1],
                                             self.fish_pitch[-1])
            self.fish_bases.append(fish_basis[1])
            if framecounter == len(px):
                print("hunt epoch complete w/out strike")
                hunt_result = 2
                break
            self.model_para_xyz = [px[framecounter], py[framecounter], pz[framecounter]]
            para_spherical = p_map_to_fish(fish_basis[1],
                                           fish_basis[0],
                                           fish_basis[3],
                                           fish_basis[2],
                                           self.model_para_xyz, 
                                           0)
            # If project 10 frames forward with known XYZ, futureframe = framecoutner + 10
            # model_para_xyz = [px[futureframe], py[futureframe], pz[futureframe]]
            p_az_hist.append(para_spherical[0])
            p_alt_hist.append(para_spherical[1])
            p_dist_hist.append(para_spherical[2])
            para_varbs = {"Para Az": para_spherical[0],
                          "Para Alt": para_spherical[1],
                          "Para Dist": para_spherical[2],
                          "Para Az Velocity": np.mean(np.diff(p_az_hist)),
                          "Para Alt Velocity": np.mean(np.diff(p_alt_hist)),
                          "Para Dist Velocity": np.mean(np.diff(p_dist_hist))}

            if framecounter == 2000:
                print("EVASIVE PARA!!!!!")
                self.para_xyz[0] = px
                self.para_xyz[1] = py
                self.para_xyz[2] = pz
                break
            
            if self.fishmodel.strike(para_varbs):
                print("Para Before Strike")
                print para_varbs
                print("STRIKE!!!!!!!!")
                nanstretch = np.full(
                    len(px)-framecounter, np.nan).tolist()
                print('para lengths')
                print len(nanstretch)
                self.para_xyz[0] = np.concatenate(
                    (px[0:framecounter], nanstretch), axis=0)
                self.para_xyz[1] = np.concatenate(
                    (py[0:framecounter], nanstretch), axis=0)
                self.para_xyz[2] = np.concatenate(
                    (pz[0:framecounter], nanstretch), axis=0)
                np.save('strikelist.npy', np.zeros(framecounter).tolist() + [1]
                        + np.zeros(len(px) - framecounter -1).tolist() )
                print self.para_xyz[0].shape
                print('len fishxyz')
                print len(self.fish_xyz)
                hunt_result = 1
                # note that para may be "eaten" by model faster than in real life (i.e. it enters the strike zone)
                # 
                break

            ''' you will store these vals in a
            list so you can determine velocities and accelerations '''
            if framecounter in self.interbouts:

                print para_varbs
                if fishmodel.linear_predict_para:
                    self.model_para_xyz = project_para([px, py, pz], 5, 10, framecounter):
                if fishmodel.bouts_forward != 0:
                    future_frame = framecounter + fishmodel.bouts_forward
                    self.model_para_xyz = [px[future_frame], py[future_frame], pz[future_frame]]
                if not np.isfinite(para_varbs.values()).all() or not np.isfinite(self.model_para_xyz).all():
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
                if self.interpolate:
                    x_prog = np.linspace(
                        self.fish_xyz[-1][0],
                        self.fish_xyz[-1][0] + dx,
                        self.bout_durations[bout_counter]).tolist()
                    y_prog = np.linspace(
                        self.fish_xyz[-1][1],
                        self.fish_xyz[-1][1] + dy,
                        self.bout_durations[bout_counter]).tolist()
                    z_prog = np.linspace(
                        self.fish_xyz[-1][2],
                        self.fish_xyz[-1][2] + dz,
                        self.bout_durations[bout_counter]).tolist()
                    yaw_prog = np.linspace(
                        self.fish_yaw[-1],
                        self.fish_yaw[-1] + fish_bout[4],
                        self.bout_durations[bout_counter]).tolist()
                    pitch_prog = np.linspace(
                        self.fish_pitch[-1],
                        self.fish_pitch[-1] + fish_bout[3],
                        self.bout_durations[bout_counter]).tolist()
                    self.fish_xyz += zip(x_prog, y_prog, z_prog)
                    self.fish_pitch += pitch_prog
                    self.fish_yaw += yaw_prog
                    for x, y, z, p, yw in zip(x_prog[1:],
                                              y_prog[1:],
                                              z_prog[1:], pitch_prog[1:], yaw_prog[1:]):
                        fish_basis = fishxyz_to_unitvecs((x,y,z), yw, p)
                        self.fish_bases.append(fish_basis[1])
                    framecounter += self.bout_durations[bout_counter]

                else:
                    new_xyz = self.fish_xyz[-1] + np.array([dx, dy, dz])
                    self.fish_xyz.append(new_xyz)
                    self.fish_pitch.append(self.fish_pitch[-1] + fish_bout[3])
                    self.fish_yaw.append(self.fish_yaw[-1] + fish_bout[4])
                    framecounter += 1
                bout_counter += 1
            else:
                self.fish_xyz.append(self.fish_xyz[-1])
                self.fish_pitch.append(self.fish_pitch[-1])
                self.fish_yaw.append(self.fish_yaw[-1])
                framecounter += 1
        if hunt_result == 3:
            para_varbs = last_finite_pvarbs
        return self.fishmodel.num_bouts_generated, para_varbs, hunt_result
    

class FishModel:
    def __init__(self, modchoice, strike_params, real_hunt_input, *spherical_bouts):
#        self.bdb_file = bl.bayesdb_open('Bolton_HuntingBouts_Sim_inverted.bdb')
        self.modchoice = modchoice
        if spherical_bouts != ():
            self.spherical_bout_bag = spherical_bouts[0]
            self.spherical_huntbout_bag = spherical_bouts[1]
        else:
            self.spherical_bout_bag = self.spherical_huntbout_bag = spherical_bouts
        self.bdb_file = bl.bayesdb_open('bdb_hunts_inverted.bdb')
        if modchoice == 0:
            self.model = (lambda pv: self.regression_model(pv))
        elif modchoice == 1:
            self.model = (lambda pv: self.bdb_model(pv))
        elif modchoice == 2:
            self.model = (lambda pv: self.real_fish(pv))
        elif 3 <= modchoice <= 7:
            self.model = (lambda pv: self.ideal_model(pv))
        # 3 is greedy1.
        # 4 is greedy + 10 frames
        # 5 is linear interpolation -- can also do this with regression 
        # 6 is greedy1 with only huntbouts
        # 7 is greedy + 10 frames with only huntbouts
        self.linear_predict_para = False
        self.strike_means = strike_params[0]
        self.strike_std = strike_params[1]
        self.real_hunt = real_hunt_input
        self.interbouts = real_hunt_input["Interbouts"]
        self.bout_durations = real_hunt_input["Bout Durations"]
        self.interbouts = np.cumsum(
            self.interbouts) + real_hunt_input["First Bout Delay"]
        # note that the fish is given 5 frames before initializing a bout.
        # this is so the model has a proper velocity to work with. occurs
        # at the exact same time as the fish bouts in reality. para
        # trajectory is contwin + intwin - 5 to start. 
        self.num_bouts_generated = 0


    def generate_random_interbouts(self, num_bouts):
        all_ibs = np.floor(np.random.random(num_bouts) * 100)
        bout_indices = np.cumsum(all_ibs)
        return bout_indices

    def strike(self, p):
        in_az_win = (p["Para Az"] < self.strike_means[0] + self.strike_std[0] and
                     p["Para Az"] > self.strike_means[0] - self.strike_std[0])
        in_alt_win = (p["Para Alt"] < self.strike_means[1] + self.strike_std[1] and
                     p["Para Alt"] > self.strike_means[1] - self.strike_std[1])
        in_dist_win = (p["Para Dist"] < self.strike_means[2] + self.strike_std[2] and
                     p["Para Dist"] > self.strike_means[2] - self.strike_std[2])
        if in_az_win and in_alt_win and in_dist_win:
            return True
        else:
            return False

    def real_fish(self, para_varbs):
        # if self.num_bouts_generated == len(self.real_hunt["Hunt Dataframe"]) - 1:
        #     hunt_df = self.real_hunt[
        #         "Hunt Dataframe"].loc[-1]
#        else:
        hunt_df = self.real_hunt[
            "Hunt Dataframe"].loc[self.num_bouts_generated]
        print self.num_bouts_generated        
        bout = np.array(
            [hunt_df["Bout Az"],
             hunt_df["Bout Alt"],
             hunt_df["Bout Dist"],
             hunt_df["Bout Delta Pitch"],
             -1*hunt_df["Bout Delta Yaw"]])
        print('bout')
        print bout
        self.num_bouts_generated += 1
        return bout

    def random_model(self, para_varbs):
        self.num_bouts_generated += 1
        r_int = np.random.randint(0, len(self.spherical_bout_bag))
        rand_bout = self.spherical_bout_bag[r_int]
        bout = np.array(
            [rand_bout["Bout Az"],
             rand_bout["Bout Alt"],
             rand_bout["Bout Dist"],
             rand_bout["Delta Pitch"],
             -1*rand_bout["Delta Yaw"]])

    def random_hunt_model(self, para_varbs):
        self.num_bouts_generated += 1
        r_int = np.random.randint(0, len(self.spherical_huntbout_bag))
        rand_bout = self.spherical_huntbout_bag[r_int]
        bout = np.array(
            [rand_bout["Bout Az"],
             rand_bout["Bout Alt"],
             rand_bout["Bout Dist"],
             rand_bout["Delta Pitch"],
             -1*rand_bout["Delta Yaw"]])

    def ideal_model(self, para_varbs):

        def score_para(p_results, strike_varbs):
            # first ask if any para are IN strike zone
            strikes = np.array([self.strike(p) for p in p_results])
            if strikes.any():
                strike_args = np.argwhere(strikes)
                rand_strike = np.random.randint(strike_args.shape[0])
                bout_choice = strike_args[rand_strike][0]
                return bout_choice
            az = np.array([p['Para Az'] for p in p_results]) - self.strike_means[0]
            alt = np.array([p['Para Alt'] for p in p_results]) - self.strike_means[1]
            dist = np.array([p['Para Dist'] for p in p_results]) - self.strike_means[2]
            norm_az = az / np.std(az)
            norm_alt = alt / np.std(alt)
            norm_dist = dist / np.std(dist)
            score = norm_az + norm_alt + norm_dist
            bout_choice = np.argmin(score)
            return bout_choice
 
        # Use this model for all greedy models and simply manipulate the model_para_xyz
        self.num_bouts_generated += 1
        if self.modchoice == 6:
            boutbag = self.spherical_huntbout_bag
        else:
            boutbag = self.spherical_bout_bag
        fish_basis = fishxyz_to_unitvecs(self.fish_xyz[-1],
                                         self.fish_yaw[-1],
                                         self.fish_pitch[-1])
        para_results = []
        for bt in self.spherical_bout_bag:
            dx, dy, dz = sphericalbout_to_xyz(bt["Bout Az"],
                                              bt["Bout Alt"],
                                              bt["Bout Dist"],
                                              fish_basis[1],
                                              fish_basis[3],
                                              fish_basis[2])
            temp_yaw = bt["Delta Yaw"] + self.fish_yaw[-1]
            temp_pitch = bt["Delta Pitch"] + self.fish_pitch[-1]
            temp_xyz = np.array(self.fish_xyz[-1]) + np.array([dx, dy, dz])
            temp_fish_basis = fishxyz_to_unitvecs(temp_xyz, 
                                                  temp_yaw, 
                                                  temp_pitch)
            postbout_para_spherical = p_map_to_fish(temp_fish_basis[1],
                                                    temp_fish_basis[0],
                                                    temp_fish_basis[3],
                                                    temp_fish_basis[2],
                                                    self.model_para_xyz,
                                                    0)
            para_results.append(postbout_para_spherical)
        best_bout = score_para(para_results)
            
            
        
        # first have to generate the bout 

        
        bout_az = (para_varbs['Para Az'] * 1.36) + .02
        # NOTE THAT YOUR BOUT DESCRIPTOR IN MASTER.PY NOW OUTPUTS THE RIGHT SIGN OF DELTA YAW. CAN GET RID OF THISINVERSION WHEN ANALYZING NEW DATASET 
        bout_yaw = -1 * ((.46 * para_varbs['Para Az']) - .02)
        bout_alt = (1.5 * para_varbs['Para Alt']) + -.37
        bout_pitch = (.27 * para_varbs['Para Alt']) - .04
        bout_dist = (.09 * para_varbs['Para Dist']) + 29
        print para_varbs
        bout_array = np.array([bout_az,
                               bout_alt,
                               bout_dist, bout_pitch, bout_yaw])
        noise_array = np.ones(5)
        # noise_array = np.array(
        # [(np.random.random() * .4) + .8 for i in bout_array])
        bout = bout_array * noise_array
        return bout


    # This model will assess the postbout az, alt, and dist and ask while whether there was a better bout to be had. can do this purely from
    # the output of the csv file. i.e. if postbout az is opp in sign, look for bouts that wouldve gotten it closer (i.e. less az). see if you could hav
        # gotten 0 az, 0 alt, 0 dist (i.e. food in the mouth)
        possible_bouts = self.spherical_bout_bag
        
        pass

    

    def regression_model(self, para_varbs):
        self.num_bouts_generated += 1
        bout_az = (para_varbs['Para Az'] * 1.36) + .02
        # NOTE THAT YOUR BOUT DESCRIPTOR IN MASTER.PY NOW OUTPUTS THE RIGHT SIGN OF DELTA YAW. CAN GET RID OF THISINVERSION WHEN ANALYZING NEW DATASET 
        bout_yaw = -1 * ((.46 * para_varbs['Para Az']) - .02)
        bout_alt = (1.5 * para_varbs['Para Alt']) + -.37
        bout_pitch = (.27 * para_varbs['Para Alt']) - .04
        bout_dist = (.09 * para_varbs['Para Dist']) + 29
        print para_varbs
        bout_array = np.array([bout_az,
                               bout_alt,
                               bout_dist, bout_pitch, bout_yaw])
        noise_array = np.ones(5)
        # noise_array = np.array(
        # [(np.random.random() * .4) + .8 for i in bout_array])
        bout = bout_array * noise_array
        return bout

    def bdb_model(self, para_varbs):
        self.num_bouts_generated += 1
        invert = True
#        invert = False
        sampling = 'median'
#        sampling = 'sample'
        
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

        if sampling == 'median':
            df_sim = query(self.bdb_file,
                           ''' SIMULATE "Bout Az", "Bout Alt",
                           "Bout Dist", "Bout Delta Pitch", "Bout Delta Yaw"
                           FROM bout_population
                           GIVEN "Para Az" = {Para Az},
                           "Para Alt" = {Para Alt},
                           "Para Dist" = {Para Dist},
                           "Para Az Velocity" = {Para Az Velocity},
                           "Para Alt Velocity" = {Para Alt Velocity},
                           "Para Dist velocity" = {Para Dist Velocity}
--                           USING MODEL 37
                           LIMIT 500 '''.format(**para_varbs))
            bout_az = df_sim['Bout Az'].median()
            bout_alt = df_sim['Bout Alt'].median()
            bout_dist = df_sim['Bout Dist'].median()
            bout_pitch = df_sim['Bout Delta Pitch'].median()
            bout_yaw = -1*df_sim['Bout Delta Yaw'].median()

        elif sampling == 'sample':
            df_sim = query(self.bdb_file,
                           ''' SIMULATE "Bout Az", "Bout Alt",
                           "Bout Dist", "Bout Delta Pitch", "Bout Delta Yaw"
                           FROM bout_population
                           GIVEN "Para Az" = {Para Az},
                           "Para Alt" = {Para Alt},
                           "Para Dist" = {Para Dist},
                           "Para Az Velocity" = {Para Az Velocity},
                           "Para Alt Velocity" = {Para Alt Velocity},
                           "Para Dist velocity" = {Para Dist Velocity}
  --                         USING MODEL 37
                           LIMIT 1 '''.format(**para_varbs))
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
        if invert:
            b_dict = invert_bout(b_dict, inv_az, inv_alt)
        bout = np.array([b_dict["Bout Az"],
                         b_dict["Bout Alt"],
                         b_dict["Bout Dist"],
                         b_dict["Bout Delta Pitch"],
                         b_dict["Bout Delta Yaw"]])
        return bout


def mapped_para_generator(hb_data):
    ind = np.int(np.random.random() * len(hb_data['Para Az']))
    p_varbs = hb_data.iloc[ind][6:14]
    if len([a for a in p_varbs.values if math.isnan(a)]) != 0:
        return mapped_para_generator(hb_data)
    return p_varbs


def generate_bouts_from_csv(data_fr):
    num_samples = 2000
    file_id = os.getcwd() + '/regmodel'
    header = [a for a in data_fr.keys()[1:14]]
    with open(file_id + '.csv', 'wb') as csvfile:
        output_data = csv.writer(csvfile)
        for i in range(num_samples):
            para_vrbs = mapped_para_generator(data_fr)
            bout = boutgen_from_regression(para_vrbs)
            if i == 0:
                output_data.writerow(header)
            output_data.writerow(bout)

#use this to characterize how similar model bouts are to real choices


def characterize_strikes(hb_data):
    strike_characteristics = []
    for i in range(len(hb_data["Bout Number"])):
        if hb_data["Bout Number"][i] == -1:
            if hb_data["Strike Or Abort"][i] < 3:
                strike = np.array([hb_data["Para Az"][i],
                                   hb_data["Para Alt"][i],
                                   hb_data["Para Dist"][i]])
                if np.isfinite(strike).all():
                    strike_characteristics.append(strike)
    avg_strike_position = np.mean(strike_characteristics, axis=0)
    std = np.std(strike_characteristics, axis=0)
    return avg_strike_position, std

def realhunt_allframes(rfo, h_id):
    delay = rfo.firstbout_para_intwin
    frames = np.array(rfo.hunt_frames[h_id]) - delay
    mod_input = rfo.model_input(h_id)
    para_xyz = [mod_input["Para XYZ"][0],
                mod_input["Para XYZ"][1],
                mod_input["Para XYZ"][2]]

    np.save('ufish.npy', rfo.fish_orientation[frames[0]:frames[1]])
    np.save('ufish_origin.npy', rfo.fish_xyz[frames[0]:frames[1]])
    np.save('3D_paracoords.npy', para_xyz)

def model_vs_real(rfo, strike_params, para_model):
    num_fishmodels = 4
    bout_container = [[] for i in range(num_fishmodels)]
    sequence_length = 10000
    # this will iterate through hunt_ids and through types of Fishmodel.
    # function takes a real_fish_object and creates a Simulation for each hunt id and each model.
    # have to have the Sim output a -1 or something if the fish never catches the para
    for h_ind, hid in enumerate(rfo.hunt_ids):
        for model_number in range(num_fishmodels):
            fish = FishModel(model_number,
                             strike_params, rfo.model_input(h_ind))
            sim = PreyCap_Simulation(
                fish,
                para_model,
                sequence_length,
                False)
            bouts_and_p_params = sim.run_simulation()
            bout_container[model_number].append(bouts_and_p_params)
    return bout_container
    
def model_vs_paramodel(rfo, strike_params, para_model):
    real_model = 2
    num_fishmodels = 4
    bout_container = [[] for i in range(num_fishmodels)]
    sequence_length = 10000
    p_xyz = []
    # Here the initial conditions of each hunt will be kept, and
    # para simulations will proceed from there. 
    for h_ind, hid in enumerate(rfo.hunt_ids):
        for model_number in range(num_fishmodels):
            if model_number == real_model:
                continue
            fish = FishModel(model_number,
                             strike_params, rfo.model_input(h_ind))
            if model_number == 0:
                sim = PreyCap_Simulation(
                    fish,
                    para_model,
                    sequence_length,
                    True)
                p_xyz = sim.para_xyz
            else:
                sim = PreyCap_Simulation(
                    fish,
                    para_model,
                    sequence_length,
                    False,
                    p_xyz)
            bouts_and_p_params = sim.run_simulation()
            bout_container[model_number].append(bouts_and_p_params)
    return bout_container

    # this will take the paramodel object
    

if __name__ == "__main__":
    csv_file = 'huntbouts_rad.csv'
    hb = pd.read_csv(csv_file)
    fish_id = '042318_6'
    real_fish_object = pd.read_pickle(
        os.getcwd() + '/' + fish_id + '/RealHuntData_' + fish_id + '.pkl')
    para_model = pickle.load(open(os.getcwd() + '/pmm.pkl', 'rb'))
    np.random.seed()
    sequence_length = 10000
    strike_params = characterize_strikes(hb)
    #before you start this have to re-run hunted_para_desc in any fish
    h_id = 1
    print("Analyzing Hunt " + str(real_fish_object.hunt_ids[h_id]))
    realhunt_allframes(real_fish_object, h_id)
    fish = FishModel(1, strike_params, real_fish_object.model_input(h_id))
    print('Creating Simulator')
    sim = PreyCap_Simulation(
        fish,
        para_model,
        sequence_length,
        False)
    num_bouts = sim.run_simulation()
    np.save('para_simulation.npy', sim.para_xyz)
    np.save('origin_model.npy', sim.fish_xyz[:-1])
    np.save('uf_model.npy', sim.fish_bases[:-1])



