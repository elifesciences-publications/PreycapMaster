import csv
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

class PreyCap_Simulation:
    def __init__(self, fishmodel, paramodel, simlen, simulate_para):
        self.fishmodel = fishmodel
        self.paramodel = paramodel
        self.sim_length = simlen
        self.para_states = []
        self.para_xyz = [fishmodel.real_hunt_df["Para XYZ"][0],
                         fishmodel.real_hunt_df["Para XYZ"][1],
                         fishmodel.real_hunt_df["Para XYZ"][2]]
        self.para_spherical = []
        self.fish_xyz = [fishmodel.real_hunt_df["Initial Conditions"][0]]
        self.fish_bases = []
        self.fish_pitch = [fishmodel.real_hunt_df["Initial Conditions"][1]]
        self.fish_yaw = [fishmodel.real_hunt_df["Initial Conditions"][2]]
        self.simulate_para = simulate_para
        self.interbouts = self.fishmodel.interbouts
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
        framecounter = 0
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

            fish_basis = fishxyz_to_unitvecs(self.fish_xyz[-1],
                                             self.fish_yaw[-1],
                                             self.fish_pitch[-1])
            self.fish_bases.append(fish_basis[1])
            if framecounter == len(px):
                print("hunt epoch complete")
                break
            para_spherical = p_map_to_fish(fish_basis[1],
                                           fish_basis[0],
                                           fish_basis[3],
                                           fish_basis[2],
                                           [px[framecounter],
                                            py[framecounter],
                                            pz[framecounter]],
                                           0)
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
                # note that para may be "eaten" by model faster than in real life (i.e. it enters the strike zone)
                # 
                
                break

            ''' you will store these vals in a
            list so you can determine velocities and accelerations '''

            if framecounter in self.interbouts:
                fish_bout = self.fishmodel.model(para_varbs)
                print para_varbs
                dx, dy, dz = sphericalbout_to_xyz(fish_bout[0],
                                                  fish_bout[1],
                                                  fish_bout[2],
                                                  fish_basis[1],
                                                  fish_basis[3],
                                                  fish_basis[2])
                new_xyz = self.fish_xyz[-1] + np.array([dx, dy, dz])
                self.fish_xyz.append(new_xyz)
                self.fish_pitch.append(self.fish_pitch[-1] + fish_bout[3])
                self.fish_yaw.append(self.fish_yaw[-1] + fish_bout[4])
            else:
                self.fish_xyz.append(self.fish_xyz[-1])
                self.fish_pitch.append(self.fish_pitch[-1])
                self.fish_yaw.append(self.fish_yaw[-1])

            framecounter += 1
        return self.fishmodel.num_bouts_generated
    

class FishModel:
    def __init__(self, modchoice, strike_params, real_hunt_df):
        self.bdb_file = bl.bayesdb_open('Bolton_HuntingBouts_Sim_inverted.bdb')
        if modchoice == 0:
            self.model = (lambda pv: self.regression_model(pv))
        elif modchoice == 1:
            self.model = (lambda pv: self.bdb_model(pv))
        elif modchoice == 2:
            self.model = (lambda pv: self.real_fish(pv))
            
        self.strike_params = strike_params
        self.real_hunt_df = real_hunt_df
        self.interbouts = real_hunt_df["Interbouts"]
        self.interbouts = np.cumsum(
            self.interbouts) + real_hunt_df["First Bout Delay"]
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
        if (
                p["Para Dist"] < self.strike_params[2]) and (
                    np.abs(p["Para Az"]) < self.strike_params[0]) and (
                        np.abs(p["Para Alt"]) < self.strike_params[1]):
            return True
        else:
            return False

    def real_fish(self, para_varbs):
        if self.num_bouts_generated == len(self.real_hunt_df["Hunt Dataframe"]) - 1:
            hunt_df = self.real_hunt_df[
                "Hunt Dataframe"].loc[-1]
        else:
            hunt_df = self.real_hunt_df[
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

    def ideal_model(self):

    # This model will assess the postbout az, alt, and dist and ask while whether there was a better bout to be had. can do this purely from
    # the output of the csv file. i.e. if postbout az is opp in sign, look for bouts that wouldve gotten it closer (i.e. less az). see if you could hav
        # gotten 0 az, 0 alt, 0 dist (i.e. food in the mouth)
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
                           USING MODEL 37
                           LIMIT 5000 '''.format(**para_varbs))
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
                           USING MODEL 37
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
    avg_strike_position = np.mean(np.abs(strike_characteristics), axis=0)
    return avg_strike_position


csv_file = 'huntbouts_rad.csv'
hb = pd.read_csv(csv_file)
fish_id = '030118_2'
real_fish_object = pd.read_pickle(
    os.getcwd() + '/' + fish_id + '/RealHuntData_' + fish_id + '.pkl')
para_model = pickle.load(open(os.getcwd() + '/pmm.pkl', 'rb'))
np.random.seed()
sequence_length = 10000
strike_params = characterize_strikes(hb)
fish = FishModel(2, strike_params, real_fish_object.model_input(0))
print('Creating Simulator')
sim = PreyCap_Simulation(
    fish,
    para_model,
    sequence_length,
    False)

sim.run_simulation()
# np.save('/home/nightcrawler/PandaModels/para_simulation.npy', sim.para_xyz)
# np.save('/home/nightcrawler/PandaModels/origin_model.npy', sim.fish_xyz)
# np.save('/home/nightcrawler/PandaModels/uf_model.npy', sim.fish_bases)
np.save('para_simulation.npy', sim.para_xyz)
np.save('origin_model.npy', sim.fish_xyz)
np.save('uf_model.npy', sim.fish_bases)



