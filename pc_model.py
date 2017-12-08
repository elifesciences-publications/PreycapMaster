import csv
import numpy as np
import pickle
from matplotlib import pyplot as pl
import seaborn as sb
import pandas as pd
import os
import math
from master import fishxyz_to_unitvecs, sphericalbout_to_xyz, p_map_to_fish
from para_hmm_final import ParaMarkovModel
import pomegranate

# this program depends on master, para_hmm_final, csv output from fish data.
# first magit try1


class PreyCap_Simulation:
    def __init__(self, fishmodel, paramodel, para_init, simlen):
        self.fishmodel = fishmodel
        self.paramodel = paramodel
        self.sim_length = simlen
        self.para_initial_conditions = para_init
        self.para_states = []
        self.para_xyz = []
        self.para_spherical = []
        self.fish_xyz = [np.array([944, 944, 944])]
        self.fish_bases = []
        self.fish_pitch = [0]
        self.fish_yaw = [0]
        self.validation = []

# have to start with an XYZ velocity that reflects the velocity profile of
# d_az, d_alt, d_dist that the fish likes. can randomly draw a para

#also, dont want just simulated para. want to use real para trajectories that you can get out of p3d in the para model. very easy. paramodel will instead just be a function that grabs random XYZ coords of hunted para. 

    def create_para_trajectory(self):
        if isinstance(self.paramodel.model, pomegranate.hmm.HiddenMarkovModel):
            px, py, pz, states, vmax = self.paramodel.generate_para(
                self.para_initial_conditions[1],
                self.para_initial_conditions[0], self.sim_length)
            self.para_xyz = [np.array(px), np.array(py), np.array(pz)]
            self.para_states = states
        else:
            pass
        return vmax
            # this will import a real para trajectory of a hunted para.
            # this will be done by importing the csv file of all para. restrict to
            # bout indicies of and para of type 1 or 2. then you align the fish based on the initial az, alt, and dist (i.e. put it in the place it started the hunt, then start it).

                    
    def run_simulation(self):
        framecounter = 0
        spacing = np.load('spacing.npy')
        interbouts = self.fishmodel.generate_interbouts(100000)
        # fish_basis is origin, x, y, z unit vecs
        print("Para Shape")
        print self.para_xyz[0].shape[0]
        while True:

            # MAKE SURE YOU KNOW WHY YOU HAVE TO CUT ONE OFF THE END
            p_frame = int(np.floor(framecounter / spacing))
            if p_frame == self.para_xyz[0].shape[0]:
                self.fish_xyz = self.fish_xyz[::spacing][:-1]
                self.fish_pitch = self.fish_pitch[::spacing][:-1]
                self.fish_yaw = self.fish_yaw[::spacing][:-1]
                self.fish_bases = self.fish_bases[::spacing]
                break
            fish_basis = fishxyz_to_unitvecs(self.fish_xyz[-1],
                                             self.fish_yaw[-1],
                                             self.fish_pitch[-1])
            self.fish_bases.append(fish_basis[1])
            para_spherical = p_map_to_fish(fish_basis[1],
                                           fish_basis[0],
                                           fish_basis[3],
                                           fish_basis[2],
                                           [self.para_xyz[0][p_frame],
                                            self.para_xyz[1][p_frame],
                                            self.para_xyz[2][p_frame]],
                                           0)
            para_varbs = {"Para Az": para_spherical[0],
                          "Para Alt": para_spherical[1],
                          "Para Dist": para_spherical[2]}
#                          "Delta Az";

            if self.fishmodel.strike(para_varbs):
                print("Para Before Strike")
                print para_varbs
                print("STRIKE!!!!!!!!")
                nanstretch = np.full(
                    len(self.para_xyz[0])-p_frame, np.nan).tolist()
                self.para_xyz[0] = np.concatenate(
                    (self.para_xyz[0][0:p_frame], nanstretch), axis=0)
                self.para_xyz[1] = np.concatenate(
                    (self.para_xyz[1][0:p_frame], nanstretch), axis=0)
                self.para_xyz[2] = np.concatenate(
                    (self.para_xyz[2][0:p_frame], nanstretch), axis=0)
                self.fish_xyz = self.fish_xyz[::spacing][:-1]
                self.fish_pitch = self.fish_pitch[::spacing][:-1]
                self.fish_yaw = self.fish_yaw[::spacing][:-1]
                self.fish_bases = self.fish_bases[::spacing]
                break
            
            ''' you will store these vals in a
            list so you can determine velocities and accelerations '''

            if framecounter in interbouts:
                print p_frame
                fish_bout = self.fishmodel.model(para_varbs)
                dx, dy, dz = sphericalbout_to_xyz(fish_bout[0],
                                                  fish_bout[1],
                                                  fish_bout[2],
                                                  fish_basis[1],
                                                  fish_basis[3],
                                                  fish_basis[2])
                new_xyz = self.fish_xyz[-1] + np.array([dx, dy, dz])
                self.validation.append([['Para Coords XYZ',
                                        self.para_xyz[0][p_frame],
                                        self.para_xyz[1][p_frame],
                                        self.para_xyz[2][p_frame]],
                                       ['Prebout Fish XYZ',
                                        self.fish_xyz[-1]],
                                       ['Prebout Pitch and Yaw',
                                        [self.fish_pitch[-1],
                                         self.fish_yaw[-1]]],
                                       ['Fish XYZ to Unit', fish_basis],
                                       ['Para Varbs', para_varbs],
                                       ['Fish Bout', fish_bout],
                                       ['Delta X,Y,Z from Spherical Transform',
                                        [dx, dy, dz]]])
                                                                              
                self.fish_xyz.append(new_xyz)
                self.fish_pitch.append(self.fish_pitch[-1] + fish_bout[3])
                self.fish_yaw.append(self.fish_yaw[-1] + fish_bout[4])
            else:
                self.fish_xyz.append(self.fish_xyz[-1])
                self.fish_pitch.append(self.fish_pitch[-1])
                self.fish_yaw.append(self.fish_yaw[-1])
                
            framecounter += 1
                

class FishModel:
    def __init__(self, modchoice, strike_params):
        if modchoice == 0:
            self.model = (lambda pv: self.regression_model(pv))
        #this will be the bayesDB model or otehr models
        elif modchoice == 1:
            self.model = (lambda pv: self.regression_model(pv))
        self.strike_params = strike_params
        
    def generate_interbouts(self, num_bouts):
        all_ibs = np.floor(np.random.random(num_bouts) * 100)
#        all_ibs = np.floor(np.random.uniform(low=20, high=60, size=num_bouts))
        bout_indices = np.cumsum(all_ibs)
        return bout_indices

    def ideal_model(self):

    # This model will assess the postbout az, alt, and dist and ask while whether there was a better bout to be had. can do this purely from
    # the output of the csv file. i.e. if postbout az is opp in sign, look for bouts that wouldve gotten it closer (i.e. less az). see if you could hav
        # gotten 0 az, 0 alt, 0 dist (i.e. food in the mouth)
        pass

    def strike(self, p):
        if (
                p["Para Dist"] < self.strike_params[2]) and (
                    np.abs(p["Para Az"]) < self.strike_params[0]) and (
                        np.abs(p["Para Alt"]) < self.strike_params[1]):
            return True
        else:
            return False

        ''' first import all para coords where strikes occurred. create a strike distribution, smooth with kde. ask prob of strike on each bout. execute a strike based on this prob. '''


# bout distance isn't getting translated correctly. if you take away the bout dist, fish are aligning well to the stimulus. 
    def regression_model(self, para_varbs):
        bout_az = (para_varbs['Para Az'] * 1.36) + .02
        bout_yaw = -1 * ((.46 * para_varbs['Para Az']) - .02)
        bout_alt = (1.5 * para_varbs['Para Alt']) + -.37
        bout_pitch = (.27 * para_varbs['Para Alt']) - .04
        bout_dist = (.09 * para_varbs['Para Dist']) + 29
        print para_varbs
        bout_array = np.array([bout_az,
                               bout_alt,
                               bout_dist, bout_pitch, bout_yaw])
        noise_array = np.ones(5)
#        noise_array = np.array(
 #           [(np.random.random() * .4) + .8 for i in bout_array])
        bout = (bout_array * noise_array).tolist()
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
    


    

# HERE GRAB CSV HUNTBOUTS FILE. GENERATE A RANDOM NUMBER GRAB FOR EACH LIST FROM 0-1 TIMES LENGTH OF THE ARRAY toint().
    
csv_file = 'huntbouts_rad.csv'
hb = pd.read_csv(csv_file)


np.random.seed()
sequence_length = 10000
para_model = pickle.load(open(os.getcwd() + '/pmm.pkl', 'rb'))
random_start_vector = np.array([np.random.random(),
                                np.random.random(),
                                np.random.random()])
strike_params = characterize_strikes(hb)
fish = FishModel(0, strike_params)
print('Creating Simulator')
sim = PreyCap_Simulation(
    fish,
    para_model,
    [np.array([1000, 1000, 1000]), random_start_vector],
    sequence_length)
sim.create_para_trajectory()
sim.run_simulation()
np.save('/Users/andrewbolton/para_simulation.npy', sim.para_xyz)
np.save('/Users/andrewbolton/origin_model.npy', sim.fish_xyz)
np.save('/Users/andrewbolton/uf_model.npy', sim.fish_bases)








# There are two functions in master: fishxyz_to_unitvecs and p_map_to_fish that will be useful for the model. You will get an initial XYZ coordinate of the para






# Make a paramecium model that is based on your data. This should be straightforward given how many paramecium records you have.
# Next, start the para at a location defined by your hunt initiation statistics. You will probably have to
# import para_master so you can run map para to heading on the fish.

