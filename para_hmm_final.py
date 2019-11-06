import pomegranate as pg
import pickle
import os
import numpy as np
import math
from toolz.itertoolz import sliding_window
from collections import Counter
import matplotlib.pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sb
import matplotlib.animation as anim
from scipy.ndimage import gaussian_filter, uniform_filter1d
from astropy.convolution import Gaussian1DKernel, convolve

# takes fish specific velocities and addresses, while
# fitting model on all para from all fish

# KEY -- Make sure you increase the spacing so a zero velocity is
# basically impossible if the para passes threshold of over 1.5.
# for your sanity, increase the velocity to 1.6 thresh in
# pvec_wrapper.

# think about whether this severely changes the way we think about
# inerpolation if the spacing gets too small. 


class ParaStateCaller:
    def __init__(self, directory, pmm):
        self.directory = directory
        self.velocities = np.load(directory + '/para_velocity_input.npy')
        self.vec_addresses = np.load(directory + '/para_vec_address.npy')
        self.spacing = np.load(directory + '/spacing.npy')
        self.non_nan_velocity_indices = np.load(directory + '/no_nan_inds.npy')
        self.model = pmm.model
        self.accelerations = [[divide_arrays(a, b) for a, b in sliding_window(
            2, v)] for v in self.velocities]
        
    def para_state_predictor(self, plotornot, h_and_p):
        hunt_index, para_id = h_and_p
        print("Executing Prediction")
        p_3d = np.load(
            self.directory + '/para3D' + str(
                hunt_index).zfill(2) + '.npy')
        vec_address = [ind for ind, ad in enumerate(self.vec_addresses)
                       if ad.tolist() == [hunt_index, para_id]][0]
        states = self.model.predict(self.accelerations[vec_address])
        print(states)
        p_bounds = [
            self.non_nan_velocity_indices[
                vec_address][0] * self.spacing,
            self.non_nan_velocity_indices[vec_address][-1] * self.spacing]
        print p_bounds[0], p_bounds[1]
        para_x = p_3d[para_id*3][p_bounds[0]:p_bounds[1]]
        print len(para_x)
        para_y = p_3d[(para_id*3) + 1][p_bounds[0]:p_bounds[1]]
        para_z = p_3d[(para_id*3) + 2][p_bounds[0]:p_bounds[1]]
        if plotornot:
            stateplot_3d(para_x, para_y, para_z, states)
        return states
    

class ParaMarkovModel:
    def __init__(self, drcts):
        self.all_para_velocities = concat_para_velocities(drcts)
        self.all_para_accelerations = [[divide_arrays(a, b)
                                        for a, b in sliding_window(
            2, v)] for v in self.all_para_velocities]
        self.all_accel_mags = [
            [magvector(a) for a in accs]
            for accs in self.all_para_accelerations]
        self.velocity_mags = [
            [magvector(v) for v in vecs] for vecs in self.all_para_velocities]
        self.mag_limit = np.percentile(
            np.concatenate(self.all_accel_mags), 99)
        self.len_simulation = 500
        self.accels_filt = []
        self.model = []
        for ac, ac_m in zip(self.all_para_accelerations, self.all_accel_mags):
            if not (np.array(ac_m) > self.mag_limit).any():
                self.accels_filt.append(ac)
            else:
                filt_acvec = []
                for ac_vec, ac_mag in zip(ac, ac_m):
                    if ac_mag < self.mag_limit:
                        filt_acvec.append(ac_vec)
                self.accels_filt.append(filt_acvec)

    def exporter(self):
        print('Saving Model')
        with open(os.getcwd() + '/pmm.pkl', 'wb') as file:
            pickle.dump(self, file)
                
    def fit_hmm(self):
        print('Fitting Model')
        s0 = pg.State(
            pg.MultivariateGaussianDistribution(
                np.array([1, 1, 1]), .1*np.eye(3)), name='0')
        s1 = pg.State(
            pg.MultivariateGaussianDistribution(
                np.array([1, 1, 1]), 3*np.eye(3)), name='1')
        s2 = pg.State(
            pg.MultivariateGaussianDistribution(
                np.array([.5, .5, .5]),
                .1*np.eye(3) + .1*np.ones([3, 3])), name='2')

        s3 = pg.State(
            pg.MultivariateGaussianDistribution(
                np.array([1.5, 1.5, 1.5]),
                .1*np.eye(3) + .1*np.ones([3, 3])), name='3')
        model = pg.HiddenMarkovModel()
        model.add_states([s0, s1, s2, s3])
        model.add_transition(model.start, s0, .85)
        model.add_transition(model.start, s1, .05)
        model.add_transition(model.start, s2, .05)
        model.add_transition(model.start, s3, .05)
        model.add_transition(s0, s0, .85)
        model.add_transition(s0, s1, .05)
        model.add_transition(s0, s2, .05)
        model.add_transition(s0, s3, .05)
        model.add_transition(s1, s0, .1)
        model.add_transition(s1, s1, .7)
        model.add_transition(s1, s2, .1)
        model.add_transition(s1, s3, .1)
        model.add_transition(s2, s0, .1)
        model.add_transition(s2, s1, .1)
        model.add_transition(s2, s2, .7)
        model.add_transition(s2, s3, .1)
        model.add_transition(s3, s0, .1)
        model.add_transition(s3, s1, .1)
        model.add_transition(s3, s2, .1)
        model.add_transition(s3, s3, .7)
        model.bake()
        model.fit(self.accels_filt)
        self.model = model
        
    def generate_model_para(self,
                            start_vector,
                            start_position, lensim,
                            vmax, plot_or_not, *states):

        mean_vec_s0 = np.array(self.model.states[0].distribution.parameters[0])
        cov_mat_s0 = np.array(self.model.states[0].distribution.parameters[1])
        mean_vec_s1 = np.array(self.model.states[1].distribution.parameters[0])
        cov_mat_s1 = np.array(self.model.states[1].distribution.parameters[1])
        mean_vec_s2 = np.array(self.model.states[2].distribution.parameters[0])
        cov_mat_s2 = np.array(self.model.states[2].distribution.parameters[1])
        mean_vec_s3 = np.array(self.model.states[3].distribution.parameters[0])
        cov_mat_s3 = np.array(self.model.states[3].distribution.parameters[1])

        def sample_from_state_distribution(mv, cm):
            acc_vector = np.random.multivariate_normal(mv, cm)
            return acc_vector

        def draw_vmax():
            max_v_per_para = [np.percentile(vm, 80)
                              for vm in self.velocity_mags]
            np.random.seed()
            max_ind = int(np.floor(np.random.random() * len(max_v_per_para)))
            return max_v_per_para[max_ind]
                               
        def tank_boundary(vals):
            vals = np.array(vals)
            if (vals < 0).any() or (vals > 1888).any():
                return True
            else:
                return False

        if vmax < 0:
            vmax = draw_vmax()
            print("Max Velocity: " + str(vmax))
        self.len_simulation = lensim
        gen_samples = self.model.sample(self.len_simulation, True)
        gen_states = []
        para_x = [start_position[0]]
        para_y = [start_position[1]]
        para_z = [start_position[2]]
        current_vel_vector = np.array(start_vector)
        for i in range(self.len_simulation):
            tank_edge = tank_boundary([para_x[-1], para_y[-1], para_z[-1]])
            if tank_edge:
                current_vel_vector *= -1
            px = para_x[-1] + current_vel_vector[0]
            py = para_y[-1] + current_vel_vector[1]
            pz = para_z[-1] + current_vel_vector[2]
            para_x.append(px)
            para_y.append(py)
            para_z.append(pz)
            if states == ():
                scale = gen_samples[0][i]
                state = gen_samples[1][i+1].name
            else:
                state = states[i]
                if state == 0:
                    scale = sample_from_state_distribution(
                        mean_vec_s0, cov_mat_s0)
                elif state == 1:
                    scale = sample_from_state_distribution(
                        mean_vec_s1, cov_mat_s1)
                elif state == 2:
                    scale = sample_from_state_distribution(
                        mean_vec_s2, cov_mat_s2)
                elif state == 3:
                    scale = sample_from_state_distribution(
                        mean_vec_s3, cov_mat_s3)

            current_vel_vector *= scale
            mag_current = magvector(current_vel_vector)
            if mag_current > vmax:
                current_vel_vector *= vmax / mag_current
            gen_states.append(int(state))

        if plot_or_not:
            stateplot_3d(para_x, para_y, para_z, gen_states)
        return para_x, para_y, para_z, gen_states, vmax

    
def magvector(vector):
    mag = np.sqrt(np.dot(vector, vector))
    return mag


def stateplot_3d(para_x, para_y, para_z, states):
    states = np.concatenate([[x, x, x] for x in states])
    sm_states = smooth_states(5, states)
    print "Length of Para Record"
    print len(para_x)
    fig = pl.figure(figsize=(12, 8))
    p_xy_ax = fig.add_subplot(131)
    p_xy_ax.set_title('XY COORDS')
    p_xy_ax.set_xlim([0, 1888])
    p_xy_ax.set_ylim([0, 1888])
    p_xz_ax = fig.add_subplot(132)
    p_xz_ax.set_title('XZ COORDS')
    p_xz_ax.set_xlim([0, 1888])
    p_xz_ax.set_ylim([0, 1888])
    p_xy_ax.set_aspect('equal')
    p_xz_ax.set_aspect('equal')
    state_ax = fig.add_subplot(133)
    state_ax.set_ylim([-.5, 3.5])
    state_ax.set_xlim([0, len(para_x)])
    state_ax.set_title('PARA STATE')
    state_ax.set_aspect('auto', 'box-forced')
    def updater(num, plotlist):
        if num < 1:
            return plotlist
        px = para_x[num]
        py = para_y[num]
        pz = para_z[num]
        px_hist = para_x[0:num]
        py_hist = para_y[0:num]
        pz_hist = para_z[0:num]
        state_x = range(num)
        state_y = states[0:num]
#        smoothed_state_y = sm_states[0:num]
        plotlist[2].set_data(px_hist, py_hist)
        plotlist[3].set_data(px_hist, pz_hist)
        plotlist[0].set_data(px, py)
        plotlist[1].set_data(px, pz)
        plotlist[4].set_data(state_x, state_y)
#        plotlist[3].set_data(state_x, smoothed_state_y)
        return plotlist
    p_xy_line, = p_xy_ax.plot([], [], linewidth=1, color='g')
    p_xz_line, = p_xz_ax.plot([], [], linewidth=1, color='g')
    p_xy_plot, = p_xy_ax.plot([], [], marker='.', ms=15, color='m')
    p_xz_plot, = p_xz_ax.plot([], [], marker='.', ms=15, color='m')
    state_plot, = state_ax.plot([], [], marker='.', linestyle='None', color='k')
  #  sm_state_plot, = state_ax.plot([], [], linewidth=1.0)
    p_list = [p_xy_plot, p_xz_plot, p_xy_line, p_xz_line, state_plot]
  # sm_state_plot]
    line_ani = anim.FuncAnimation(
        fig,
        updater,
        len(para_x),
        fargs=[p_list],
        interval=2,
        repeat=True,
        blit=True)
#        line_ani.save('test.mp4')
    pl.show()


def smooth_states(sd_filter, states):
    gkern = Gaussian1DKernel(sd_filter)
    smoothed = convolve(states, gkern)
    return smoothed


def concat_para_velocities(directories):
    all_vels = []
    for d in directories:
        directory = os.getcwd() + '/' + d
        v = np.load(directory + '/para_velocity_input.npy')
        all_vels += v.tolist()
    return all_vels


def divide_arrays(arr1, arr2):
    divid = arr2 / arr1
    if np.isfinite(divid).all():
        return divid
    else:
        new_arr = []
        for a1, a2 in zip(arr1, arr2):
            if a1 == 0 and a2 == 0:
                new_arr.append(1)
            elif a1 == 0:
                new_arr.append(1)
            else:
                new_arr.append(a2 / a1)
        return np.array(new_arr)

    
def plot_xyz_components(vels_or_accs, *raw_arr):
    x, y, z = zip(*vels_or_accs)
    x = gaussian_filter(x, 2)
    y = gaussian_filter(y, 2)
    z = gaussian_filter(z, 2)
    pl.plot(x)
    pl.plot(y)
    pl.plot(z)
    if raw_arr != ():
        raw_arr = raw_arr[0]
        lengths = [len(j) for j in raw_arr]
        cum_lengths = np.cumsum(lengths)
        pl.plot(cum_lengths, np.zeros(len(cum_lengths)), marker='.')
    pl.show()


# model.predict(vecs[hunt_index]) is your posterior on the states given the training data. 

if __name__ == '__main__':

#    directories_for_fitting = ['091418_1', '091418_2', '091418_3',
#                               '091418_4', '091418_6']
    directories_for_fitting = ['091418_5']
    new_model = True
    np.random.seed()
    # # Vec Addresses are hunt and para within the hunt.
    # # Vecs are velocity vectors in 3 frame windows
    if new_model:
        pmm = ParaMarkovModel(directories_for_fitting)
        pmm.fit_hmm()
        pmm.exporter()
    else:
        pmm = pickle.load(open(os.getcwd() + '/pmm.pkl', 'rb'))
    para_caller = ParaStateCaller('091418_5', pmm)
    

# TO DO:

# Make this program more general so that it can take in a probabilistic model of the para. Model could be created using bayesDB and entering acceleration vectors at each line. However, these shouldn't necessarily be correlated. Want to have the option of your explicit markov model and the probabilistic model.

# I think its a good idea to get rid of the current para data which lasts only 5 seconds or so and do large chunks where you can get longer trajectories over time. It's very easy to do -- you just get a paramaster object for a given duration. 

