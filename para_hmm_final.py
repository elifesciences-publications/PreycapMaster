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

# takes fish specific velocities and addresses, while
# fitting model on all para from all fish


class ParaMarkovModel:
    def __init__(self, velocities, vec_addresses, apv, spacing, nn, *model):
        self.all_para_velocities = apv
        self.all_para_accelerations = np.diff(apv, axis=0)
        self.all_accel_mags = [
            [magvector(a) for a in accs]
            for accs in self.all_para_accelerations]
        self.vec_addresses = vec_addresses
        self.spacing = spacing
        self.non_nan_velocity_indices = nn
        self.velocity_mags = [
            [magvector(v) for v in vecs] for vecs in apv]
        self.accelerations = np.diff(velocities, axis=0)
        self.mag_limit = np.percentile(95, self.all_accel_mags)
        self.len_simulation = 500
        self.accels_filt = []
        if model != ():
            self.model = model
        else:
            self.model = []
        for ac, ac_m in zip(self.all_para_accelerations, self.all_accel_mags):
            if not (np.array(ac_m) > self.mag_limit).any():
                self.accels_filt.append(ac)

    def exporter(self):
        print('Saving Model')
        with open(os.getcwd() + '/pmm.pkl', 'wb') as file:
            pickle.dump(self, file)
                
    def fit_hmm(self):
        print('Fitting Model')
        s1 = pg.State(
            pg.MultivariateGaussianDistribution(
                np.array([0, 0, 0]), .1*np.eye(3)), name='0')
        s2 = pg.State(
            pg.MultivariateGaussianDistribution(
                np.array([0, 0, 0]), 2*np.eye(3)), name='1')
        model = pg.HiddenMarkovModel()
        model.add_states([s1, s2])
        model.add_transition(model.start, s1, .95)
        model.add_transition(model.start, s2, .05)
        model.add_transition(s1, s1, .95)
        model.add_transition(s1, s2, .05)
        model.add_transition(s2, s1, .2)
        model.add_transition(s2, s2, .8)
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
                noise = gen_samples[0][i]
                state = gen_samples[1][i+1].name
            else:
                state = states[i]
                if state == 0:
                    noise = sample_from_state_distribution(
                        mean_vec_s0, cov_mat_s0)
                elif state == 1:
                    noise = sample_from_state_distribution(
                        mean_vec_s1, cov_mat_s1)
            current_vel_vector += noise
            mag_current = magvector(current_vel_vector)
            if mag_current > vmax:
                current_vel_vector *= vmax / mag_current
            gen_states.append(int(state))

        if plot_or_not:
            stateplot_3d(para_x, para_y, para_z, gen_states)
        return para_x, para_y, para_z, gen_states, vmax

    def para_state_predictor(self, plotornot, directory, h_and_p):
        hunt_index, para_id = h_and_p
        print("Executing Prediction")
        p_3d = np.load(
            directory + '/para3D' + str(
                hunt_index).zfill(2) + '.npy')
        vec_address = [ind for ind, ad in enumerate(self.vec_addresses)
                       if ad.tolist() == [hunt_index, para_id]][0]
        states = self.model.predict(self.accelerations[vec_address])
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

    
def magvector(vector):
    mag = np.sqrt(np.dot(vector, vector))
    return mag


def stateplot_3d(para_x, para_y, para_z, states):
    states = np.concatenate([[x, x, x] for x in states])
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
    state_ax.set_ylim([-.5, 1.5])
    state_ax.set_xlim([0, len(para_x)])
    state_ax.set_title('PARA STATE')
    state_ax.set_aspect('auto', 'box-forced')
    def updater(num, plotlist):
        if num < 1:
            return plotlist
        px = para_x[num]
        py = para_y[num]
        pz = para_z[num]
        state_x = range(num)
        state_y = states[0:num]
        plotlist[0].set_data(px, py)
        plotlist[1].set_data(px, pz)
        plotlist[2].set_data(state_x, state_y)
        return plotlist
    p_xy_plot, = p_xy_ax.plot([], [], marker='.', ms=10)
    p_xz_plot, = p_xz_ax.plot([], [], marker='.', ms=10)
    state_plot, = state_ax.plot([], [], linewidth=1.0)
    p_list = [p_xy_plot, p_xz_plot, state_plot]
    line_ani = anim.FuncAnimation(
        fig,
        updater,
        len(para_x),
        fargs=[p_list],
        interval=3,
        repeat=True,
        blit=False)
#        line_ani.save('test.mp4')
    pl.show()

    
def smooth_states(len_filter, states):
    u_filter = 1.0 / len_filter * np.ones(len_filter)
    smoothed_states = [np.sum(u_filter * np.array(
        state_win)) for state_win in sliding_window(len_filter, states)]
    return smoothed_states


def concat_para_velocities(directories):
    all_vels = []
    for d in directories:
        directory = os.getcwd() + '/' + d
        v = np.load(directory + '/para_velocity_input.npy')
        all_vels += v.tolist()
    return all_vels


    #here you will have animations of the para of interest in xy, xz, and xyz, side by side with the state call from the model. model.predict(vecs[hunt_index]) is your posterior on the states given the training data. 

if __name__ == '__main__':

    directory = os.getcwd() + '/042318_6_orig_analysis'
    # save_model = False
    # np.random.seed()
    # # Vec Addresses are hunt and para within the hunt.
    # # Vecs are velocity vectors in 3 frame windows
    # # 
    
    v = np.load('para_velocity_input.npy')
    va = np.load('para_vec_address.npy')
    # spacing = np.load('spacing.npy')
    # non_nans = np.load('no_nan_inds.npy')
    # pmm = ParaMarkovModel(v, va, spacing, non_nans)
    para_model = pickle.load(open(os.getcwd() + '/pmm.pkl', 'rb'))
#     pmm.fit_hmm()
#     if save_model:
#         pmm.exporter()
# #    pmm.para_state_predictor(6, 6, 2, directory)


# NOTE CURRENT MODEL IS DEVELOPED FROM 44 PARA TRAJECTORIES in
# 042318_6. This data is in 042318_6_orig_analysis.

# When you are finished with each fish, call pvec_wrapper to output all
# para trajectories from hunts

