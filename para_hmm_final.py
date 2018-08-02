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


class ParaMarkovModel:
    def __init__(self, velocities, vec_addresses, spacing, nn):
        self.vec_addresses = vec_addresses
        self.spacing = spacing
        self.non_nan_velocity_indices = nn
        self.velocity_mags = [
            [magvector(v) for v in vecs] for vecs in velocities]
        self.accelerations = [
            [v2 - v1 for v2, v1 in sliding_window(2, vecs)]
            for vecs in velocities]
        self.acc_mags = [
            [magvector(a) for a in accs] for accs in self.accelerations]
        self.sd_limit = 20
        self.len_simulation = 500
        self.accels_filt = []
        self.model = []
        for ac, ac_m in zip(self.accelerations, self.acc_mags):
            if not (np.array(ac_m) > self.sd_limit).any():
                self.accels_filt.append(ac)
            else:
                temp_vd = []
                for j, k in zip(ac, ac_m):
                    if k < self.sd_limit:
                        temp_vd.append(j)
                self.accels_filt.append(temp_vd)

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
    
    def generate_para(self, start_vector, start_position, lensim):

        def draw_vmax():
            five_sec_filter = 300 / self.spacing
            max_v_per_para = [np.mean(vm) + 2*np.nanstd(vm)
                              for vm in self.velocity_mags
                              if len(vm) > five_sec_filter]
            np.random.seed()
            max_ind = int(np.floor(np.random.random() * len(max_v_per_para)))
            return max_v_per_para[max_ind]
                               
        def tank_boundary(vals):
            vals = np.array(vals)
            if (vals < 0).any() or (vals > 1888).any():
                return True
            else:
                return False
            
        self.len_simulation = lensim
        # vmax = np.mean(
        #     np.concatenate(self.velocity_mags)) + 2 * np.std(
        #         np.concatenate(self.velocity_mags))
        vmax = draw_vmax()
        print("Max Velocity")
        print(vmax)
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
            noise = gen_samples[0][i]
            state = gen_samples[1][i+1].name
            current_vel_vector += noise
            mag_current = magvector(current_vel_vector)
            if mag_current > vmax:
                current_vel_vector *= vmax / mag_current
            gen_states.append(int(state))
        return para_x, para_y, para_z, gen_states, vmax

    def para_state_predictor(self, hunt_index, para_id, plotornot, directory):
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
#        return states

    
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




    #here you will have animations of the para of interest in xy, xz, and xyz, side by side with the state call from the model. model.predict(vecs[hunt_index]) is your posterior on the states given the training data. 

if __name__ == '__main__':

    directory = os.getcwd() + '/042318_6'
    save_model = False
    np.random.seed()
    v = np.load('input.npy')
    va = np.load('vec_address.npy')
    spacing = np.load('spacing.npy')
    non_nans = np.load('no_nan_inds.npy')
    pmm = ParaMarkovModel(v, va, spacing, non_nans)
    pmm.fit_hmm()
    if save_model:
        pmm.exporter()
    pmm.para_state_predictor(6, 6, 2, directory)
        
    # np.random.seed()
    # random_vec = v[int(np.random.random() * v.shape[0])][0]
    # print np.random.random()
    # random_pos = [944, 944, 944]
    # px, py, pz, states = pmm.generate_para(random_vec, random_pos)
    # stateplot_3d(px, py, pz, states)
    # predicted_states = pmm.para_state_predictor(115, 5, True)





    

   

# SD limit is critical. High sd diff vectors are required for making the paramecium tumble. However, there is a set of vectors that are unreasonably high (probably from mistakes in tracking). If you set this at 20, para look like para and enter tumbling states with a frequency that I believe. However, there is no empirical reason for choosing 20. What this is is a decaying exponential of diff mags that probably have outliers. Using SD is incorrect b/c the dist is not gaussian. What you want is an outlier filter for an exponentially decaying diff mag distribution. I only went with 20 because this looked like the true tail of the distribution. 

# NO MATTER WHAT THIS LIMIT PARA STILL GOES OUT OF CONTROL WITH VELOCITY AFTER INITIAL PROMISING BEHAVIOR!
# WHat restricting the sd_limit does definitely do is allow the model to simulate small amplitude vectors, which it DOES, but still goes out of controL!




# all_xdiffs = np.concatenate([np.array(vd)[:,0] for vd in vdiffs])
# all_ydiffs = np.concatenate([np.array(vd)[:,1] for vd in vdiffs])
# all_zdiffs = np.concatenate([np.array(vd)[:,2] for vd in vdiffs])








#state is going to be 3 times shorter than the xyz coords it describes.
# could also use sliding_Window instead and have a value for each xyzcoord.
# this seems like a good idea. use sliding window of size 5. cut off the last 5 coords of para xyz. also make sure that the nan stretches only happen at the end and beginnign of the para records. can then simply filter pxyz for nans, and do that when making the vectors as well. if you are going to model the noise, you have to take 6 off the para xyz recs. can take it off the beginning as well so that each vector is a velocity pointing to where the para will be in 5 frames. 



# KEY POINT. Para are probably not just drawing noise to the previous vector on each event. If they were, they'd get stuck in the maxima that are generated now. What happens is one in every 200 draws or so, the noise is large amplitude. When you add large amplitude noise to the velocity vector, the vector gets big. When this happens, you're in a feedback loop where the velocity oscilates around a central value for every vector BUT the bump. Can either add a damping factor or model the magnitude of each vector. 


# you can draw the magnitude difference on each as well. if you sample magnitude differences and model them, you can get a transition of velocity states too. 
