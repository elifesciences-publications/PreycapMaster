import ast
import os
import cv2
import imageio
from matplotlib import pyplot as pl
import numpy as np
import pickle
import math
import scipy.ndimage
import toolz
from scipy.stats.stats import pearsonr
import copy
import matplotlib.animation as anim
from collections import deque
import seaborn
import matplotlib.gridspec as gridspec
import mpl_toolkits.mplot3d.axes3d as p3

#  TO DO. Try to comment out part of process_corrmat where it attempts to go from XY back to XZ. I think this became redundant once I fed the entire overlapping timestamp to the matrix instead of just the overlap.


class Para:
    def __init__(self, timestmp, coord):
        self.location = [coord]
        self.lastcoord = coord
        self.timestamp = timestmp
        self.color = [np.random.random(), np.random.random(), np.random.random(
        )]
        self.completed = False
        self.waitindex = 0
        # waits at max 4 frames before giving up on particular para.
        self.waitmax = 4
        # max distance between para in consecutive frames in xy or xz vector space.
        self.thresh = 20
        self.double = False

    def endrecord(self):
        # CHOPS OFF THE HANGING LEN(WAITMAX) PORTION WHERE IT COULNDT FIND PARTNER
        self.location = self.location[:-self.waitmax]
        self.completed = True

    def nearby(self, contlist):
        #first, search the list of contours found in the image to see if any of them are near the previous position of this Para.
        # pcoords = [[ind, crd] for ind, crd in enumerate(contlist)
        #            if np.sqrt(
        #                np.dot((self.lastcoord[0] - crd[0], self.lastcoord[
        #                    1] - crd[1]), (self.lastcoord[0] - crd[
        #                        0], self.lastcoord[1] - crd[1]))) < self.thresh]

        if len(contlist) == 0:
            pcoords = np.array([])
        else:
            cont_arr = np.array(contlist)
            lastc = np.reshape(self.lastcoord, (1, 2))
            distance_past_thresh = np.where(
                np.sqrt(
                    np.sum(
                        (cont_arr-lastc)*(cont_arr-lastc), axis=1)) < self.thresh)
            pcoords = cont_arr[distance_past_thresh]
        
#if there's nothing found, add 1 to the waitindex, and say current position is the last position

        if pcoords.shape[0] == 0:
            self.location.append(self.lastcoord)
            if self.waitindex == self.waitmax:
                #end record if you've gone 'waitmax' frames without finding anything. this value greatly changes things. its a delicate balance between losing the para and waiting too long while another para enters
                self.endrecord()
            self.waitindex += 1

# this case is only one contour is within the threshold distance to this Para.
        elif pcoords.shape[0] == 1:
            newcoord = pcoords[0]
            index_to_pop = distance_past_thresh[0][0]
            self.location.append(newcoord)
            self.lastcoord = newcoord
            self.waitindex = 0
            contlist.pop(index_to_pop)

# this case is that two or more contours fit threshold distance. stop the record and mark it as a double.
        elif pcoords.shape[0] > 1:            
            self.endrecord()
            self.double = True
        return contlist


class ParaMaster():
    def __init__(self, start_ind, end_ind, directory, pcw):
        self.pcw = pcw
        self.directory = directory
        self.decimated_vids = False
        self.startover = False
        self.all_xy = []
        self.all_xz = []
        self.corr_mat = []
        self.corr_mat_original = []
        self.xyzrecords = []
        self.framewindow = [start_ind, end_ind]
        self.para3Dcoords = []
        self.distance_thresh = 100
        self.length_thresh = 30
        self.time_thresh = 60
        self.filter_width = 5
        self.unpaired_xy = []
        self.unpaired_xz = []
        self.paravectors = []
        self.paravectors_normalized = []
        self.dots = []
        self.makemovies = False
        self.topframes = deque()
        self.sideframes = deque()
        self.topframes_original = deque()
        self.sideframes_original = deque()
        self.dotprod = []
        self.velocity_mags = []
        self.length_map = np.vectorize(lambda a: a.shape[0])
        self.visited_xy = []
        self.visited_xz = []
        self.long_xy = []
        self.long_xz = []
        self.interp_indices = []
        self.min_coeff = .9
        
    def exporter(self):
        print('exporting ParaMaster')
        with open('paradata.pkl', 'wb') as file:
            pickle.dump(self, file)

# This function fits contours to each frame of the high contrast videos created in flparse2. Contours are filtered for proper paramecia size, then the locations of contours are compared to the current locations of paramecia. Locations of contours falling within a distance threshold of a known paramecia are appended to the location list for that para. At the end, individual paramecia records are established for each paramecia in the tank. Each para object is stored in the all_xy or all_xz para object list depending on plane.

    def findpara(self, params, record_para, *ask):
        params_top, params_side, area = params
        if ask != ():
            dec_vid = raw_input('Decimate Video?: ')
            if dec_vid == 'y':
                pms = raw_input('Enter Decimation Params: ')
                params = ast.literal_eval(pms)
                params_top, params_side, area = params
            else:
                params_top = []
                params_side = []
                area = 6
                record_para = True
        if params_top != [] or params_side != []:
            self.decimated_vids = True
            
        def repeat_vids():
            self.watch_event(1)
            self.watch_event(2)
            ra = raw_input('Repeat, Yes, or No: ')
            if ra == 'r':
                return repeat_vids()
            elif ra == 'y':
                return 'y'
            else:
                return 'n'

        def contrast_frame(vid, br, frameid, prms):
            frame = vid.get_data(frameid)
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brsub = brsub_img(grayframe, br)
            cont_ = imcont(brsub, prms).astype(np.uint8)
            return cont_

        directory = self.directory
        top_br = np.load(directory + 'backgrounds_top.npy').astype(np.uint8)
        side_br = np.load(directory + 'backgrounds_side.npy').astype(np.uint8)
        pcw = self.pcw
        p_t = []
        p_s = []
        completed_t = []
        completed_s = []
        if params_top == []:
            top_file = directory + 'top_contrasted.AVI'
        else:
            top_file = '/Volumes/WIKmovies/' + directory[-9:-1] + '_cam0.AVI'
        if params_side == []:
            side_file = directory + 'side_contrasted.AVI'
        else:
            side_file = '/Volumes/WIKmovies/' + directory[-9:-1] + '_cam1.AVI'
        paravid_top = imageio.get_reader(top_file, 'ffmpeg')
        paravid_side = imageio.get_reader(side_file, 'ffmpeg')
        firstframe = True
        for framenum, go_to_frame in enumerate(
                range(self.framewindow[0],
                      self.framewindow[1], 1)):
            if framenum % 100 == 0:
                print go_to_frame
            if not record_para:
                if framenum >= pcw:
                    if params_top != []:
                        cont_top = contrast_frame(paravid_top,
                                                  top_br[go_to_frame / 1875],
                                                  go_to_frame,
                                                  params_top)
                        self.topframes.append(cont_top)
                    if params_side != []:
                        cont_side = contrast_frame(paravid_side,
                                                   side_br[go_to_frame / 1875],
                                                   go_to_frame,
                                                   params_side)
                        self.sideframes.append(cont_side)
                    continue
                else:
                    continue

            if params_top != []:
                cont_top = contrast_frame(paravid_top,
                                          top_br[go_to_frame / 1875],
                                          go_to_frame,
                                          params_top)
                cont_color_top = cv2.cvtColor(cont_top, cv2.COLOR_GRAY2RGB)

            else:
                cont_color_top = paravid_top.get_data(go_to_frame)
                cont_top = cv2.cvtColor(
                    cont_color_top, cv2.COLOR_BGR2GRAY)
                r, cont_top = cv2.threshold(cont_top, 120, 255,
                                            cv2.THRESH_BINARY)
            if params_side != []:
                cont_side = contrast_frame(paravid_side,
                                           side_br[go_to_frame / 1875],
                                           go_to_frame,
                                           params_side)
                cont_color_side = cv2.cvtColor(cont_side, cv2.COLOR_GRAY2RGB)

            else:
                cont_color_side = paravid_side.get_data(go_to_frame)
                cont_side = cv2.cvtColor(
                    cont_color_side, cv2.COLOR_BGR2GRAY)
                r, cont_side = cv2.threshold(cont_side, 120, 255,
                                             cv2.THRESH_BINARY)
            rim, contours_t, hierarchy = cv2.findContours(
                cont_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rim, contours_s, hierarchy = cv2.findContours(
                cont_side, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            parafilter_top = [cv2.minEnclosingCircle(t)[0] for t in contours_t
                              if area <= cv2.contourArea(t) < 1000]
            parafilter_side = [cv2.minEnclosingCircle(s)[0] for s in contours_s
                               if area <= cv2.contourArea(s) < 1000]
            for para in parafilter_top:
                cv2.circle(cont_color_top, (int(para[0]), int(para[1])), 3,
                           (0, 0, 255), -1)
            for para in parafilter_side:
                cv2.circle(cont_color_side, (int(para[0]), int(para[1])), 3,
                           (0, 0, 255), -1)
            if self.makemovies:
                if framenum >= pcw:
                    self.topframes.append(cont_color_top)
                    self.sideframes.append(cont_color_side)

            if firstframe:
                firstframe = False
                p_t = [Para(framenum, pr) for pr in parafilter_top]
                p_s = [Para(framenum, pr2) for pr2 in parafilter_side]

# p_t is a list of para objects. asks if any elements of contour list are nearby each para p.
            else:
                xylist = map(lambda n: n.nearby(parafilter_top), p_t)
                xzlist = map(lambda n: n.nearby(parafilter_side), p_s)
                newpara_t = [Para(framenum, cord) for cord in parafilter_top]
                newpara_s = [Para(framenum, cord_s)
                             for cord_s in parafilter_side]
                p_t = p_t + newpara_t
                p_s = p_s + newpara_s
                xy_complete = filter(lambda x: x.completed, p_t)
                xz_complete = filter(lambda z: z.completed, p_s)
                completed_t = completed_t + xy_complete
                completed_s = completed_s + xz_complete
                p_t = filter(lambda x: not x.completed, p_t)
#current para list p_t and p_s are cleansed of records that are complete.
                p_s = filter(lambda z: not z.completed, p_s)

        self.watch_event(1)
        self.watch_event(2)
        modify = raw_input('Modify Videos?: ')
        if modify == 'r':
            modify = repeat_vids()
        if modify == 'q':
            self.startover = True
            return
        if modify == 'y':
            action = raw_input('Enter new params: ')
            self.topframes = deque()
            self.sideframes = deque()
            paravid_top.close()
            paravid_side.close()
            return self.findpara(ast.literal_eval(action), False)
        else:
            if not record_para:
                self.topframes = deque()
                self.sideframes = deque()
                paravid_top.close()
                paravid_side.close()
                return self.findpara(params, True)
        if record_para:
            all_xy = completed_t + p_t
            all_xz = completed_s + p_s
            all_xy = sorted(all_xy, key=lambda x: len(x.location))
            all_xy.reverse()
            all_xz = sorted(all_xz, key=lambda z: len(z.location))
            all_xz.reverse()
            self.all_xy = [para for para in all_xy
                           if len(para.location) > 0 and
                           para.timestamp + len(para.location) > self.pcw]
            self.all_xz = [para for para in all_xz
                           if len(para.location) > 0 and
                           para.timestamp + len(para.location) > self.pcw]
            self.long_xy = [para for para in self.all_xy
                            if len(para.location) >= self.length_thresh]
            self.long_xz = [para for para in self.all_xz
                            if len(para.location) >= self.length_thresh]

            print('Para Found in XY')
            print len(self.all_xy)
            print('Para Found in XZ')
            print len(self.all_xz)
            self.topframes_original = copy.deepcopy(self.topframes)
            self.sideframes_original = copy.deepcopy(self.sideframes)
            paravid_top.close()
            paravid_side.close()



    #This function is the most important for paramecia matching in both planes. First, an empty correlation matrix is created with len(all_xy) columns and len(all_xz) rows. The datatype is a 3-element tuple that takes a pearson coefficient, a pvalue, and a time of overlap for each para object in all_xy vs all_xz. I initialize the corrmat with 0,1 and [] for each element.

    def clear_frames(self):
        self.topframes = copy.deepcopy(self.topframes_original)
        self.sideframes = copy.deepcopy(self.sideframes_original)
        
    def makecorrmat(self):
        print('yo my brotha')
        self.corr_mat = np.zeros(
            (len(self.long_xz), len(self.long_xy)),
            dtype=[('coeff', np.float64), ('pval', np.float64), ('time',
                                                                 np.ndarray)])
        # row (m) and column (n) index for corrmat. loop through all rows of each column
        m = 0
        n = 0
        listinit = np.array(
            [(0, 1, np.array([])) for ind in self.long_xy],
            dtype=[('coeff', np.float64), ('pval', np.float64), ('time',
                                                                 np.ndarray)])
        for j in range(self.corr_mat.shape[0]):
            self.corr_mat[j] = listinit
# start correlation matrix with all 0 for pearson coeff and 1 for pval.
        for px in self.long_xy:
            # first step is to iterate through xy Para objects. compare timestamps and locations to all xz Para objects
            timerange_xy = range(px.timestamp, px.timestamp + len(px.location))
            # timestamp is frame number in top_ir where record begins.
            x_location = [x[0] for x in px.location]
            m = 0  # resets m after a full runthrough
            for pz in self.long_xz:
                timerange_xz = range(pz.timestamp,
                                     pz.timestamp + len(pz.location))
                x2_location = [x2[0] for x2 in pz.location]
                #next check if timestamps from xy and xz overlap. if they do to any extent, compare the region of overlap using correlation coefficient.
                if timerange_xy[0] in timerange_xz and timerange_xy[
                        -1] in timerange_xz:
                    firstind = timerange_xz.index(timerange_xy[0])
#this is index of timerange where it is equal to timerange_xy[0]
                    lastind = timerange_xz.index(timerange_xy[-1])
                    corrwindow_xy = x_location
                    corrwindow_xz = x2_location[
                        firstind:lastind + 1]
#indicies are noninclusive. add 1.
                    timeoverlap = np.array(timerange_xy)

                elif timerange_xy[0] in timerange_xz:
                    firstind = timerange_xz.index(timerange_xy[0])
                    lastind = timerange_xy.index(timerange_xz[-1])
                    corrwindow_xy = x_location[:lastind + 1]
                    corrwindow_xz = x2_location[firstind:]
                    #+1 for inclusive.
                    timeoverlap = np.arange(timerange_xy[0],
                                            timerange_xz[-1] + 1)

                elif timerange_xy[-1] in timerange_xz:
                    firstind = timerange_xy.index(timerange_xz[0])
                    lastind = timerange_xz.index(timerange_xy[-1])
                    corrwindow_xy = x_location[firstind:]
                    corrwindow_xz = x2_location[:lastind + 1]
                    #+1 for inclusive
                    timeoverlap = np.arange(timerange_xz[0],
                                            timerange_xy[-1] + 1)

                elif timerange_xz[-1] in timerange_xy and timerange_xz[
                        0] in timerange_xy:
                    firstind = timerange_xy.index(timerange_xz[0])
                    lastind = timerange_xy.index(timerange_xz[-1])
                    corrwindow_xy = x_location[firstind:lastind + 1]
                    corrwindow_xz = x2_location
                    timeoverlap = np.array(timerange_xz)

                else:
                    m += 1
                    continue

                xystep, xzstep = (10, 10) if len(corrwindow_xy) >= 10 else (1,
                                                                            1)
                xyavgpos = np.mean(corrwindow_xy[0:-1:xystep])
                xzavgpos = np.mean(corrwindow_xz[0:-1:xzstep])

                # correlation will find relationships between mutual dynamics of growth and decay of both xy and xz. but also have to make sure they grow and decay on the same scale (i.e. when xy increases by 5, xz also increases by 5). to check this, compare total pathlength. also want to make sure that the actual x positions are close to each other, so that the dynamics aren't just similar, the relative positions are as well.

#this says paras have to be within 100 pixels of each other in x.
                if abs(xyavgpos - xzavgpos) > 200:
                    m += 1
                    continue
#eliminate  analysis of really small overlaps that would be more likely to correlate.
                elif len(corrwindow_xy) < 30:
                    m += 1
                    continue

#eliminate records where avg value of location indicates a reflection at the edge of the tank, which of course correlates to the real para.

                elif xyavgpos < 50 or xyavgpos > 1858 or xzavgpos < 50 or xzavgpos > 1858:
                    m += 1
                    continue


# the sampling rate is sufficiently high that small bits of noise can throw off correlation. filter the records first. this level of filtering is perfect.

                else:
                    filteredcorr_xy = scipy.ndimage.filters.gaussian_filter(
                        corrwindow_xy, self.filter_width)
                    filteredcorr_xz = scipy.ndimage.filters.gaussian_filter(
                        corrwindow_xz, self.filter_width)
                    xypathgen = toolz.itertoolz.sliding_window(2,
                                                               filteredcorr_xy)
                    xzpathgen = toolz.itertoolz.sliding_window(2,
                                                               filteredcorr_xz)
                    xypathlength = sum([abs(b - a) for a, b in xypathgen])
#pathlength, not just displacement.
                    xzpathlength = sum([abs(d - c) for c, d in xzpathgen])
                    if xypathlength != 0 and xzpathlength != 0:
                        if float(min(xypathlength, xzpathlength)) / float(
                                max(xypathlength, xzpathlength)) < .8:
                            m += 1
                            continue

                coeff, pval = pearsonr(filteredcorr_xy, filteredcorr_xz)
                if not np.isnan(coeff):
                    if coeff > 0:
                        self.corr_mat[m, n] = (coeff, pval, timeoverlap)
                m += 1
            if n % 100 == 0:
                print(n)
            n += 1

    def makexyzrecords(self):

        #calculates magnitude of a vector based on 2 2D coords.
        def magvector(coord1, coord2):
            vec = [coord2[0] - coord1[0], coord2[1] - coord1[1]]
            mag = np.sqrt(np.dot(vec, vec))
            return mag

    #this function gets rid of all correlations that could contaminate future pairings after a pair has been established. it does this by checking if any correlation exists between a known paired record and any other records in the same timewindow. since the algorithm is based on proportion of total correlation in a given timewindow, this step is critical.

        def corrmat_reduce(corrmat, pair):

            def time_map(arr):
                f = np.vectorize(lambda a: np.intersect1d(arr, a).any())
                return f

            corrmat_copy = np.copy(corrmat)
            xy, xz = pair
            tau = copy.deepcopy(corrmat[xz, xy]['time'])
            map_func = time_map(tau)
            xz_overlap = np.where(map_func(corrmat_copy[xz]['time']))
            xy_overlap = np.where(map_func(corrmat_copy[:, xy]['time']))
            corrmat_copy[xz][xz_overlap] = (0, 1, np.array([]))
            corrmat_copy[:, xy][xy_overlap] = (0, 1, np.array([]))
            return corrmat_copy

# Each XYZrecord is a set of pairs ordered in time. Between correlation matching, this function checks if any XYZ records should be matched to another XYZ record to create a new longer record (i.e. the first or last pair in a record is sufficiently close in time and distance to the first or last pair in another XYZ record). Can think of this as a double stranded break being mended.

        def matchxyzinternal(recordlist, xypara, xzpara):
            for i in range(len(recordlist)):
                base = recordlist.pop()
                if not base:
                    recordlist.insert(0, base)
                    continue
# grabs the first and last pair from the base XYZ record.
                firstpair_base, lastpair_base = base[0], base[-1]
# initial and final positions are obtained for xy and xz at the beginning and end of the XYZ record.
                base_xyinit_time, base_xyend_time = xypara[firstpair_base[
                    0]].timestamp, xypara[lastpair_base[0]].timestamp + len(
                        xypara[lastpair_base[0]].location)
                base_xyinit_position, base_xyend_position = xypara[
                    firstpair_base[0]].location[0], xypara[lastpair_base[
                        0]].location[-1]
                base_xzinit_time, base_xzend_time = xzpara[firstpair_base[
                    1]].timestamp, xzpara[lastpair_base[1]].timestamp + len(
                        xzpara[lastpair_base[1]].location)
                base_xzinit_position, base_xzend_position = xzpara[
                    firstpair_base[1]].location[0], xzpara[lastpair_base[
                        1]].location[-1]

# create two lists: one that takes possible matches that are in front of the base XYZ record and one that takes possible matches that are behind the base XYZ record in time. index the comp records so you can delete any matches after appending to the base.
                temp_forward_matches = []
                temp_backward_matches = []
                for ind, comp in enumerate(recordlist):
                    if not comp:
                        continue
# repeat grabbing of initial time and positions and end time and position for each comp XYZ record.
                    firstpair_comp, lastpair_comp = comp[0], comp[-1]
                    comp_xyinit_time, comp_xyend_time = xypara[firstpair_comp[
                        0]].timestamp, xypara[lastpair_comp[
                            0]].timestamp + len(xypara[lastpair_comp[
                                0]].location)
                    comp_xyinit_position, comp_xyend_position = xypara[
                        firstpair_comp[0]].location[0], xypara[lastpair_comp[
                            0]].location[-1]
                    comp_xzinit_time, comp_xzend_time = xzpara[firstpair_comp[
                        1]].timestamp, xzpara[lastpair_comp[
                            1]].timestamp + len(xzpara[lastpair_comp[
                                1]].location)
                    comp_xzinit_position, comp_xzend_position = xzpara[
                        firstpair_comp[1]].location[0], xzpara[lastpair_comp[
                            1]].location[-1]
# now ask if the first or last pairs in the comp record are closeby to the base record's first or last pairs in time and position.
                    if 0 < comp_xyinit_time - base_xyend_time < self.time_thresh and 0 < comp_xzinit_time - base_xzend_time < self.time_thresh:
                        if magvector(
                                comp_xyinit_position,
                                base_xyend_position) < self.distance_thresh and magvector(
                                    comp_xzinit_position,
                                    base_xzend_position) < self.distance_thresh:
                            temp_forward_matches.append((ind, comp))
                    elif 0 < base_xyinit_time - comp_xyend_time < self.time_thresh and 0 < base_xzinit_time - comp_xzend_time < self.time_thresh:
                        if magvector(
                                comp_xyend_position,
                                base_xyinit_position) < self.distance_thresh and magvector(
                                    comp_xzend_position,
                                    base_xzinit_position) < self.distance_thresh:
                            temp_backward_matches.append((ind, comp))
                    elif ind == len(recordlist) - 1:
                        pass
#if only a unique match was found, add entire record to the front or back of the base. reinsert the base into the list. delete the comp record.
                if len(temp_forward_matches) == 1:
                    base = base + temp_forward_matches[0][1]
                    ind_to_delete = temp_forward_matches[0][0]
                    recordlist[ind_to_delete] = []
                if len(temp_backward_matches) == 1:
                    base = temp_backward_matches[0][1] + base
                    ind_to_delete = temp_backward_matches[0][0]
                    recordlist[ind_to_delete] == []
                recordlist.insert(0, base)
            recordlist_emptyfilteredout = [rec for rec in recordlist if rec]
            return recordlist_emptyfilteredout

#This is the distance / time based part of the algorithm. Known pairs are represented in this algorithm to allow extension of known XYZ records based on only unpaired XY and XZ objects. Internal matching is accomplished by matchxyzinternal.

#First and last pair in each XYZ record are compared to

        def pair_by_distance(xypara, xzpara, recordlist, corrmat):
            pairlist = []
            #this is simply a list of all your current pairs used to rule out already paired Para Objects.
            for xyz_record in recordlist:
                for pair in xyz_record:
                    pairlist.append([pair[0], pair[1]])
            recordlist = matchxyzinternal(recordlist, xypara, xzpara)

            #Searches backwards for nearby records in time and space
            def searchbackwardmatch(known_rec, paralist, knownpairs):
                candidate_list = []
                known_time = paralist[known_rec].timestamp
                for ind, para in [(i, p) for i, p in enumerate(paralist)
                                  if i not in knownpairs]:
                    comp_time = para.timestamp + len(para.location)
                    if 0 < known_time - comp_time < self.time_thresh:
                        magdiff = magvector(paralist[known_rec].location[0],
                                            para.location[-1])
                        if magdiff < self.distance_thresh:
                            candidate_list.append([known_rec, ind])
                return candidate_list

#Searches forward for nearby records in time and space

            def searchforwardmatch(known_rec, paralist, knownpairs):
                candidate_list = []
                known_time = paralist[known_rec].timestamp + len(paralist[
                    known_rec].location)
                for ind, para in [(i, p) for i, p in enumerate(paralist)
                                  if i not in knownpairs]:
                    comp_time = para.timestamp
                    if 0 < comp_time - known_time < self.time_thresh:
                        magdiff = magvector(paralist[known_rec].location[-1],
                                            para.location[0])
                        if magdiff < self.distance_thresh:
                            candidate_list.append([known_rec, ind])
                return candidate_list

#DIRECT PAIRS are pairs that arise from an example like this: Pair is [xy1,xz1]. xy3 is nearby xy1 in space and time. xy3 correlates to xz1 greater than .9.

            def find_direct_pairs(candlist_xy, candlist_xz, pair, cmat):
                directpairlist = []
                for cand_xy in candlist_xy:
                    xz_corr = cmat[pair[1], cand_xy[1]]['coeff']
                    if xz_corr > .9:
                        directpair = [cand_xy[1], pair[1]]
                        directpairlist.append(directpair)
                        cmat = corrmat_reduce(cmat,
                                              [directpair[0], directpair[1]])
                for cand_xz in candlist_xz:
                    xy_corr = cmat[cand_xz[1], pair[0]]['coeff']
                    if xy_corr > .9:
                        directpair = [pair[0], cand_xz[1]]
                        directpairlist.append(directpair)
                        cmat = corrmat_reduce(cmat,
                                              [directpair[0], directpair[1]])
                return directpairlist, cmat

#NEIGHBORS are pairs that arise from a mutual extension of the same [xy,xz] pair from the end or beginning of an XYZ record. Eg. [xy1,xz1] are a pair from a known XYZ record. xy3 is nearby xy1 in space and time, xz3 is nearby xz1 in space and time. xy3 and xz3 strongly correlate with eachother.

            def find_neighbors(candlist_xy, candlist_xz, pair, cmat):
                neighborlist = []
                neighborlist_unique = []
                for cand_xy in candlist_xy:
                    coefflist = [rec['coeff'] for rec in cmat[:, cand_xy[1]]]
                    xz_possiblepartner = np.argmax(np.array(coefflist))
                    if coefflist[
                            xz_possiblepartner] > .9 and xz_possiblepartner in [
                                cnd[1] for cnd in candlist_xz
                            ]:
                        newpair = [cand_xy[1], xz_possiblepartner, pair[0],
                                   'xyseed']
                        neighborlist.append(newpair)
                for cand_xz in candlist_xz:
                    coefflist = [rec['coeff'] for rec in cmat[cand_xz[1]]]
                    xy_possiblepartner = np.argmax(np.array(coefflist))
                    if coefflist[
                            xy_possiblepartner] > .9 and xy_possiblepartner in [
                                cnd[1] for cnd in candlist_xy
                            ]:
                        newpair = [xy_possiblepartner, cand_xz[1], pair[1],
                                   'xzseed']
                        neighborlist.append(newpair)
                for uniquepair in neighborlist:
                    if uniquepair not in neighborlist_unique:
                        neighborlist_unique.append(uniquepair)
                return neighborlist_unique



            # Create candidate lists of possible nearby XY and XZ records to known ends of XYZ records. Items in candidate lists represent para objects that satisfy both a time and 2D distance threshold from ends of XYZ records.

            for i in range(len(recordlist)):
                current_record = recordlist.pop()
                firstpair, lastpair = current_record[0], current_record[-1]
                candlist_xy_backward = searchbackwardmatch(
                    firstpair[0], xypara, [x[0] for x in pairlist])
                candlist_xz_backward = searchbackwardmatch(
                    firstpair[1], xzpara, [z[1] for z in pairlist])
                candlist_xy_forward = searchforwardmatch(
                    lastpair[0], xypara, [x[0] for x in pairlist])
                candlist_xz_forward = searchforwardmatch(
                    lastpair[1], xzpara, [z[1] for z in pairlist])

                #Candidate lists are now lists of [known pair xy record or xz record, candidate continuation in same plane]
                #now use candidate information to derive direct pairs and neighbors.

                directpairs_forward, corrmat = find_direct_pairs(
                    candlist_xy_forward, candlist_xz_forward, lastpair,
                    corrmat)
                directpairs_backward, corrmat = find_direct_pairs(
                    candlist_xy_backward, candlist_xz_backward, firstpair,
                    corrmat)
                #next if found no direct pairs in forward, search for neighbors forward. if found no direct pairs backward, look for neighbors backward.
                neighbors_forward = []
                neighbors_backward = []
                if not directpairs_forward:
                    neighbors_forward = find_neighbors(
                        candlist_xy_forward, candlist_xz_forward, lastpair,
                        corrmat)
                    if len(neighbors_forward) == 1:
                        neighbor_pair = [neighbors_forward[0][0],
                                         neighbors_forward[0][1]]
                        corrmat = corrmat_reduce(corrmat, neighbor_pair)
                    elif len(neighbors_forward) > 1:
                        neighbors_forward = []
                if not directpairs_backward:
                    neighbors_backward = find_neighbors(
                        candlist_xy_backward, candlist_xz_backward, firstpair,
                        corrmat)
                    if len(neighbors_backward) == 1:
                        neighbor_pair = [neighbors_backward[0][0],
                                         neighbors_backward[0][1]]
                        corrmat = corrmat_reduce(corrmat, neighbor_pair)
                    elif len(neighbors_backward) > 1:
                        neighbors_backward = []

            # if len neighbors forward and backward are greater than one, throw out b/c don't know if its a continuation of the current record. should eventually get picked up in the garbage collection if a real hit.

                final_pairs_to_add = directpairs_forward + directpairs_backward + neighbors_forward + neighbors_backward
                current_record = current_record + final_pairs_to_add
                for p in final_pairs_to_add:
                    pairlist.append([p[0], p[1]])
                recordlist.insert(0, current_record)

            return recordlist, pairlist, corrmat

    #This is the recursive correlation-based part of the algorithm. Process_corrmat iterates until it finds matches at multiple levels of stringency.

        def process_corrmat(corr_mat):

            #Findmatch takes an entire row of correlations to a single xz record (i.e. a row of the correlation matrix). Parameters variable is a set of coeff,pvalue,timewindow of overlap to each xy record (i.e. cols of correlation mat)

            def findmatch(record, length_thresh, coeff_sum_threshold,
                          coeff_threshold):

                def time_map(arr):
                    f = np.vectorize(lambda a: np.intersect1d(arr, a).any())
                    return f
                
                pass_inds = np.intersect1d(
                    np.where(record['coeff'] > coeff_threshold),
                    np.where(self.length_map(record['time']) > length_thresh))
                nonzero_inds = np.where(record['coeff'] > 0)
                
# Want indices where current pass ind overlaps with the timestamp of other inds
                possible_pairs = []
                for pi in pass_inds:
                    map_func = time_map(record[pi]['time'])
                    coeff_sum = np.sum(
                        record[nonzero_inds][np.where(
                            map_func(record[nonzero_inds]['time']))]['coeff'])
                    if record[pi]['coeff'] / coeff_sum > coeff_sum_threshold:
                        possible_pairs.append(pi)
                if len(possible_pairs) != 0:
                    return possible_pairs
                else:
                    return []
                
            def findhit(corrmat,
                        l_thresh, coeff_sum_thresh, coeff_thresh,
                        cands, xz, orig_query):
             
                def pair_overlap(o_query, hit_cands):
                    to_pair = np.intersect1d(o_query, hit_cands)
                    new_cnds = np.delete(hit_cands, to_pair)
                    return to_pair.any(), new_cnds
                
                if np.unique(
                        self.visited_xy).shape[0] == corrmat.shape[1] or np.unique(
                            self.visited_xz).shape[0] == corrmat.shape[0]:
                    return []
                pairs = []
                for ind in cands:
                    if xz:
                        if ind in self.visited_xz:
                            continue
                        else:
                            self.visited_xz.append(ind)
                        rec = corrmat[ind]
                    elif not xz:
                        if ind in self.visited_xy:
                            continue
                        else:
                            self.visited_xy.append(ind)
                        rec = corrmat[:, ind]
                    if not (rec['coeff'] >= coeff_thresh).any():
                        continue
                    if not (self.length_map(rec['time']) > l_thresh).any():
                        continue
                    hits = findmatch(rec, l_thresh,
                                     coeff_sum_thresh, coeff_thresh)
                    should_i_pair, rest = pair_overlap(orig_query, hits)
                    if should_i_pair:
                        if xz:
                            pairs.append([orig_query[0], ind])
                        else:
                            pairs.append([ind, orig_query[0]])
                    if len(rest) != 0:
                        return pairs + findhit(corrmat,
                                               l_thresh, coeff_sum_thresh,
                                               coeff_thresh,
                                               rest, not xz, [ind])
                    else:
                        if not xz:
                            
                            return pairs + findhit(corrmat,
                                                   l_thresh, coeff_sum_thresh,
                                                   coeff_thresh,
                                                   range(corrmat.shape[0]),
                                                   not xz, [])
                        else:

                            return pairs + findhit(corrmat,
                                                   l_thresh, coeff_sum_thresh,
                                                   coeff_thresh,
                                                   range(corrmat.shape[1]),
                                                   not xz, [])

                return []
                    
                        
# minimum duration starts at 5 seconds.
# keep looping until you don't get any more records. then drop length threshold.
            min_duration = 500
            min_coeff_sum = .9
            min_coeff = self.min_coeff
            parapairs = []
            while True:
                if min_coeff_sum < .4:
                    break
                temp_pairs = findhit(corr_mat,
                                     min_duration,
                                     min_coeff_sum,
                                     min_coeff,
                                     range(corr_mat.shape[0]),
                                     True,
                                     [])
                if len(temp_pairs) != 0:
                    for tp in temp_pairs:
                        corr_mat = corrmat_reduce(corr_mat, tp)
                        parapairs.append(tp + [round(min_coeff_sum, 2), round(
                            min_coeff, 2), min_duration])
                else:
                    min_duration -= 50
                    if min_duration < 50:
                        min_duration = 500
                        min_coeff_sum -= .2
                    self.visited_xy = []
                    self.visited_xz = []
            return parapairs, corr_mat

#This function takes in para pairs found in the distance and correlation based algorithms and creates XYZ records of multiple pairs per record. Very simply, it adds pairs to a record if the record already contains either of its XY or XZ para objects.

        def createxyzrecords(all_pairs, xyzrecs, xypara, xzpara):

            # initializes the first record with the first pair
            if not xyzrecs:
                final_xyz_records = [[all_pairs[0]]]
                pairstocycle = all_pairs[1:]
            else:
                final_xyz_records = xyzrecs
                pairstocycle = all_pairs
            if pairstocycle:
                for pair in pairstocycle:
                    num_recs_in_finalxyz = len(final_xyz_records)
                    #first run, cycle_number should be 0 only
                    for cycle_number in range(num_recs_in_finalxyz):
                        base = final_xyz_records.pop()
                        if pair[0] in [xyind[0] for xyind in base] or pair[
                                1] in [xzind[1] for xzind in base]:
                            base.append(pair)
                            final_xyz_records.insert(0, base)
                            break
                        elif len(pair) > 3 and pair[3] == 'xyseed' and pair[
                                2] in [xyind[0] for xyind in base]:
                            base.append(pair)
                            final_xyz_records.insert(0, base)
                            break
                        elif len(pair) > 3 and pair[3] == 'xzseed' and pair[
                                2] in [xzind[1] for xzind in base]:
                            base.append(pair)
                            final_xyz_records.insert(0, base)
                            break
#if you've gotten to the end of popping records with no match, insert the pair as a new record.
                        elif cycle_number == num_recs_in_finalxyz - 1:
                            final_xyz_records.insert(0, base)
                            final_xyz_records.insert(0, [pair])
                        else:
                            final_xyz_records.insert(
                                0,
                                base)  #just puts the record back if it had nothing for this trial

    #this orders the pairs in the records according to their timestamp, first to last.
            final_xyz_records_ordered = []
            for rec in final_xyz_records:
                rec = sorted(rec, key=lambda x: xypara[x[0]].timestamp)
                rec = sorted(rec, key=lambda x: xzpara[x[1]].timestamp)
                final_xyz_records_ordered.append(rec)
            return final_xyz_records_ordered

#This is the garbage collection portion of the algorithm. After everything possible has been paired by recursive correlation and distance based matching, go back and see if a very strict high correlation / long timerange combination can fish out additional matches. If it does, go back into recursion.

        def checkforhighcorr(corrmat, corr_thresh, length_thresh):
            pairlist = []
            filteredpairs = []
            for xzind, row in enumerate(corrmat):
                # for xyind, corr_variables in enumerate(row):
                #     if corr_variables['coeff'] > corr_thresh and len(
                #             corr_variables['time']) > length_thresh:
                #         xyxzpair = [xyind, xzind]
                #     #    print(xyxzpair)
                #         pairlist.append(xyxzpair)
                threshpass = np.where(
                    np.logical_and(
                        row['coeff'] > corr_thresh,
                        self.length_map(row['time']) > length_thresh))[0]
                if threshpass.shape[0] == 0:
                    continue
                pairlist += [[tp, xzind] for tp in threshpass]
                #now filter for overlap. choose one if its longer AND has higher correlation, else toss both.
                filteredpairs = []
                xyinds = [xy[0] for xy in pairlist]
                xzinds = [xz[1] for xz in pairlist]
                for pair in pairlist:
                    if xyinds.count(pair[0]) == 1 and xzinds.count(pair[
                            1]) == 1:
                        filteredpairs.append(pair)
                    else:
                        pair_timewindow = corrmat[pair[1], pair[0]]['time']
                        indices_of_pairs_to_compare = [
                            ind for ind, xy in enumerate(xyinds)
                            if xy == pair[0]
                        ] + [ind for ind, xz in enumerate(xzinds)
                             if xz == pair[1]]
                        pairs_to_compare = [pairlist[
                            ind] for ind in indices_of_pairs_to_compare
                                            if pairlist[ind]]
                        #now check each pair for overlap: if overlapping, make a new list to compare. if this returns just one pair, append to filteredpairs
                        overlap_in_time = [
                            comp_pair for comp_pair in pairs_to_compare
                            if corrmat[comp_pair[1], comp_pair[0]]['time'][
                                0] in pair_timewindow or pair_timewindow[0] in
                            corrmat[comp_pair[1], comp_pair[0]]['time']
                        ]
                        if len(overlap_in_time) == 1:
                            #i.e. if the pair is the only one that overlaps with itself
                            filteredpairs.append(pair)
                        else:
                            coeffs = [corrmat[x[1], x[0]]['coeff']
                                      for x in overlap_in_time]
                            lengths = [len(corrmat[x[1], x[0]]['time'])
                                       for x in overlap_in_time]
                            if np.argmax(coeffs) == np.argmax(lengths):
                                bestpair = overlap_in_time[np.argmax(coeffs)]
                            else:
                                continue
                            if bestpair == pair:
                                filteredpairs.append(pair)
                            else:
                                continue

            for finalpair in filteredpairs:
                corrmat = corrmat_reduce(corrmat, finalpair)
            return filteredpairs, corrmat

# THIS IS THE MAIN LINE OF THE MAKEXYZRECORDS METHOD. Recursively calls process_corrmat for correlation algorithm and pair_by_distance for distance algorithm. In between, XYZ records are consistently created using createxyzrecords and reordered and organized using matchxyzinternal. If correlation returns nothing, tries strict high correlation. The output is a list of XYZ records ordered in time.

        finalpairlist = []
        xyzrecords = []
        highcorr = .99
        #  highcorr_length = 100
        highcorr_length = 30
        while True:
            pairs, self.corr_mat = process_corrmat(self.corr_mat)
            if not pairs:
                pairs, self.corr_mat = checkforhighcorr(
                     self.corr_mat, highcorr, highcorr_length)
                if not pairs:
                    if highcorr > self.min_coeff * .75:
                        highcorr -= .045
                    # COMMENT THIS BACK IN IF IT GETS TOO LOOSEY GOOSEY
                        highcorr_length += 10
                        continue
                    else:
                        break
            xyzrecords = createxyzrecords(pairs, xyzrecords, self.long_xy,
                                          self.long_xz)
            recursion_depth = 1
            while True:
                length_pairlist_before_algorithm = np.copy(len(pairs))
                xyzrecords, pairs, self.corr_mat = pair_by_distance(
                    self.long_xy, self.long_xz, xyzrecords, self.corr_mat)
                xyzrecords = createxyzrecords([], xyzrecords, self.long_xy,
                                              self.long_xz)
                #this will reorder all pairs within records according to timestamp.
                if len(pairs) == length_pairlist_before_algorithm:
                    break
                recursion_depth += 1

        self.xyzrecords = sorted(xyzrecords, key=lambda (x): len(x))
        self.xyzrecords.reverse()
# Here is previous save location

    #This function plots any para objects that were missed by the complete running of the algorithm.        

    def find_misses(self, plotornot):
        xyzrecs = self.xyzrecords
        pcw = self.pcw
        length_thresh = 0
        xyrecs = range(len(self.all_xy))
        xzrecs = range(len(self.all_xz))
        all_xy_matched = []
        all_xz_matched = []
        for xyzrec in xyzrecs:
            all_xy_matched += [xy[0] for xy in xyzrec]
            all_xz_matched += [xz[1] for xz in xyzrec]
        for xyrec, xzrec in zip(all_xy_matched, all_xz_matched):
            if xyrec in xyrecs:
                xyrecs.remove(xyrec)
            if xzrec in xzrecs:
                xzrecs.remove(xzrec)
        all_remaining_xy = [self.all_xy[i] for i in xyrecs]
        all_remaining_xz = [self.all_xz[j] for j in xzrecs]
        if plotornot:
            fig, (ax, ax2) = pl.subplots(
                1, 2, sharex=True, sharey=True, figsize=(6, 6))
            ax.set_xlim([pcw, self.framewindow[1] - self.framewindow[0]])
            ax.set_title("XY Misses")
            ax2.set_xlim([pcw, self.framewindow[1] - self.framewindow[0]])
            ax.set_ylim([0, 1888])
            ax2.set_ylim([0, 1888])
            ax2.set_title("XZ Misses")
            for pcounter, para in zip(xyrecs, all_remaining_xy):
                if len(para.location) > length_thresh:
                    ax.plot(
                        range(para.timestamp,
                              para.timestamp + len(para.location)),
                        [x[0] for x in para.location],
                        color='r')
                    ax.text(para.timestamp + len(para.location),
                            para.location[-1][0], str(pcounter),
                            fontsize=8, clip_on=True)
            for pcounter, para in zip(xzrecs, all_remaining_xz):
                if len(para.location) > length_thresh:
                    ax2.plot(
                        range(para.timestamp,
                              para.timestamp + len(para.location)),
                        [x[0] for x in para.location],
                        color='b')
                    ax2.text(para.timestamp + len(para.location),
                             para.location[-1][0], str(pcounter),
                             fontsize=8, clip_on=True)

            pl.ion()
            pl.show()

        self.unpaired_xy = []
        self.unpaired_xz = []
        for xy_id, xy_para in zip(xyrecs, all_remaining_xy):
            up_xy_coords = [(np.nan, np.nan) for i in range(
                self.framewindow[0], self.framewindow[1])]
            inv_y = [(x, 1888-y) for (x, y) in xy_para.location]
            up_xy_coords[xy_para.timestamp:xy_para.timestamp+len(
                xy_para.location)] = inv_y
            self.unpaired_xy.append((xy_id, xy_para, up_xy_coords))
        for xz_id, xz_para in zip(xzrecs, all_remaining_xz):
            up_xz_coords = [(np.nan, np.nan) for i in range(
                self.framewindow[0], self.framewindow[1])]
            inv_z = [(x2, 1888-z) for (x2, z) in xz_para.location]
            up_xz_coords[xz_para.timestamp:xz_para.timestamp+len(
                xz_para.location)] = inv_z
            self.unpaired_xz.append((xz_id, xz_para, up_xz_coords))
            
#        self.unpaired_xz = zip(xzrecs, all_remaining_xz)

    #this function plots a list of pairs in one figure, with all xy recs in magenta, all xz in cyan

    def x_y_z_coords(self, rec_id):
        pl.figure(figsize=(6, 6))
        pl.plot(self.para3Dcoords[rec_id*3])
        pl.plot(self.para3Dcoords[(rec_id*3) + 1])
        pl.plot(self.para3Dcoords[(rec_id*3) + 2])
        self.plotxyzrec(rec_id)
    
    def plotxyzrec(self, rec_id):
        fg = pl.figure(figsize=(6, 6))
        ax = fg.add_subplot(111)
        ax.set_title('XY XZ Matcher')
        ax.set_xlim([0, self.framewindow[-1] - self.framewindow[0]])
        ax.set_ylim([0, 1888])
        pl.hold(True)
        for pair in self.xyzrecords[rec_id]:
            ax.plot(
                range(self.all_xy[pair[0]].timestamp,
                      self.all_xy[pair[0]].timestamp +
                      len(self.all_xy[pair[0]].location)),
                [x[0] for x in self.all_xy[pair[0]].location],
                color='m')
            ax.text(self.all_xy[pair[0]].timestamp +
                    len(self.all_xy[pair[0]].location) + 5,
                    [x[0] for x in self.all_xy[pair[0]].location][-1],
                    str(pair[0]))
            ax.plot(
                range(self.all_xz[pair[1]].timestamp,
                      self.all_xz[pair[1]].timestamp +
                      len(self.all_xz[pair[1]].location)),
                [x2[0] for x2 in self.all_xz[pair[1]].location],
                color='c')
            ax.text(self.all_xz[pair[1]].timestamp +
                    len(self.all_xz[pair[1]].location) + 5,
                    [x2[0] for x2 in self.all_xz[pair[1]].location][-1],
                    str(pair[1]))
            ax.text(20, 1800, str(rec_id))
        pl.show()

#This function just tells you which xyzrecord a particular xy or xz para object was assigned to.

    def returnxyzrec(self, para_id, xyorxz):
        if xyorxz == 'xy':
            pairindex = 0
        elif xyorxz == 'xz':
            pairindex = 1
        for rec in self.xyzrecords:
            for pair in rec:
                if pair[pairindex] == para_id:
                    return rec
        return float('NaN')

#This function plots all the recs including misses. You can see here how the algorithm is doing and where records might be missed.

    def assign_z(self, xyrec, timewindow, zmax, skip_3D):
        # invert (i.e. small is higher) b/c not yet inverted
        max_z = zmax
        # 0 is placeholder for x2
        xz_locations = [(0, max_z)
                        for i in range(timewindow[1] - timewindow[0])]
        new_xz_para = Para(timewindow[0], (0, 0))
        new_xz_para.location = xz_locations
        new_xz_para.completed = True
        self.all_xz.append(new_xz_para)
        self.xyzrecords.append([[xyrec, len(self.all_xz) - 1, (1, 1, 500)]])
        for ind_xy, up_xy in enumerate(self.unpaired_xy):
            if up_xy[0] == xyrec:
                del self.unpaired_xy[ind_xy]
                break
        if not skip_3D:
            self.make_3D_para()
            self.clear_frames()
            self.label_para()
        
    def manual_match(self):
        xy = 0
        xz = 0
        key = raw_input("Fix?: ")
        if key == 'y':
            xy = raw_input('Enter XY rec #  ')
            xz = raw_input('Enter XZ rec #  ')
            xy = int(xy)
            xz = int(xz)
            pl.close()
            for ind_xy, up_xy in enumerate(self.unpaired_xy):
                if up_xy[0] == xy:
                    del self.unpaired_xy[ind_xy]
                    break
            for ind_xz, up_xz in enumerate(self.unpaired_xz):
                if up_xz[0] == xz:
                    del self.unpaired_xz[ind_xz]
                    break
            self.xyzrecords.append([[xy, xz, (1, 1, 500)]])
            self.clear_frames()
            return 1
        else:
            pl.close()
            return 0

    def onerec_and_misses(self, rec_id):
        fig, (ax, ax2) = pl.subplots(
            1, 2, sharex=True, sharey=True, figsize=(6, 6))
        ax.set_xlim([self.pcw, self.framewindow[1] - self.framewindow[0]])
        ax.set_ylim([0, 1888])
        ax.set_title('XY Misses')
        ax2.set_title('XZ Misses')
        for pcounter, para, xyc in self.unpaired_xy:
            ax.plot(
                range(para.timestamp, para.timestamp + len(para.location)),
                [x[0] for x in para.location],
                color='r')
            ax.text(para.timestamp + len(para.location), para.location[-1][0],
                    str(pcounter), fontsize=8, clip_on=True)
        for pcounter, para, xzc in self.unpaired_xz:
            ax2.plot(
                range(para.timestamp, para.timestamp + len(para.location)),
                [x[0] for x in para.location],
                color='b')
            ax2.text(para.timestamp + len(para.location), para.location[-1][0],
                     str(pcounter), fontsize=8, clip_on=True)
        rec = self.xyzrecords[rec_id]
        for pair in rec:
            ax.plot(
                range(self.all_xz[pair[1]].timestamp,
                      self.all_xz[pair[1]].timestamp +
                      len(self.all_xz[pair[1]].location)),
                [x2[0] for x2 in self.all_xz[pair[1]].location],
                color='c')
            ax.text(self.all_xz[pair[1]].timestamp +
                    len(self.all_xz[pair[1]].location),
                    self.all_xz[pair[1]].location[-1][0],
                    str(pair[1]), fontsize=8)
            ax2.plot(
                range(self.all_xy[pair[0]].timestamp,
                      self.all_xy[pair[0]].timestamp +
                      len(self.all_xy[pair[0]].location)),
                [x[0] for x in self.all_xy[pair[0]].location],
                color='m')
            ax2.text(self.all_xy[pair[0]].timestamp +
                     len(self.all_xy[pair[0]].location),
                     self.all_xy[pair[0]].location[-1][0],
                     str(pair[0]), fontsize=8)
        
    def recs_and_misses(self):
        fig = pl.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim([0, self.framewindow[1] - self.framewindow[0]])
        ax.set_ylim([0, 1888])
        ax.set_title('XY Misses')
        fig2 = pl.figure(figsize=(6, 6))
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim([0, self.framewindow[1] - self.framewindow[0]])
        ax2.set_ylim([0, 1888])
        ax2.set_title('XZ Misses')
        for pcounter, para, xyc in self.unpaired_xy:
            ax.plot(
                range(para.timestamp, para.timestamp + len(para.location)),
                [x[0] for x in para.location],
                color='r')
            ax.text(para.timestamp + len(para.location), para.location[-1][0],
                    str(pcounter), fontsize=8)
        for pcounter, para in self.unpaired_xz:
            ax2.plot(
                range(para.timestamp, para.timestamp + len(para.location)),
                [x[0] for x in para.location],
                color='b')
            ax2.text(para.timestamp + len(para.location), para.location[-1][0],
                     str(pcounter), fontsize=8)

        for rec in self.xyzrecords:
            for pair in rec:
                ax.plot(
                    range(self.all_xz[pair[1]].timestamp,
                          self.all_xz[pair[1]].timestamp +
                          len(self.all_xz[pair[1]].location)),
                    [x2[0] for x2 in self.all_xz[pair[1]].location],
                    color='c')
                ax.text(self.all_xz[pair[1]].timestamp +
                        len(self.all_xz[pair[1]].location),
                        self.all_xz[pair[1]].location[-1][0],
                        str(pair[1]), fontsize=8)
                ax2.plot(
                    range(self.all_xy[pair[0]].timestamp,
                          self.all_xy[pair[0]].timestamp +
                          len(self.all_xy[pair[0]].location)),
                    [x[0] for x in self.all_xy[pair[0]].location],
                    color='m')
                ax2.text(self.all_xy[pair[0]].timestamp +
                         len(self.all_xy[pair[0]].location),
                         self.all_xy[pair[0]].location[-1][0],
                         str(pair[0]), fontsize=8)
        pl.show()

    #This function takes the xyz records and makes 3 rows of x,y,z coordinates for each record. The timestamp of each record is encoded in its column location within the para3Dcoords matrix.t

    def make_3D_para(self):
        self.interp_indices = []
        self.para3Dcoords = np.zeros((
            len(self.xyzrecords) * 3, self.framewindow[1] - self.framewindow[0]
        ))
        # *3 because each record will have x,y, and z.
        # initialize the matrix with zeros.
        # this function creates a list of nans the length of para3Dcoords.
        # in regions where the xyz record is defined,
        # nans are replaced with an interped  xy coordinate from the xy para
        # object and the z coordinate from the xz object.

        def create3dpath(xyzrec):
            nan_base_x = [float('nan')
                          for i in range(self.framewindow[1] -
                                         self.framewindow[0])]
            nan_base_y = [float('nan')
                          for i in range(self.framewindow[1] -
                                         self.framewindow[0])]
            nan_base_z = [float('nan')
                          for i in range(self.framewindow[1] -
                                         self.framewindow[0])]
            nan_base_x2 = [float('nan')
                           for i in range(self.framewindow[1] -
                                          self.framewindow[0])]
            for pair in xyzrec:
                x = [xcoord
                     for xcoord, ycoord in self.all_xy[pair[0]].location]
                y = [ycoord
                     for xcoord, ycoord in self.all_xy[pair[0]].location]
                z = [zcoord
                     for x2coord, zcoord in self.all_xz[pair[1]].location]
                x2 = [x2coord
                      for x2coord, zcoord in self.all_xz[pair[1]].location]
                nan_base_x[self.all_xy[pair[0]].timestamp:self.all_xy[pair[
                    0]].timestamp + len(self.all_xy[pair[0]].location)] = x
                nan_base_y[self.all_xy[pair[0]].timestamp:self.all_xy[pair[
                    0]].timestamp + len(self.all_xy[pair[0]].location)] = y
                nan_base_z[self.all_xz[pair[1]].timestamp:self.all_xz[pair[
                    1]].timestamp + len(self.all_xz[pair[1]].location)] = z
                nan_base_x2[self.all_xz[pair[1]].timestamp:self.all_xz[pair[
                    1]].timestamp + len(self.all_xz[pair[1]].location)] = x2

            return nan_base_x, nan_base_y, nan_base_z, nan_base_x2

        def max_z_scan(x2input, poi):
            max_z_stretch = False
            start_ind = 0
            for ind, x2c in enumerate(x2input):
                if max_z_stretch:
                    if x2c != 0 or ind == (len(x2input) - 1):
                        self.interp_indices.append(
                            [poi, 2, [[start_ind, ind]]])
                        max_z_stretch = False
                else:
                    if x2c == 0:
                        start_ind = ind
                        max_z_stretch = True

        def coord_inference(coords_in, para_number):
            duration_thresh = 200
            nanstretch = False
            nan_start_ind = 0
            nan_end_ind = 0
            nan_windows = []
            coords_out = np.copy(coords_in)
            inferred_windows = []
            for ind, c in enumerate(coords_in):
                if math.isnan(c) and not nanstretch and ind > 0:
                    if not math.isnan(coords_in[ind-1]):
                        nan_start_ind = ind
                        nanstretch = True

                elif nanstretch and not math.isnan(c):
                    nan_end_ind = ind
                    nan_windows.append([nan_start_ind, nan_end_ind])
                    nanstretch = False
# this checks if you are at the end of the record and have no z info.
# create a linear interpolation based on most recent 10 z coords

            for win in nan_windows:
                if win[1]-win[0] < duration_thresh:
                    if win[1] == len(coords_in) - 1:
                        slope = (
                            coords_in[win[0]-1] - coords_in[win[0]-11]) / 10.0
                        width = win[1] - win[0]
                        endpoint = int(slope * width) + coords_in[win[0]-1]
                        if not math.isnan(slope):
                            coords_out[win[0]:win[1]] = np.linspace(
                                coords_in[win[0]-1],
                                endpoint,
                                width).astype(np.int)
                    else:
                        coords_out[win[0]:win[1]] = np.linspace(
                            coords_in[win[0]-1],
                            coords_in[win[1]],
                            win[1]-win[0]).astype(np.int)
                    inferred_windows.append(win)
            if inferred_windows:
                self.interp_indices.append([para_number, 1,
                                            inferred_windows])
            # so here can have up to 3 entries per para
            return coords_out

        for recnumber, rec in enumerate(self.xyzrecords):
            x, y, z, x2 = create3dpath(rec)
            index = recnumber * 3
            yinv = [1888 - ycoord for ycoord in y]
            zinv = [1888 - zcoord for zcoord in z]
            zinv_nonan = coord_inference(zinv, recnumber)
            xinf = coord_inference(x, recnumber)
            yinf = coord_inference(yinv, recnumber)
            max_z_scan(x2, recnumber)
            self.para3Dcoords[index] = xinf
            self.para3Dcoords[index + 1] = yinf
#these inversions put paracoords in same reference frame as fish
            self.para3Dcoords[index + 2] = zinv_nonan
            # pl.plot(zinv_nonan)
            # pl.plot(zinv)
            # pl.show()
        np.save(
            '/Users/nightcrawler2/PreycapMaster/3D_paracoords.npy',
            self.para3Dcoords)



# use tsne here? have to figure out some way to say something about states and velocities. need to have a "mode" where you can correlate any given record to that mode and say "para is in x state at this time, then y state, then x state". mode will be a kernel? like a "this is tumbling" kernel. probably want to add dot_list to tsne and also magnitude list. 

        # dot each paravector with itself at varying intervals.

        #Simple function to take the 3D paramecia coords and create an animated graph.

    def graph3D(self, animatebool):
        framecount = self.framewindow[1] - self.framewindow[0]
        graph_3D = pl.figure(figsize=(6, 6))
        ax = graph_3D.add_subplot(111, projection='3d')
        ax.set_title('3D Para Record')
        ax.set_xlim([0, 1888])
        ax.set_ylim([0, 1888])
        ax.set_zlim([0, 1888])
        pl.hold(True)
        if not animatebool:
            for ind in range(0, self.para3dcoords.shape[0], 3):
                x = self.para3Dcoords[ind]
                y = self.para3Dcoords[ind + 1]
                z = self.para3Dcoords[ind + 2]
                ax.plot(
                    xs=x,
                    ys=y,
                    zs=z,
                    color=[np.random.random(), np.random.random(),
                           np.random.random()],
                    ls='-',
                    marker='o',
                    ms=8,
                    markevery=[-1])
            pl.show()
            return True

        elif animatebool:

            def updater(num, plots):
                for id, plt in enumerate(plots):
                    if num > 0:
#be careful here make sure you know what's being plotted when
                        x = self.para3Dcoords[id * 3, 0:num]
                        y = self.para3Dcoords[id * 3 + 1, 0:num]
                        z = self.para3Dcoords[id * 3 + 2, 0:num]
                        if not math.isnan(x[-1]):
                            plt.set_data(x, y)
                            plt.set_3d_properties(z)
                return plots

            plotlist = []
            for para in range(self.para3Dcoords.shape[0] / 3):
                templot, = ax.plot(
                    xs=[],
                    ys=[],
                    zs=[],
                    color=[np.random.random(), np.random.random(),
                           np.random.random()],
                    ls='-',
                    marker='o',
                    ms=8,
                    markevery=[-1])
                plotlist.append(templot)
            line_ani = anim.FuncAnimation(
                graph_3D,
                updater,
                framecount,
                fargs=[plotlist],
                interval=10,
                repeat=False,
                blit=False)
            pl.show()

#Tihs function labels the paramecia videos generated in the findpara method and adds the index of the xyz record to the video in each plane.

    def make_id_movies(self):
        fourcc = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S')
        pvid_id_top = cv2.VideoWriter('pvid_id_top.AVI', fourcc, 30,
                                      (1888, 1888), True)
        pvid_id_side = cv2.VideoWriter('pvid_id_side.AVI', fourcc, 30,
                                       (1888, 1888), True)

        while True:
            try:
                im = self.topframes.popleft()
                im2 = self.sideframes.popleft()
                pvid_id_top.write(im)
                pvid_id_side.write(im2)
            except IndexError:
                break
        pvid_id_top.release()
        pvid_id_side.release()

# OPENCV STILL USED HERE. USE IMAGEIO HERE AND IT WILL BE FINE. 
        
    def watch_event(self, top_side_or_cont):
        pcw = self.pcw
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        if top_side_or_cont == 1:
            for im in self.topframes:
                im = cv2.resize(im, (700, 700))
                cv2.imshow('vid', im)
#                cv2.resizeWindow('win', 500, 500)
                cv2.waitKey(15)
        elif top_side_or_cont == 2:
            for im in self.sideframes:
                im = cv2.resize(im, (700, 700))
                cv2.imshow('vid', im)
                cv2.waitKey(15)
        elif top_side_or_cont == 0:
            contvid = imageio.get_reader(
                self.directory + 'conts.AVI', 'ffmpeg')
            for fr in range(self.framewindow[0]+pcw, self.framewindow[1], 1):
                contframe = contvid.get_data(fr)
                cv2.imshow('vid', contframe)
                cv2.waitKey(15)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    def label_para(self):
        frame_num = self.pcw
        frame_num = int(frame_num)
        temp_top = deque()
        temp_side = deque()
        while True:
            try:
                im = self.topframes.popleft()
                im2 = self.sideframes.popleft()
            except IndexError:
                break
            if frame_num % 100 == 0:
                print(frame_num)
            para_id = 0
            # the row index of the para3Dcooords matrix defines xyz id.
            for xyr in self.unpaired_xy:
                # indicates a nan.
                if math.isnan(xyr[2][frame_num][0]):
                    pass
                else:
                    cv2.putText(
                        im,
                        str(xyr[0]),
                        (int(xyr[2][frame_num][0]),
                         int(1888 - xyr[2][frame_num][1])),
                        0,
                        1,
                        color=(255, 0, 255))
                    
            for xzr in self.unpaired_xz:
                # indicates a nan.
                if math.isnan(xzr[2][frame_num][0]):
                    pass
                else:
                    cv2.putText(
                        im2,
                        str(xzr[0]),
                        (int(xzr[2][frame_num][0]),
                         int(1888 - xzr[2][frame_num][1])),
                        0,
                        1,
                        color=(255, 0, 255))
                
            for id in range(0, self.para3Dcoords.shape[0], 3):
                if math.isnan(self.para3Dcoords[id, frame_num]) or math.isnan(
                        self.para3Dcoords[id + 2, frame_num]):
                    pass
                else:
                    cv2.putText(
                        im,
                        str(para_id),
                        (int(self.para3Dcoords[id, frame_num]),
                         1888 - int(self.para3Dcoords[id + 1, frame_num])),
                        0,
                        1,
                        color=(0, 255, 255))
                    cv2.putText(
                        im2,
                        str(para_id),
                        (int(self.para3Dcoords[id, frame_num]),
                         1888 - int(self.para3Dcoords[id + 2, frame_num])),
                        0,
                        1,
                        color=(0, 255, 255))
                para_id += 1
            temp_top.append(im)
            temp_side.append(im2)
            frame_num += 1
        self.topframes = temp_top
        self.sideframes = temp_side
        


#this function simply wraps all ParaMaster methods. It calls them sequentially after the ParaMaster class has been initialized with proper time bounds.


    def parawrapper(self, showstats):
        print('hey im in parawrapper')
        self.findpara([[10, 3, 5, 3], [10, 3, 5, 3], 6], False, True)
        if self.startover:
            return
        self.makecorrmat()
        self.corr_mat_original = copy.deepcopy(self.corr_mat)
        self.makexyzrecords()
        self.make_3D_para()
        self.find_misses(0)
        if showstats:
            self.manual_match()
            self.recs_anqd_misses()
            self.graph3D(True)
        if self.makemovies:
            self.label_para()
#            self.make_id_movies()
#        self.exporter()

    def manual_join(self, rec1, rec2):
        self.xyzrecords[int(rec1)] += self.xyzrecords[int(rec2)]
        del self.xyzrecords[int(rec2)]
        self.make_3D_para()
        self.clear_frames()
        self.label_para()

    def manual_subtract(self, rec_id):
        print("Manual Subtraction")
        self.onerec_and_misses(rec_id)
        self.plotxyzrec(rec_id)
        fix = raw_input("Fix? ")
        if fix != 'y':
            pl.close()
            return 0
        xy = raw_input("Enter XY rec ")
        xz = raw_input("Enter XZ rec ")
        xy = int(xy)
        xz = int(xz)
        temp_xyzrec = []
        for xyz_pair in self.xyzrecords[rec_id]:
            xy_id = xyz_pair[0]
            xz_id = xyz_pair[1]
            if xy_id == xy and xz_id == xz:
                xypara_obj = self.all_xy[xy_id]
                xzpara_obj = self.all_xz[xz_id]
                xy_coords = [(np.nan, np.nan) for i in range(
                    self.framewindow[0], self.framewindow[1])]
                inv_y = [(x, 1888-y) for (x, y) in xypara_obj.location]
                xy_coords[xypara_obj.timestamp:xypara_obj.timestamp+len(
                    xypara_obj.location)] = inv_y
                self.unpaired_xy.append((xy_id, xypara_obj, xy_coords))
                xz_coords = [(np.nan, np.nan) for i in range(
                    self.framewindow[0], self.framewindow[1])]
                inv_z = [(x2, 1888-z) for (x2, z) in xzpara_obj.location]
                xz_coords[xzpara_obj.timestamp:xzpara_obj.timestamp+len(
                    xzpara_obj.location)] = inv_z
                self.unpaired_xz.append((xz_id, xzpara_obj, xz_coords))
            else:
                temp_xyzrec.append(xyz_pair)
        self.xyzrecords[rec_id] = temp_xyzrec
        print temp_xyzrec
        self.make_3D_para()
        self.clear_frames()
        self.label_para()
        return 1

    def manual_add(self, rec_id):
        self.onerec_and_misses(rec_id)
        self.plotxyzrec(rec_id)
        fix = raw_input("Fix? ")
        if fix != 'y':
            pl.close()
            return 0
        xy = raw_input("Enter XY rec ")
        xz = raw_input("Enter XZ rec ")
        xy = int(xy)
        xz = int(xz)
        for ind_xy, up_xy in enumerate(self.unpaired_xy):
            if up_xy[0] == xy:
                del self.unpaired_xy[ind_xy]
                break
        for ind_xz, up_xz in enumerate(self.unpaired_xz):
            if up_xz[0] == xz:
                del self.unpaired_xz[ind_xz]
                break
        self.xyzrecords[rec_id].append([int(xy), int(xz), (1, 1, 500)])
        self.make_3D_para()
        self.clear_frames()
        self.label_para()
        pl.close()
        return 1


def imcont(image, params):
    thresh, erode_win, dilate_win, med_win = params
    r, th = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    ek = np.ones([erode_win, erode_win]).astype(np.uint8)
    dk = np.ones([dilate_win, dilate_win]).astype(np.uint8)
    er = cv2.erode(th, ek)
    dl = cv2.dilate(er, dk)
    md = cv2.medianBlur(dl, med_win)
#    th_c = cv2.cvtColor(dl.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return md

def fix_blackline(im):
    cols_to_replace = [833, 1056, 1135, 1762, 1489]
    for c in cols_to_replace:
#        col_replacement = [int(np.mean([a,b])) for a,b in zip(im[:, c-1],im[:, c+1])]
        c_pre = np.array(im[:, c-1])
        c_post = np.array(im[:, c+1])
        col_replacement = (c_pre + c_post) / 2
        im[:, c] = col_replacement
    return im

# mean norm gets rid of luminance diffs between frames by looking at tank edge
# illumination
def brsub_img(img, ir_br):
#    brmean = np.mean(ir_br[1880:, :])
    img = fix_blackline(img)
 #   im_avg = np.mean(img[1880:, :])
#    img_adj = (img * (brmean/im_avg)).astype(np.uint8)
  #  brsub = cv2.absdiff(img_adj, ir_br)
    brsub = cv2.absdiff(img, ir_br)
    return brsub

def return_paramaster_object(start_ind,
                             end_ind,
                             makemovies, directory, showstats, pcw):
    paramaster = ParaMaster(start_ind, end_ind, directory, pcw)
    if makemovies:
        paramaster.makemovies = True
    paramaster.parawrapper(showstats)
    return paramaster


if __name__ == '__main__':
    pmaster = return_paramaster_object(1000,
                                       1599,
                                       False,
                                       os.getcwd() + '/Fish00/',
                                       False,
                                       600)
#     paramaster = ParaMaster(1499, 1550, os.getcwd() + '/Fish00/')
#     paramaster.makemovies = False
#     paramaster.parawrapperq(False)
# #    paramaster.find_paravectors(False)
