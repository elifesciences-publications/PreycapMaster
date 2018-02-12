import copy
import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr
from matplotlib import use
use('Qt4Agg')
import seaborn as sb
from matplotlib import pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, ListedColormap


def generate_random_data(raw_data, invert):
    def create_noise(row):
        noise = np.random.uniform(0.9, 1.1, row.shape[0])
        noisy_row = row * noise
        return noisy_row

    new_csv = raw_data.copy()
    random_samples = 5000
    for i in range(random_samples):
        random_index = np.int(np.random.uniform(0, data.shape[0]-1))
        random_row_dict = data.loc[random_index]
        if invert:
            inverted_row = bout_inversion(random_row_dict)
            random_row_values = inverted_row.values
        else:
            random_row_values = random_row_dict.values
        row_w_noise = create_noise(random_row)
        if not np.isfinite(row_w_noise).all():
            continue

#        new_csv.loc[i+data.shape[0]] = row_w_noise
        new_csv.loc[i] = row_w_noise
    new_csv.to_csv('huntbouts_extended.csv')


# here you will invert all bouts wrt para position so that you only have right side and upward para.
# in the model, you will need to transform backwards...i.e. when you get a left down coord, have to transform it
# back into a rightward up.
    
def bout_inversion(row):
    inverted_row = copy.deepcopy(row)
    if row("Para Az") < 0:
        inverted_row("Para Az") *= -1
        inverted_row("Para Az Velocity") *= -1
        inverted_row("Bout Az") *= -1
        inverted_row("Bout Delta Yaw") *= -1
    if row("Para Alt") < 0:
        inverted_row("Para Alt") *= -1
        inverted_row("Para Alt Velocity") *= -1
        inverted_row("Bout Alt") *= -1
        inverted_row("Bout Delta Pitch") *= -1
    return inverted_row
        

def make_regression_plots(x1, y1, x2, y2, labels):
    colorpal = sb.color_palette("husl", 8)
    c_red = colorpal[0]
    c_green = colorpal[4]
    redplot = sb.regplot(np.array(x1),
                         np.array(y1), fit_reg=True,
                         n_boot=100, robust=True, color=c_red)
    greenplot = sb.regplot(np.array(x2),
                           np.array(y2), fit_reg=True,
                           n_boot=100,  robust=True, color=c_green)
    greenplot.set_xlabel(labels[0], fontsize=16)
    greenplot.set_ylabel(labels[1], fontsize=16)
    greenplot.set_axis_bgcolor('w')
#    pl.show()
    maxpos = np.max([np.max(y1), np.max(y1)])
    greenplot.set_axis_bgcolor('white')
    rx, ry = redplot.get_lines()[0].get_data()
    gx, gy = greenplot.get_lines()[1].get_data()
    r_slope = np.around((ry[1] - ry[0])/(rx[1] - rx[0]), 2)
    g_slope = np.around((gy[1] - gy[0])/(gx[1] - gx[0]), 2)
    r_yint = np.around(ry[1] - r_slope*rx[1], 2)
    g_yint = np.around(gy[1] - g_slope*gx[1], 2)
    coeff_red = np.around(pearsonr(x1, y1)[0], 2)
    coeff_green = np.around(pearsonr(x2, y2)[0], 2)
    greenplot.text(rx[0], maxpos, '  ' +
                   str(r_slope) + 'x + ' + str(
                   r_yint) + ', ' + 'r = ' + str(coeff_red),
                   color=c_red, fontsize=14)
    greenplot.text(rx[0], 1.2*maxpos, '  ' +
                   str(g_slope) + 'x + ' + str(
                   g_yint) + ', ' + 'r = ' + str(coeff_green),
                   color=c_green, fontsize=14)
#    greenplot.set_ylim([-1, 1])
    pl.show()


def colorcode(v1, v2, v3, labels):
    if np.min(v3) >= 0:
        cmap = sb.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True)
        norm = Normalize(np.min(v3), np.max(v3))
    else:
        cmap = sb.diverging_palette(260, 10, s=99, l=30, as_cmap=True)
        v3max = np.max(np.abs(v3))
        norm = Normalize(-v3max, v3max)
 #       norm = Normalize(-1, 1)
    f, ax = pl.subplots()
    points = ax.scatter(v1, v2,
                        c=v3, s=40, cmap=cmap, norm=norm)
    ax.set_xlabel(labels[0], fontsize=16)
    ax.set_ylabel(labels[1], fontsize=16)
    ax.tick_params(labelsize=13)
    ax.set_axis_bgcolor('w')
    f.colorbar(points)
    pl.show()


def value_over_hunt(data, valstring, actions, f_or_r, absval):
    def nanfill(f_or_r, huntlist):
        new_huntlist = []
        lengths = map(lambda x: len(x), huntlist)
        print np.mean(lengths)
#        sb.distplot(lengths, kde=False, color='g')
#        pl.show()
        max_length = np.max(lengths)
        for hunt in huntlist:
            lh = len(hunt)
            nanstretch = np.full(max_length - lh, np.nan).tolist()
            if f_or_r:
                new_huntlist.append(hunt + nanstretch)
            else:
                new_huntlist.append(nanstretch + hunt)
        return new_huntlist, max_length
                            
    all_hunts = []
    whole_hunt = []
    for ind, bn in enumerate(data["Bout Number"]):
        if data["Strike Or Abort"][ind] not in actions:
            continue
        if bn != -1:
            whole_hunt.append(data[valstring][ind])
        else:
            all_hunts.append(whole_hunt)
            whole_hunt = []
    all_hunts, mx_len = nanfill(f_or_r, all_hunts)
    if absval:
        all_hunts = [np.abs(h) for h in all_hunts]
    p_color = ''
    if actions == [3]:
        p_color = 'r'
    else:
        p_color = 'g'
    all_fig = pl.figure()
    all_ax = all_fig.add_subplot(111)
    for p in all_hunts:
        all_ax.plot(p)
    pl.show()
    if f_or_r:
        bout_numbers = range(0, mx_len)
    else:
        bout_numbers = range(-mx_len+1, 1)
    e_plot = sb.tsplot(all_hunts,
                       time=bout_numbers,
                       estimator=np.nanmean, color=p_color, ci=95)
    e_plot.set_ylabel(valstring, fontsize=16)
    e_plot.set_xlabel('Bout Number During Hunt', fontsize=16)
#    e_plot.set_ylim([-.6, .6])
    e_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    # if f_or_r:
    #     e_plot.set_xlim([0, 5])
    # else:
    #     e_plot.set_xlim([-5, 0])
    e_plot.tick_params(labelsize=13)
    pl.show()
        

def prediction_plotter(pred):
    counts = [len(x) for x in pred]
    lead_lead_intersect = np.intersect1d(pred[0], pred[2])
    lag_lag_intersect = np.intersect1d(pred[1], pred[3])
    leadaz_lagalt = np.intersect1d(pred[0], pred[3])
    lagaz_leadalt = np.intersect1d(pred[1], pred[2])
    p_leadaz_cond_leadalt = lead_lead_intersect.shape[0] / float(len(pred[2]))
    p_leadaz_cond_lagalt = leadaz_lagalt.shape[0] / float(len(pred[3]))
    p_leadalt_cond_leadaz = lead_lead_intersect.shape[0] / float(len(pred[0]))
    p_leadalt_cond_lagaz = lagaz_leadalt.shape[0] / float(len(pred[1]))
    print(p_leadaz_cond_leadalt,
          p_leadaz_cond_lagalt,
          p_leadalt_cond_leadaz,
          p_leadalt_cond_lagaz)
    sb.barplot(range(len(counts)), counts)
    pl.show()

    
def pred_wrapper(data, limits, condition):
    ratio_list = []
    for lim in limits:
        pred = prediction_calculator(data, lim, condition)
        ratio_list.append(len(pred[0]) / float(len(pred[1])))
    sb.barplot(range(len(ratio_list)), ratio_list)
    pl.show()
#    pl.bar(ratio_list)
#    pl.show()
    return ratio_list
 

def prediction_calculator(data, limit, condition):
    leading_az = []
    lagging_az = []
    leading_alt = []
    lagging_alt = []
    for i in range(len(data["Para Az"])):

        if not (limit[0] <= np.abs(data["Para Az"][i]) < limit[1]):
            continue
                    
        if data["Strike Or Abort"][i] not in condition:
            continue

        if data["Bout Number"][i] < 1:
            continue
        
        if not np.isfinite([data["Para Az Velocity"][i],
                            data["Para Alt Velocity"][i],
                            data["Para Az"][i],
                            data["Para Alt"][i],
                            data["Postbout Para Az"][i],
                            data["Postbout Para Alt"][i]]).all():
            continue
        az_sign_same = False
        alt_sign_same = False
        if np.sign(data["Para Az Velocity"][i]) == np.sign(data["Para Az"][i]):
            az_sign_same = True
        if np.sign(
                data["Para Alt Velocity"][i]) == np.sign(data["Para Alt"][i]):
            alt_sign_same = True
        if az_sign_same and np.sign(
                data["Para Az"][i]) == np.sign(data["Postbout Para Az"][i]):
            lagging_az.append(i)
        else:
            leading_az.append(i)
        if alt_sign_same and np.sign(
                data["Para Alt"][i]) == np.sign(data["Postbout Para Alt"][i]):
            lagging_alt.append(i)
        else:
            leading_alt.append(i)
    print len(leading_az) + len(lagging_az)
    print len(leading_az) / float(len(lagging_az))
    return leading_az, lagging_az, leading_alt, lagging_alt


def twod_scatter(data, var1, var2):
    colorpal = sb.color_palette("husl", 8)
    att_color = colorpal[-1]
    ig_color = '.75'
    attended1 = []
    attended2 = []
    ignored1 = []
    ignored2 = []
    hittypes = [1, 2, 3, 4]
    for h, val1, val2 in zip(data["Hunted Or Not"], data[var1], data[var2]):
        if math.isnan(val1) or math.isnan(val2):
            print h
            continue
        if h in hittypes:
            print('hit')
            attended1.append(val1)
            attended2.append(val2)
        if h == 0:
            print('miss')
            ignored1.append(val1)
            ignored2.append(val2)
    
    f = pl.figure()
    ax = f.add_subplot(111)
    ax.plot(ignored1, ignored2, marker='o', linestyle='None', color=ig_color)
    ax.plot(attended1, attended2,
            marker='o', linestyle='None', color=att_color, markeredgecolor='w')
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_axis_bgcolor('white')
    pl.show()
    return attended1, attended2, ignored1, ignored2



#csv_file = 'huntingbouts_all.csv'
#csv_file = 'stimuli_all.csv'
#csv_file = 'huntbouts1_2s.csv'
csv_file = 'huntbouts_rad.csv'
data = pd.read_csv(csv_file)


if csv_file == 'huntbouts1_2s.csv':
    generate_random_data(data, True)



        

# #Filter vals here

if csv_file == 'stimuli_all.csv':
    colorpal = sb.color_palette("husl", 8)
    hittypes = [1, 2, 3, 4]
    attended = []
    ignored = []
    para_variable = "Distance"
    for ind, (h, val) in enumerate(zip(data["Hunted Or Not"],
                                     data[para_variable])):
        if math.isnan(val):
            continue
        if h in hittypes:
            attended.append(val)
        if h == 0:
            ignored.append(val)
    dp = sb.distplot(ignored + attended, color='b')
 #   dp = sb.distplot(ignored, color=colorpal[0])
    sb.distplot(attended, color=colorpal[3])
    dp.set_axis_bgcolor('w')
    dp.tick_params(labelsize=13)
    dp.set_xlabel(para_variable, fontsize=16)
    dp.set_ylabel('Probability Density', fontsize=16)
    pl.show()

if csv_file == 'huntingbouts_all.csv':
    v1_cond1 = []
    v2_cond1 = []
    v1_cond2 = []
    v2_cond2 = []
    v3 = []
    v1_char = "Bout Delta Yaw"
    v2_char = "Para Az"
    v3_char = "Para Az Velocity"
#    to_reject = [-1]
    bout_limit = 1
    for bn, action, val1, val2, val3 in zip(data["Bout Number"],
                                            data["Strike Or Abort"],
                                            data[v1_char],
                                            data[v2_char],
                                            data[v3_char]):
        if math.isnan(val1) or math.isnan(val2) or math.isnan(val3):
            continue
        if bn < 1:
            continue
        # if v3_char == 'Bout Dist':
        #     if v3 < 0:
        #         continue
        if action == 3:
            v1_cond1.append(val1)
            v2_cond1.append(val2)
        if action < 3:
            v1_cond2.append(val1)
            v2_cond2.append(val2)
            v3.append(val3)
            
    if v1_char == 'Bout Delta Pitch' or v1_char == "Bout Delta Yaw":
        v1_cond1 = np.radians(v1_cond1)
        v1_cond2 = np.radians(v1_cond2)
        # Right turns create negative delta yaws b/c yaw is on unit circle. 
        if v1_char == "Bout Delta Yaw":
            v1_cond1 *= -1
            v1_cond2 *= -1
    print('Regression Fitting')
    make_regression_plots(v2_cond1,
                          v1_cond1,
                          v2_cond2, v1_cond2, [v2_char, v1_char])


pred_wrapper(data, [[0, .1], [.1, .2], [.3, .4], [.4, .5]], [3])
#pred_wrapper(data, [[0, .05], [.05, .1], [.1, .15], [.15, .2]], [3])
    
#pred = prediction_calculator(data)
#prediction_plotter(pred)
#a = twod_scatter(data, "Az Coord", "Alt Coord")
#a = twod_scatter(data, "Raw Velocity", "Dot Product")

# v_corr = velocity_correlation(data, 'Az')
#  make_regression_plots(v_corr[2], v_corr[0],
#                       v_corr[3], v_corr[1], ['Para Velocity', 'Delta Angle'])
# make_regression_plots(v_corr[6], v_corr[4],
#                       v_corr[7], v_corr[5], ['Para Velocity', 'Delta Angle'])

# TO DO:

# try seaborn pairplot across the entire dataframe. 
# randomize paramecium motion and see if it still correlates. 
