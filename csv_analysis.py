import csv
import os
import copy
import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr
from matplotlib import use
import bayeslite as bl
from iventure.utils_bql import query
from iventure.utils_bql import subsample_table_columns
from scipy.stats import norm
import seaborn as sb
from matplotlib import pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, ListedColormap


class BayesDB_Simulator:
    def __init__(self, fish_id, file_id):
        self.bdb_file = bl.bayesdb_open(fish_id + '/' + file_id)
#        self.bdb_file = bl.bayesdb_open(
#            '/home/nightcrawler/bayesDB/Bolton_HuntingBouts_Sim_Inverted_optimized.bdb')
        self.model_varbs = {"Model Number": 37,
                            "Row Limit": 5000,
                            "Para Az": "BETWEEN -3.14 AND 3.14",
                            "Para Alt": "BETWEEN -1.57 AND 1.57",
                            "Para Dist": "BETWEEN 0 AND 5000",
                            "Para Az Velocity": "BETWEEN -20 AND 20",
                            "Para Alt Velocity": "BETWEEN -20 AND 20",
                            "Para Dist Velocity": "BETWEEN -5000 AND 5000"}
        self.orig_model_varbs = copy.deepcopy(self.model_varbs)
        self.query_dataframe = pd.DataFrame()
        self.query_params = {"query expression": 0, "conditioner": 0, "source": ""}

    def score_models(self):
        pass

    def set_conditions(self, condition, value):
        self.model_varbs[condition] = value

    def reset_conditions(self):
        self.model_varbs = copy.deepcopy(self.orig_model_varbs)
        
    def simulate_from_exact_pvarbs(self):
        self.query_dataframe = query(self.bdb_file,
                                     ''' SIMULATE "Bout Az", "Bout Alt",
                                     "Bout Dist", "Bout Delta Pitch", "Bout Delta Yaw"
                                     FROM bout_population
                                     GIVEN "Para Az" = {Para Az},
                                     "Para Alt" = {Para Alt},
                                     "Para Dist" = {Para Dist},
                                     "Para Az Velocity" = {Para Az Velocity},
                                     "Para Alt Velocity" = {Para Alt Velocity},
                                     "Para Dist velocity" = {Para Dist Velocity}
                                     LIMIT 5000 '''.format(**self.model_varbs))

    def setup_rejection_query(self):
        self.bdb_file.execute('''DROP TABLE IF EXISTS "bout_simulation"''')
        self.bdb_file.execute('''CREATE TABLE "bout_simulation" AS
        SIMULATE "Bout Az", "Bout Alt", "Bout Dist", "Bout Delta Yaw", "Bout Delta Pitch", "Para Az", "Para Az Velocity", "Para Alt", "Para Alt Velocity", "Para Dist", "Para Dist Velocity"
        FROM bout_population
        USING MODEL {Model Number}
        LIMIT {Row Limit}; '''.format(**self.model_varbs))

    def set_query_params(self, query_expression, conditioner):
        self.query_params['query_expression'] = query_expression
        self.query_params['conditioner'] = conditioner

    def single_hist(self, query_exp, condition):
        self.set_query_params(query_exp, condition)
        df = self.rejection_query(1)        
        hist_data = df[query_exp.replace('"','')]
        p = sb.distplot(hist_data, color='b')
        return p 

    def rejection_query(self, real):
        if not real:
            self.query_params['source'] = "bout_simulation"
        elif real:
            self.query_params['source'] = "bout_table"
        if self.query_params['conditioner'] == '':
            qstring = '''SELECT {query_expression} FROM {source}'''.format(**self.query_params)
        else:
            qstring =  '''SELECT {query_expression} FROM {source} WHERE {conditioner}'''.format(**self.query_params)
#        df = query(self.bdb_file,
#                   '''SELECT {query_expression} FROM {source} WHERE {conditioner}'''.format(
#                       **self.query_params))
        print qstring
        df = query(self.bdb_file, qstring)
        self.query_dataframe = df
        return df

    # format of query_expression is '"Var1","Var2"'
    def two_variable_regression(self, query_expression, condition):
        self.set_query_params(query_expression, condition)
        v1 = query_expression.split(',')[0].replace('"', '')
        v2 = query_expression.split(',')[1].replace('"', '')           
        df_real = self.rejection_query(1)
        fig = pl.figure()
        reg_plot = sb.regplot(df_real[v1],
                              df_real[v2],
                              fit_reg=True, n_boot=100, robust=True, color='g')
        rx, ry = reg_plot.get_lines()[0].get_data()
        r_slope = np.around((ry[1] - ry[0])/(rx[1] - rx[0]), 2)
        r_yint = np.around(ry[1] - r_slope*rx[1], 2)
        reg_fit = np.around(pearsonr(df_real[v1], df_real[v2])[0], 2)
        reg_plot.text(rx[0], np.max(df_real[v2]), '  ' +
                      str(r_slope) + 'x + ' + str(
                          r_yint) + ', ' + 'r = ' + str(reg_fit),
                      color='k', fontsize=14)
        return fig

    def compare_sim_to_real(self, query_expression):
        df_real = self.rejection_query(1)
        df_sim = self.rejection_query(0)
        # here want mean and std for all models...loop over 50 and make a list of means, stds
        sb.distplot(df_real[query_expression], bins=100, color='g')
        sb.distplot(df_sim[query_expression], bins=100, color='m')
        pl.show()

    def compare_2_queries(self, q_exp, condition1, condition2, real, new_sim):
        if not real:
            if new_sim:
                self.setup_rejection_query()
        self.set_query_params(q_exp, condition1)
        c1_result = self.rejection_query(real)
        self.set_query_params(q_exp, condition2)
        c2_result = self.rejection_query(real)
        fig = pl.figure()
        sb.distplot(c1_result[q_exp.replace('"', '')], fit_kws={"color":"blue"}, fit=norm, kde=False,color='b')
        sb.distplot(c2_result[q_exp.replace('"', '')], fit_kws={"color":"yellow"}, fit=norm, kde=False,color='y')
        return fig
        
                        
def concatenate_all_csv(fish_list, file_name, invert):
    with open(os.getcwd() + '/all_huntingbouts.csv', 'wb') as csvfile:
        output_data = csv.writer(csvfile)
        firstfile = True
        for fish in fish_list:
            file_id = os.getcwd() + "/" + fish + "/" + file_name
            data = pd.read_csv(file_id)
            num_entries = len(data[data.dtypes.index[0]])
            data["Fish ID"] = [fish] * num_entries
            if firstfile:
                output_data.writerow(data.dtypes.index)
                firstfile = False
            for row in range(num_entries):
                row_dict = data.iloc[row]
                # row_dict["Bout Delta Yaw"] = -1 * np.radians(
                #     row_dict["Bout Delta Yaw"])
                # row_dict["Bout Delta Pitch"] = np.radians(
                #     row_dict["Bout Delta Pitch"])
                if invert:
                    row_dict = bout_inversion(row_dict)
                output_data.writerow(row_dict.values)
        return output_data


def generate_random_data(raw_data, invert):
    def create_noise(row):
        noise = np.random.uniform(0.9, 1.1, row.shape[0])
        noisy_row = row * noise
        return noisy_row
    row_counter = 0
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
        row_w_noise = create_noise(random_row_values)
        if not np.isfinite(row_w_noise).all():
            continue
        new_csv.loc[row_counter] = row_w_noise
        row_counter += 1
    new_csv.to_csv('huntbouts_extended_inverted.csv')


# here you will invert all bouts wrt para position so that you only have right side and upward para.
# in the model, you will need to transform backwards...i.e. when you get a left down coord, have to transform it
# back into a rightward up.

def invert_all_bouts(raw_data, drct):
    new_df = pd.DataFrame(columns=raw_data.columns.tolist())
    for i in range(raw_data.shape[0]):
        row_dict = raw_data.loc[i]
        inverted_row = bout_inversion(row_dict)
        row_values = inverted_row.values
        if not np.isfinite(row_values).all():
            continue
        new_df.loc[i] = row_values
    new_df.to_csv(drct + 'huntbouts_inverted_pre.csv')
    inv_file = drct + 'huntbouts_inverted_pre.csv'
    output_file = drct + 'huntbouts_inverted.csv'
#    column_indices = range(1, len(raw_data.columns.tolist()))
    with open(inv_file, 'rb') as source:
        reader = csv.reader(source)
        with open(output_file, 'wb') as result:
            wtr = csv.writer(result)
            for row in reader:
                wtr.writerow(row[1:])
                
                
    
def bout_inversion(row):
    inverted_row = copy.deepcopy(row)
    if row["Para Az"] < 0:
        inverted_row["Para Az"] *= -1
        inverted_row["Para Az Velocity"] *= -1
        inverted_row["Postbout Para Az"] *= -1
        inverted_row["Bout Az"] *= -1
        inverted_row["Bout Delta Yaw"] *= -1
    if row["Para Alt"] < 0:
        inverted_row["Para Alt"] *= -1
        inverted_row["Para Alt Velocity"] *= -1
        inverted_row["Postbout Para Alt"] *= -1
        inverted_row["Bout Alt"] *= -1
        inverted_row["Bout Delta Pitch"] *= -1
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
    e_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    e_plot.tick_params(labelsize=13)
    pl.show()
        

def prediction_conditionals(pred):
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
    cond_plot = sb.barplot(range(4), [p_leadaz_cond_leadalt,
                                      p_leadaz_cond_lagalt,
                                      p_leadalt_cond_leadaz,
                                      p_leadalt_cond_lagaz], color='c')
    cond_plot.set_ylim([0, 1])
    pl.show()


def pred_wrapper(data, limits, skip_bout_numbers,
                 condition, az_or_alt):
    ratio_list = []
    total_bouts = []
    for lim in limits:
        pred = prediction_calculator(
            data, lim, skip_bout_numbers, condition, az_or_alt)
        prediction_conditionals(pred)
        if az_or_alt == 'az':
            ratio_list.append(len(pred[0]) / float(len(pred[1])))
            total_bouts.append(len(pred[0]) + len(pred[1]))
        if az_or_alt == 'alt':
            ratio_list.append(len(pred[2]) / float(len(pred[3])))
            total_bouts.append(len(pred[2]) + len(pred[3]))
    sb.barplot(range(len(ratio_list)), ratio_list, color='g')
    pl.show()
    bout_assignments = pred[-1]
    return total_bouts, bout_assignments


def prediction_calculator(data, limit, skip_bout_numbers, condition, az_or_alt):
    leading_az = []
    lagging_az = []
    leading_alt = []
    lagging_alt = []
    bout_assignment = []
    for i in range(len(data["Para Az"])):
        if az_or_alt == 'az':
            if not (limit[0] <= np.abs(data["Para Az"][i]) < limit[1]):
                bout_assignment.append(0)
                continue
        if az_or_alt == 'alt':
            if not (limit[0] <= np.abs(data["Para Alt"][i]) < limit[1]):
                bout_assignment.append(0)
                continue
        if data["Strike Or Abort"][i] not in condition:
            bout_assignment.append(0)
            continue
        if data["Bout Number"][i] in skip_bout_numbers:
            bout_assignment.append(0)
            continue
        if not np.isfinite([data["Para Az Velocity"][i],
                            data["Para Alt Velocity"][i],
                            data["Para Az"][i],
                            data["Para Alt"][i],
                            data["Postbout Para Az"][i],
                            data["Postbout Para Alt"][i]]).all():
            bout_assignment.append(0)
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
            if az_or_alt == 'az':
                bout_assignment.append(0)
        else:
            leading_az.append(i)
            if az_or_alt == 'az' and az_sign_same:
                bout_assignment.append(2)
            if az_or_alt == 'az' and not az_sign_same:
                bout_assignment.append(1)
        if alt_sign_same and np.sign(
                data["Para Alt"][i]) == np.sign(data["Postbout Para Alt"][i]):
            lagging_alt.append(i)
            if az_or_alt == 'alt':
                bout_assignment.append(0)
        else:
            leading_alt.append(i)
            if az_or_alt == 'alt' and alt_sign_same:
                bout_assignment.append(2)
            if az_or_alt == 'alt' and not alt_sign_same:
                bout_assignment.append(1)
            
    sb.barplot(range(4),
               [len(leading_az),
                len(lagging_az), len(leading_alt), len(lagging_alt)])
    pl.show()
    return leading_az, lagging_az, leading_alt, lagging_alt, bout_assignment


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


def stim_analyzer(data):
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
    sb.distplot(attended, color=colorpal[3])
    dp.set_axis_bgcolor('w')
    dp.tick_params(labelsize=13)
    dp.set_xlabel(para_variable, fontsize=16)
    dp.set_ylabel('Probability Density', fontsize=16)
    pl.show()


def huntbouts_plotter(data):
    v1_cond1 = []
    v2_cond1 = []
    v1_cond2 = []
    v2_cond2 = []
    v3 = []
    v1_char = "Bout Delta Yaw"
    v2_char = "Para Az"
    v3_char = "Para Az Velocity"
#    to_reject = [-1]
    for bn, action, val1, val2, val3 in zip(data["Bout Number"],
                                            data["Strike Or Abort"],
                                            data[v1_char],
                                            data[v2_char],
                                            data[v3_char]):
        if math.isnan(val1) or math.isnan(val2) or math.isnan(val3):
            continue
        if bn < 1:
            continue
        if action == 3:
            v1_cond1.append(val1)
            v2_cond1.append(val2)
        if action < 3:
            v1_cond2.append(val1)
            v2_cond2.append(val2)
            v3.append(val3)
            
    print('Regression Fitting')
    make_regression_plots(v2_cond1,
                          v1_cond1,
                          v2_cond2, v1_cond2, [v2_char, v1_char])


if __name__ == "__main__":

#NOTE ONLY RUN FUNCTIONS AFTER YOU HAVE NORMALIZED THE YAW AND PITCH
#TO RADIANS. 
    
#csv_file = 'huntingbouts_all.csv'
#csv_file = 'stimuli_all.csv'
#csv_file = 'huntbouts1_2s.csv'
 
    fish_id = '042318_6'
    drct = os.getcwd() + '/' + fish_id + '/'
#csv_file = drct + 'huntbouts_inverted.csv'
#    csv_file = drct + 'huntbouts_filtmore.csv'
#csv_file = '~/bayesDB/huntbouts_inverted.csv'
    csv_file = drct + 'huntingbouts_filt10_velover1.csv'
    data = pd.read_csv(csv_file)
    invert_all_bouts(data, drct)

#    bdsim = BayesDB_Simulator(fish_id, 'bdb_hunts_filt.bdb')
    

    
#pred_wrapper(data, [[0, .1], [.1, .2], [.3, .4], [.4, .5]], [1,2], 'alt')

#pred_wrapper(data, [[0, .05], [.05, .1], [.1, .15], [.15, .2], [.2, .25], [.25, .3]], [1,2], 'alt')

#pred_wrapper(data, [[0, 1]], [1,2], 'az')





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


