import csv
import os
import copy
import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr, ttest_ind
from matplotlib import use
import bayeslite as bl
from iventure.utils_bql import query
from iventure.utils_bql import subsample_table_columns
from scipy.stats import norm
import seaborn as sb
from matplotlib import pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, ListedColormap
from toolz.itertoolz import sliding_window

# for two variable regression, compare 2 queries, and single reg,
# take a color arg. for 2 queries, towards and away
# will be dark and light, just like in minefield. 

class BayesDB_Simulator:
    # use -1 model for no particular model
    def __init__(self, fish_id, file_id, model_num):
        self.bdb_file = bl.bayesdb_open(fish_id + '/' + file_id)
#        self.bdb_file = bl.bayesdb_open(
#            '/home/nightcrawler/bayesDB/Bolton_HuntingBouts_Sim_Inverted_optimized.bdb')

        self.model_varbs = {"Model Number": model_num,
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
        if self.model_varbs['Model Number'] < 0:
            self.bdb_file.execute('''CREATE TABLE "bout_simulation" AS
            SIMULATE "Bout Az", "Bout Alt", "Bout Dist", "Bout Delta Yaw", "Bout Delta Pitch", "Para Az", "Para Az Velocity", "Para Alt", "Para Alt Velocity", "Para Dist", "Para Dist Velocity"
            FROM bout_population
            LIMIT {Row Limit}; '''.format(**self.model_varbs))
        else:
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
        hist_data = df[query_exp.replace('"', '')]
        print(str(hist_data.shape[0]) + " total bouts")
        p = sb.distplot(hist_data, color='b')
        print(np.mean(hist_data))
        print(np.std(hist_data))
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
    def two_variable_regression(self, query_expression, condition, color, *labels):
        self.set_query_params(query_expression, condition)
        v1 = query_expression.split(',')[0].replace('"', '')
        v2 = query_expression.split(',')[1].replace('"', '')
        df_real = self.rejection_query(1)
        fig = pl.figure()
        nanfilt_varbs = np.array([vrb for vrb in zip(df_real[v1],
                                                     df_real[v2]) if
                                  np.isfinite(vrb).all()])
        reg_plot = sb.regplot(nanfilt_varbs[:, 0],
                              nanfilt_varbs[:, 1],
                              fit_reg=True, n_boot=100,
                              scatter_kws={'alpha': 0.15},
                              robust=False, color=color)
        rx, ry = reg_plot.get_lines()[0].get_data()
        r_slope = np.around((ry[1] - ry[0])/(rx[1] - rx[0]), 2)
        r_yint = np.around(ry[1] - r_slope*rx[1], 3)
        reg_fit = np.around(pearsonr(nanfilt_varbs[:, 0],
                                     nanfilt_varbs[:, 1])[0], 2)
        if v2 == "Bout Distance" or v2 == "Postbout Para Dist":
            reg_plot.set_ylim([0, 1200])
            reg_plot.set_xlim([0, 1200])
            reg_plot.text(100, 1000, '  ' +
                          str(r_slope) + 'x + ' + str(
                          r_yint) + ', ' + '$r^{2}$ = ' + str(reg_fit**2),
                          color=color, fontsize=16)

        else:
            reg_plot.set_xlim([-2.5, 2.5])
            reg_plot.set_ylim([-2.5, 2.5])
            reg_plot.text(-2, 2, '  ' +
                          str(r_slope) + 'x + ' + str(
                          r_yint) + ', ' + '$r^{2}$ = ' + str(reg_fit**2),
                          color=color, fontsize=16)

        if labels != ():
            labels = labels[0]
            reg_plot.set_xlabel(labels[0], fontsize=16)
            reg_plot.set_ylabel(labels[1], fontsize=16)
        print(str(len(nanfilt_varbs[:, 0])) + " Bouts")
        sb.despine()
        sb.axes_style({'ytick.right': False})
        return fig, nanfilt_varbs[:, 0], nanfilt_varbs[:, 1]
        

    def compare_sim_to_real(self, query_expression, colors):
        df_real = self.rejection_query(1)
        df_sim = self.rejection_query(0)
        # here want mean and std for all models...loop over 50 and make a list of means, stds
        sb.distplot(df_real[query_expression], bins=100, color=colors[0])
        sb.distplot(df_sim[query_expression], bins=100, color=colors[1])
        ttest_results = ttest_ind(df_real[query_expression], df_sim[query_expression])
        print ttest_results
        pl.show()

    def compare_2_queries(self, q_exp, condition1,
                          condition2, real, new_sim, color):
        colors = [(np.array(color) * 1 / np.max(color)).tolist(),
                  (np.array(color) * .6).tolist()]
        if not real:
            if new_sim:
                self.setup_rejection_query()
        self.set_query_params(q_exp, condition1)
        c1_result = self.rejection_query(real)
        self.set_query_params(q_exp, condition2)
        c2_result = self.rejection_query(real)
        fig = pl.figure()
        c1_distribution = c1_result[q_exp.replace('"', '')]
        c2_distribution = c2_result[q_exp.replace('"', '')]
        print(str(c1_distribution.shape[0]) + ' bouts in Query 1')
        print('Mean Q1 = ' + str(np.mean(c1_distribution)))
        print(str(c2_distribution.shape[0]) + ' bouts in Query 2')
        print('Mean Q2 = ' + str(np.mean(c2_distribution)))
        sb.distplot(c1_distribution, fit_kws={"color": colors[0]},
                    fit=norm, kde=False, color=colors[0], hist_kws={"alpha": .8})
        sb.distplot(c2_distribution, fit_kws={"color": colors[1]},
                    fit=norm, kde=False, color=colors[1], hist_kws={"alpha": .8})
        ttest_results = ttest_ind(c1_distribution, c2_distribution)
        print ttest_results
        sb.despine()
        pl.savefig('2query.pdf')
        pl.show()
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

def add_bouts_reversed_label(df):
    new_bout_ids = []
    first0 = True
    counter = 0
    for ind, b_num in enumerate(df["Bout Number"]):
        if b_num == 0:
            if first0:
                first0 = False
            else:
                if df["Strike Or Abort"][ind-1] <= 3 and df[
                        "Bout Number"][ind-1] < 0:
                    new_bout_ids += [i - counter for i in range(counter)]
                else:
                    new_bout_ids += np.ones(counter).tolist()
                counter = 0
        if ind == len(df["Bout Number"]) - 1:
            if b_num < 0 and df["Strike Or Abort"][ind] <= 3:
                new_bout_ids += [i - (counter+1) for i in range(counter+1)]
            else:
                new_bout_ids += np.ones(counter+1).tolist()
        counter += 1
        
    new_bout_array = np.array(new_bout_ids)
    print new_bout_array.shape
    print df["Bout Number"].shape
    df.insert(1, 'Rev Bout Number', new_bout_array)
    df.to_csv('revbouts_added.csv')
    return new_bout_ids

            
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
        inverted_row["Para Az Accel"] *= -1
        inverted_row["Postbout Para Az"] *= -1
        inverted_row["Bout Az"] *= -1
        inverted_row["Bout Delta Yaw"] *= -1
    if row["Para Alt"] < 0:
        inverted_row["Para Alt"] *= -1
        inverted_row["Para Alt Velocity"] *= -1
        inverted_row["Para Alt Accel"] *= -1
        inverted_row["Postbout Para Alt"] *= -1
        inverted_row["Bout Alt"] *= -1
        inverted_row["Bout Delta Pitch"] *= -1
    return inverted_row
        

def make_regression_plots(x1, y1, x2, y2, labels, colors):
    fig = pl.figure()
    plot1 = sb.regplot(np.array(x1),
                       np.array(y1), fit_reg=True,
                       n_boot=100, robust=False,
                       scatter_kws={'alpha': 0.15},
                       color=colors[0])
    plot2 = sb.regplot(np.array(x2),
                       np.array(y2), fit_reg=True,
                       n_boot=100,  robust=False,
                       scatter_kws={'alpha': 0.15},
                       color=colors[1])
    plot1.set_xlabel(labels[0], fontsize=16)
    plot1.set_ylabel(labels[1], fontsize=16)
    p1x, p1y = plot1.get_lines()[1].get_data()
    p2x, p2y = plot2.get_lines()[0].get_data()
    slope1 = np.around((p1y[1] - p1y[0])/(p1x[1] - p1x[0]), 2)
    slope2 = np.around((p2y[1] - p2y[0])/(p2x[1] - p2x[0]), 2)
    yint1 = np.around(p1y[1] - slope1*p1x[1], 2)
    yint2 = np.around(p2y[1] - slope2*p2x[1], 2)
    coeff1_nanfilt = np.array(
        [c1 for c1 in zip(x2, y2) if np.isfinite(c1).all()])
    coeff2_nanfilt = np.array([c2 for c2 in zip(
        x1, y1) if np.isfinite(c2).all()])
    coeff1 = np.around(pearsonr(coeff1_nanfilt[:, 0],
                                coeff1_nanfilt[:, 1])[0], 2)
    coeff2 = np.around(pearsonr(coeff2_nanfilt[:, 0],
                                coeff2_nanfilt[:, 1])[0], 2)
    xinit = np.min(p1x[0], p2x[0])
    plot1.text(xinit, 2.1, '  ' +
               str(slope2) + 'x + ' + str(
                   yint2) + ', ' + '$r^{2}$ = ' + str(coeff2**2),
               color=colors[1], fontsize=14)
    plot1.text(xinit, 1.7, '  ' +
               str(slope1) + 'x + ' + str(
                   yint1) + ', ' + '$r^{2}$ = ' + str(coeff1**2),
               color=colors[0], fontsize=14)
#    greenplot.set_ylim([-1, 1])
    if labels[1] != "Bout Distance" and labels[1] != "Postbout Para Distance":
        plot1.set_ylim([-2.5, 2.5])
    pl.show()
    return fig


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

    def div0check(div):
        if div[1] == 0:
            return np.nan
        else:
            return div[0] / float(div[1])
            
    lead_lead_intersect = np.intersect1d(pred[0], pred[2])
    lag_lag_intersect = np.intersect1d(pred[1], pred[3])
    leadaz_lagalt = np.intersect1d(pred[0], pred[3])
    lagaz_leadalt = np.intersect1d(pred[1], pred[2])
    p_leadaz_cond_leadalt = div0check(
        [lead_lead_intersect.shape[0], len(pred[2])])
    p_leadaz_cond_lagalt = div0check(
        [leadaz_lagalt.shape[0], len(pred[3])])
    p_leadalt_cond_leadaz = div0check(
        [lead_lead_intersect.shape[0], len(pred[0])])
    p_leadalt_cond_lagaz = div0check([lagaz_leadalt.shape[0], len(pred[1])])
    cond_plot = sb.barplot(range(4), [p_leadaz_cond_leadalt,
                                      p_leadaz_cond_lagalt,
                                      p_leadalt_cond_leadaz,
                                      p_leadalt_cond_lagaz], color='c')
    cond_plot.set_ylim([0, 1])
    pl.show()


def pred_wrapper(data, limits, skip_bout_numbers,
                 condition, dist_limit, absval, vels, norm_az, norm_alt,
                 az_or_alt):
    ratio_list = []
    total_bouts = []
    for lim in limits:
        pred = prediction_calculator(
            data, lim, skip_bout_numbers,
            condition, az_or_alt, dist_limit,
            absval, vels, norm_az, norm_alt)
        prediction_conditionals(pred)
        if az_or_alt == 'az':
            try:
                ratio_list.append(len(pred[0]) / float(len(pred[1])))
            except ZeroDivisionError:
                ratio_list.append(np.nan)
            total_bouts.append(len(pred[0]) + len(pred[1]))
        if az_or_alt == 'alt':
            try:
                ratio_list.append(len(pred[2]) / float(len(pred[3])))
            except ZeroDivisionError:
                ratio_list.append(np.nan)
            total_bouts.append(len(pred[2]) + len(pred[3]))
    sb.barplot(range(len(ratio_list)), ratio_list, color='g')
    pl.savefig('pred_wrapper.pdf')
    pl.show()
    bout_assignments = pred[-1]
    print("TOTAL BOUTS")
    print total_bouts
    return total_bouts, bout_assignments


def prediction_calculator(data, limit,
                          skip_bout_numbers, condition,
                          az_or_alt, dist_limit, absval, vels,
                          norm_az, norm_alt):
    leading_az = []
    lagging_az = []
    leading_alt = []
    lagging_alt = []
    bout_assignment = []
    for i in range(len(data["Para Az"])):
        if absval:
            if az_or_alt == 'az':
                if not (limit[0] <= np.abs(data["Para Az"][i]) < limit[1]):
                    bout_assignment.append(0)
                    continue
                if not (vels[0] < data["Para Az Velocity"][i] < vels[1]):
                    continue
            if az_or_alt == 'alt':
                if not (limit[0] <= np.abs(data["Para Alt"][i]) < limit[1]):
                    bout_assignment.append(0)
                    continue
                if not (vels[0] < data["Para Alt Velocity"][i] < vels[1]):
                    continue
        else:
            if az_or_alt == 'az':
                if not (limit[0] <= data["Para Az"][i] < limit[1]):
                    bout_assignment.append(0)
                    continue

                if not (vels[0] < data["Para Az Velocity"][i] < vels[1]):
                    continue
            if az_or_alt == 'alt':
                if not (limit[0] <= data["Para Alt"][i] < limit[1]):
                    bout_assignment.append(0)
                    continue
                if not (vels[0] < data["Para Alt Velocity"][i] < vels[1]):
                    continue

        if (data["Strike Or Abort"][i] not in condition) or data[
                "Strike Or Abort"][i] > 3:
            bout_assignment.append(0)
            continue

        if skip_bout_numbers[0] == 'forward':
            bnumber = data["Bout Number"][i]
        elif skip_bout_numbers[0] == 'reverse':
            bnumber = data["Rev Bout Number"][i]
        if bnumber in skip_bout_numbers[1]:
            bout_assignment.append(0)
            continue
        if not (dist_limit[0] <= data["Para Dist"][i] <= dist_limit[1]):
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


        
# if az sign or alt sign NOT the same,
# means coming towards the fish. undershoot is a lead.
# if are the same sign,
# overshoot is a lead. lead is correct response in terms of prediction.
# a bout assignment of 1 is an undershooting lag.
# bout assignment of 2 is an overshooting lag
# bout assignment of 3 is an undershooting lead.
# bout assignment of 4 is an overshooting lead

        if np.sign(data["Para Az Velocity"][i]) == np.sign(
                data["Para Az"][i] - norm_az):
            az_sign_same = True
        else:
            az_sign_same = False
        if np.sign(
                data["Para Alt Velocity"][i]) == np.sign(
                    data["Para Alt"][i] - norm_alt):
            alt_sign_same = True
        else:
            alt_sign_same = False
        if np.sign(data["Para Az"][i] - norm_az) == np.sign(
                data["Postbout Para Az"][i] - norm_az):
            az_undershoot = True
        else:
            az_undershoot = False
        if np.sign(
                data["Para Alt"][i] - norm_alt) == np.sign(
                    data["Postbout Para Alt"][i] - norm_alt):
            alt_undershoot = True
        else:
            alt_undershoot = False
        
        if az_sign_same and az_undershoot:
            lagging_az.append(i)
            if az_or_alt == 'az':
                bout_assignment.append(1)
        if not az_sign_same and not az_undershoot:
            lagging_az.append(i)
            if az_or_alt == 'az':
                bout_assignment.append(2)
        if not az_sign_same and az_undershoot:
            leading_az.append(i)
            if az_or_alt == 'az':
                bout_assignment.append(3)
        if az_sign_same and not az_undershoot:
            leading_az.append(i)
            if az_or_alt == 'az':
                bout_assignment.append(4)

        if alt_sign_same and alt_undershoot:
            lagging_alt.append(i)
            if az_or_alt == 'alt':
                bout_assignment.append(1)
        if not alt_sign_same and not alt_undershoot:
            lagging_alt.append(i)
            if az_or_alt == 'alt':
                bout_assignment.append(2)
        if not alt_sign_same and alt_undershoot:
            leading_alt.append(i)
            if az_or_alt == 'alt':
                bout_assignment.append(3)
        if alt_sign_same and not alt_undershoot:
            leading_alt.append(i)
            if az_or_alt == 'alt':
                bout_assignment.append(4)
            
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


def stim_analyzer(data, para_variable, *random_stat):
    colorpal = sb.color_palette("husl", 8)
    hittypes = [1, 2, 3, 4]
    attended = []
    ignored = []
    for ind, (h, val) in enumerate(zip(data["Hunted Or Not"],
                                     data[para_variable])):
        if math.isnan(val):
            continue
        if h in hittypes:
            attended.append(val)
        if h == 0:
            ignored.append(val)
    ig_and_att = np.array(ignored + attended)
    ig_and_att = ig_and_att[~np.isnan(ig_and_att)]
    attended = np.array(attended)
    attended = attended[~np.isnan(attended)]
    sb.distplot(ig_and_att, color=colorpal[5])
    sb.distplot(attended, color=colorpal[3])
    if random_stat != ():
        for i, rs in enumerate(random_stat[0]):
            rs = np.array(rs)
            sb.distplot(rs, color=colorpal[i])
    pl.show()
    return attended, ig_and_att
    
# def calc_MI(x, y, bins):
#     c_xy = np.histogram2d(x, y, bins)[0]
#     mi = mutual_info_score(None, None, contingency=c_xy)
#     return mi

def stim_conditionals(data, conditioner_stat, stat, n_smallest):
    hunt_id_list = data["Hunt ID"]
    para_stat = data[stat]
    para_cstat = data[conditioner_stat]
    hunt_id_limits = np.where(np.diff(hunt_id_list) != 0)[0] + 1
    stat_per_hunt = []
    for firstind, secondind in sliding_window(2, hunt_id_limits):
        minstat_args = np.argsort(para_cstat[firstind:secondind])[0:n_smallest] + firstind
        stat_per_hunt += para_stat[minstat_args].tolist()
    stat_per_hunt = np.array(stat_per_hunt)
    return stat_per_hunt[~np.isnan(stat_per_hunt)]
    
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


