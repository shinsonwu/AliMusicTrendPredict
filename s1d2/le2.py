########################## fit ##################################################
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import time

from pp2 import *

artists_num = len(artists_play_inday)
ap_dfs = {}
d = []
date_rank = range(0, 183)
for i in range(0, 183):
    d.append(np.datetime64(pd.to_datetime(rank_to_date[date_rank[i]], format='%Y%m%d').to_datetime()))
def APtoDF():
    ap_dfs.clear()
    for j in range(0, artists_num):
        p = artists_play_inday[j]
        c = [2.72] * 183
        for i in p:
            ci = i[0]
            if ci < 2.72:
                ci = 2.72
            c[i[1]] = ci

        s = pd.Series(np.array(c, dtype=np.float64), index=d, name='artist' + str(j) + 'play')
        ap_dfs[j] = s

    ap_df = pd.DataFrame(ap_dfs)
    return ap_df

ap_df = APtoDF()


d2 = []
date_rank2 = range(0, 244)
for i in range(0, 244):
    d2.append(np.datetime64(pd.to_datetime(rank_to_date[date_rank2[i]], format='%Y%m%d').to_datetime()))
def stl(j):
    plt.close('all')
    ap_df[j].plot()
    orig = np.log(ap_df[j])
    #print 'len of orig', len(orig)
    #print_full(orig)
    stl_w = sm.tsa.seasonal_decompose(orig.tolist(), freq=7)
    stl_w_se = stl_w.seasonal
    stl_w_tr = stl_w.trend
    stl_w_res = stl_w.resid
    stl_w.plot()
    #print 'stl_w_se type', type(stl_w_se)
    #print len(stl_w_se)
    #print 'stl_w_se', stl_w_se
    w_s = stl_w_se[-7:]
    #print 'w_s', w_s

    stl_w_rest = orig - stl_w_se
    stl_m = sm.tsa.seasonal_decompose(np.nan_to_num(stl_w_rest).tolist(), freq=30)
    stl_m_se = stl_m.seasonal
    stl_m_tr = stl_m.trend
    stl_m_res = stl_m.resid
    stl_m.plot()
    #print 'stl_m_se type', type(stl_m_se)
    #print len(stl_m_se)
    #print 'stl_m_se', stl_m_se
    m_s = stl_m_se[-30:]
    #print 'm_s', m_s

    # rest = stl_m_tr[15:167]
    #rest = stl_m_tr
    rest = stl_w_rest - stl_m_se
    rest_s = pd.Series(rest, index=d, name='artist' + str(j) + 'rest')
    plt.figure(4)
    rest_s.plot()
    rest_x = range(0, 183)
    try:
        print '++++++++ frac 0.2 ++++++++++++'
        rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.2, return_sorted=False)
        rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')
        plt.figure(4)
        rest_ss.plot()

        order = (6, 0, 1)
        model  = ARIMA(rest_ss, order, freq='D')
        model = model.fit()
        model.predict(1, 255).plot()
    except:
        try:
            print '++++++++ frac 0.18 ++++++++++++'
            rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.18, return_sorted=False)
            rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')
            plt.figure(4)
            rest_ss.plot()

            order = (6, 0, 1)
            model  = ARIMA(rest_ss, order, freq='D')
            model = model.fit()
            model.predict(1, 255).plot()
        except:
            try:
                print '++++++++ frac 0.16 ++++++++++++'
                rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.16, return_sorted=False)
                rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')
                plt.figure(4)
                rest_ss.plot()

                order = (6, 0, 1)
                model  = ARIMA(rest_ss, order, freq='D')
                model = model.fit()
                model.predict(1, 255).plot()
            except:
                try:
                    print '++++++++ frac 0.14 ++++++++++++'
                    rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.14, return_sorted=False)
                    rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')
                    plt.figure(4)
                    rest_ss.plot()

                    order = (6, 0, 1)
                    model  = ARIMA(rest_ss, order, freq='D')
                    model = model.fit()
                    model.predict(1, 255).plot()
                except:
                    try:
                        print '++++++++ frac 0.12 ++++++++++++'
                        rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.12, return_sorted=False)
                        rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')
                        plt.figure(4)
                        rest_ss.plot()

                        order = (6, 0, 1)
                        model  = ARIMA(rest_ss, order, freq='D')
                        model = model.fit()
                        model.predict(1, 255).plot()
                    except:
                        try:
                            print '++++++++ frac 0.1 ++++++++++++'
                            rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.1, return_sorted=False)
                            rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')
                            plt.figure(4)
                            rest_ss.plot()

                            order = (6, 0, 1)
                            model  = ARIMA(rest_ss, order, freq='D')
                            model = model.fit()
                            model.predict(1, 255).plot()
                        except:
                            try:
                                print '++++++++ frac 0.08 ++++++++++++'
                                rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.08, return_sorted=False)
                                rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')
                                plt.figure(4)
                                rest_ss.plot()

                                order = (6, 0, 1)
                                model  = ARIMA(rest_ss, order, freq='D')
                                model = model.fit()
                                model.predict(1, 255).plot()
                            except:
                                print '++++++++ frac 0.05 ++++++++++++'
                                rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.05, return_sorted=False)
                                rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')
                                plt.figure(4)
                                rest_ss.plot()

                                order = (4, 0, 1)
                                model  = ARIMA(rest_ss, order, freq='D')
                                model = model.fit()
                                model.predict(1, 255).plot()

    rest_pred = model.predict(1, 244)
    #print type(rest_pred)
    #print len(rest_pred)
    #print 'rest_pred', rest_pred

    rest_pred_nda = rest_pred.values
    #print 'rest_pred_nda', rest_pred_nda
    # rest_pred_nda = np.insert(rest_pred_nda, 0, 1)
    #rest_pred_nda = np.append(rest_pred_nda, rest_pred_nda)
    #print 'rest_pred_nda', rest_pred_nda
    #print 'rest_pred_nda len', len(rest_pred_nda)

    for i in range(0, 8):
        stl_w_se = np.append(stl_w_se, w_s)
    stl_w_se = np.append(stl_w_se, w_s[:5])
    #print len(stl_w_se)
    #print 'stl_w_se', stl_w_se

    stl_m_se = np.append(stl_m_se, 0)
    stl_m_se = np.append(stl_m_se, m_s)
    stl_m_se = np.append(stl_m_se, m_s)
    #print len(stl_m_se)
    #print 'stl_m_se', stl_m_se

    compose_stl = stl_w_se + stl_m_se + rest_pred_nda
    fit_ap = np.exp(compose_stl)
    s = pd.Series(fit_ap, index=d2, name='artist' + str(j) + 'compose')
    plt.figure(1)
    s.plot()

frac_l = {}
def predstl():
    frac_l.clear()
    predict_file_path = "./data/mars_tianchi_artist_plays_predict.csv"
    fp = open(predict_file_path, 'wb')
    fpwriter = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    for j in range(0, artists_num):
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        print '++++++++++++++++++++++artists', j
        plt.close('all')
        plt.figure(j)
        ap_df[j].plot()
        orig = np.log(ap_df[j])
        stl_w = sm.tsa.seasonal_decompose(orig.tolist(), freq=7)
        stl_w_se = stl_w.seasonal
        w_s = stl_w_se[-7:]

        stl_w_rest = orig - stl_w_se
        stl_m = sm.tsa.seasonal_decompose(np.nan_to_num(stl_w_rest).tolist(), freq=30)
        stl_m_se = stl_m.seasonal
        m_s = stl_m_se[-30:]

        rest = stl_w_rest - stl_m_se
        rest_s = pd.Series(rest, index=d, name='artist' + str(j) + 'rest')
        rest_x = range(0, 183)
        sp = [11, 19, 25, 30, 31, 33, 34, 43, 51, 62, 63]
        if j in sp:
            frac_l[j] = 0.05
            rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.05, return_sorted=False)
            rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

            order = (4, 0, 1)
            model  = ARIMA(rest_ss, order, freq='D')
            model = model.fit()
            model.predict(1, 255).plot()
        else:
            try:
                frac_l[j] = 0.2
                rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.2, return_sorted=False)
                rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

                order = (6, 0, 1)
                model  = ARIMA(rest_ss, order, freq='D')
                model = model.fit()
                model.predict(1, 255).plot()
            except:
                try:
                    frac_l[j] = 0.18
                    rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.18, return_sorted=False)
                    rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

                    order = (6, 0, 1)
                    model  = ARIMA(rest_ss, order, freq='D')
                    model = model.fit()
                    model.predict(1, 255).plot()
                except:
                    try:
                        frac_l[j] = 0.16
                        rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.16, return_sorted=False)
                        rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

                        order = (6, 0, 1)
                        model  = ARIMA(rest_ss, order, freq='D')
                        model = model.fit()
                        model.predict(1, 255).plot()
                    except:
                        try:
                            frac_l[j] = 0.14
                            rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.14, return_sorted=False)
                            rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

                            order = (6, 0, 1)
                            model  = ARIMA(rest_ss, order, freq='D')
                            model = model.fit()
                            model.predict(1, 255).plot()
                        except:
                            try:
                                frac_l[j] = 0.12
                                rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.12, return_sorted=False)
                                rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

                                order = (6, 0, 1)
                                model  = ARIMA(rest_ss, order, freq='D')
                                model = model.fit()
                                model.predict(1, 255).plot()
                            except:
                                try:
                                    frac_l[j] = 0.1
                                    rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.1, return_sorted=False)
                                    rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

                                    order = (6, 0, 1)
                                    model  = ARIMA(rest_ss, order, freq='D')
                                    model = model.fit()
                                    model.predict(1, 255).plot()
                                except:
                                    try:
                                        frac_l[j] = 0.08
                                        rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.08, return_sorted=False)
                                        rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

                                        order = (6, 0, 1)
                                        model  = ARIMA(rest_ss, order, freq='D')
                                        model = model.fit()
                                        model.predict(1, 255).plot()
                                    except:
                                        frac_l[j] = 0.05
                                        rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.05, return_sorted=False)
                                        rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

                                        order = (4, 0, 1)
                                        model  = ARIMA(rest_ss, order, freq='D')
                                        model = model.fit()
                                        model.predict(1, 255).plot()

        rest_pred = model.predict(1, 244)
        rest_pred_nda = rest_pred.values

        for i in range(0, 8):
            stl_w_se = np.append(stl_w_se, w_s)
        stl_w_se = np.append(stl_w_se, w_s[:5])

        stl_m_se = np.append(stl_m_se, 0)
        stl_m_se = np.append(stl_m_se, m_s)
        stl_m_se = np.append(stl_m_se, m_s)

        compose_stl = stl_w_se + stl_m_se + rest_pred_nda
        fit_ap = np.exp(compose_stl)

        s = pd.Series(fit_ap, index=d2, name='artist' + str(j) + 'compose')
        plt.figure(j)
        s.plot()
        fig = plt.figure(j)
        fig.savefig('pic_trend_3/' + str(j) + '.png')

        artist_id = artists_rank_to_id[j]
        for idx in range(184, 244):
            date = rank_to_date[idx]
            play_num = int(math.ceil(fit_ap[idx]))
            if play_num < 0:
                play_num = 0
            row = [artist_id, play_num, date]
            #print row
            fpwriter.writerow(row)


    fp.close()

def lenOfAP():
    for j in range(0, artists_num):
        l = len(artists_play_inday[j])
        print j, 'th artist play ', l, 'days'

def cf():
    plt.close('all')

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def predp(k, l):
    for j in range(k, l):
        #time.sleep(5)
        plt.close('all')
        plt.figure(j)
        ap_df[j].plot()
        orig = np.log(ap_df[j])
        stl_w = sm.tsa.seasonal_decompose(orig.tolist(), freq=7)
        stl_w_se = stl_w.seasonal
        w_s = stl_w_se[-7:]

        stl_w_rest = orig - stl_w_se
        stl_m = sm.tsa.seasonal_decompose(np.nan_to_num(stl_w_rest).tolist(), freq=30)
        stl_m_se = stl_m.seasonal
        m_s = stl_m_se[-30:]

        rest = stl_w_rest - stl_m_se
        rest_s = pd.Series(rest, index=d, name='artist' + str(j) + 'rest')
        rest_x = range(0, 183)
        rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.03, return_sorted=False)
        rest_ss = pd.Series(rest_as, index=d, name='artist' + str(j) + 'rest')

        order = (2, 0, 1)
        model  = ARIMA(rest_ss, order, freq='D')
        model = model.fit()
        rest_pred = model.predict(1, 244)

        rest_pred_nda = rest_pred.values

        for i in range(0, 8):
            stl_w_se = np.append(stl_w_se, w_s)
        stl_w_se = np.append(stl_w_se, w_s[:5])

        stl_m_se = np.append(stl_m_se, 0)
        stl_m_se = np.append(stl_m_se, m_s)
        stl_m_se = np.append(stl_m_se, m_s)

        compose_stl = stl_w_se + stl_m_se + rest_pred_nda
        fit_ap = np.exp(compose_stl)
        s = pd.Series(fit_ap, index=d2, name='artist' + str(j) + 'compose')
        plt.figure(j)
        s.plot()
        fig = plt.figure(j)
        fig.savefig('pic_trend/' + str(j) + '.png')
