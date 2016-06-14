########################## fit ##################################################
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

from pp import *

def fitJA(j, start_date_rank):
    pltf.clf()
    p = artists_play_inday[j]
    p = p[start_date_rank:]
    print p
    apcount = [0] * (183 - start_date_rank)
    apdate = range(start_date_rank, 183)
    for i in p:
        apcount[i[1] - start_date_rank] = i[0]

    print apcount

    d_train = np.asarray(apdate)
    c_train = np.asarray(apcount)

    # create matrix versions of these arrays
    D_train = d_train[:, np.newaxis]
    d_test_plot = np.asarray(range(start_date_rank, 244))
    D_test_plot = d_test_plot[:, np.newaxis]

    pltf.scatter(d_train, c_train, label="training points")

    for degree in [1,2,3]:
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(D_train, c_train)
        c_test_plot = model.predict(D_test_plot)
        pltf.plot(d_test_plot, c_test_plot, label="degree %d" % degree)

    pltf.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
    pltf.show()


def test(degree):
    error_rate_of_artist = []
    weight_of_artist = []
    f_of_artist = []
    F = 0.0
    for j in range(0, 50):
        p = artists_play_inday[j]
        apcount = [0] * 184
        apdate = range(0, 184)
        for i in p:
            apcount[i[1]] = i[0]

        x = np.asarray(apdate[:122])
        x_test = np.asarray(apdate[122:])
        X = x[:, np.newaxis]
        y = np.asarray(apcount[:122])
        y_test_true = np.asarray(apcount[122:])

        X_test = x_test[:, np.newaxis]

        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_test_pred = model.predict(X_test)

        error_rate_pow2_sum = 0.0
        weight = 0.0
        for idx in range(0, len(x_test)):
            y_true = y_test_true[idx]
            if y_true == 0:
                y_true = 1 # deal with divide by zero

            error_rate_pow2_sum += (float((int(math.ceil(y_test_pred[idx])) - y_true)) / float(y_true) )**2
            weight += y_test_true[idx]

        error_rate_j = math.sqrt(error_rate_pow2_sum / float(len(x_test)))
        error_rate_of_artist.append(error_rate_j)
        weight_j = math.sqrt(weight)
        weight_of_artist.append(weight_j)
        f_j = (1 - error_rate_j) * weight_j
        f_of_artist.append(f_j)
        F += f_j

    print 'degree', degree
    print 'error_rate_of_artist', error_rate_of_artist
    print 'weight_of_artist', weight_of_artist
    print 'f_of_artist', f_of_artist
    print 'F', F


def pred(degree):
    predict_file_path = "./data/mars_tianchi_artist_plays_predict.csv"
    fp = open(predict_file_path, 'wb')
    fpwriter = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    for j in range(0, 50):
        p = artists_play_inday[j]
        apcount = [0] * 184
        apdate = range(0, 184)
        for i in p:
            apcount[i[1]] = i[0]

        x = np.asarray(apdate)
        X = x[:, np.newaxis]
        y = np.asarray(apcount)

        x_future = np.asarray(range(184, 245))
        X_future = x_future[:, np.newaxis]

        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_future = model.predict(X_future)

        artist_id = artists_rank_to_id[j]
        for idx in range(0, 61):
            date = rank_to_date[x_future[idx]]
            play_num = int(math.ceil(y_future[idx]))
            if play_num < 0:
                play_num = 0
            row = [artist_id, play_num, date]
            print row
            fpwriter.writerow(row)

    fp.close()

def pred(degree):
    predict_file_path = "./data/mars_tianchi_artist_plays_predict.csv"
    fp = open(predict_file_path, 'wb')
    fpwriter = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    for j in range(0, 50):
        p = artists_play_inday[j]
        apcount = [0] * 184
        apdate = range(0, 184)
        for i in p:
            apcount[i[1]] = i[0]

        x = np.asarray(apdate)
        X = x[:, np.newaxis]
        y = np.asarray(apcount)

        x_future = np.asarray(range(184, 245))
        X_future = x_future[:, np.newaxis]

        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_future = model.predict(X_future)

        artist_id = artists_rank_to_id[j]
        for idx in range(0, 61):
            date = rank_to_date[x_future[idx]]
            play_num = int(math.ceil(y_future[idx]))
            if play_num < 0:
                play_num = 0
            row = [artist_id, play_num, date]
            print row
            fpwriter.writerow(row)

    fp.close()

def predDegs(degree, start_date_rank_list):
    predict_file_path = "./data/mars_tianchi_artist_plays_predict.csv"
    fp = open(predict_file_path, 'wb')
    fpwriter = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    for j in range(0, 50):
        start_date_rank = start_date_rank_list[j]
        p = artists_play_inday[j]
        p = p[start_date_rank:]
        apcount = [0] * (183 - start_date_rank)
        apdate = range(start_date_rank, 183)
        for i in p:
            apcount[i[1] - start_date_rank] = i[0]

        d_train = np.asarray(apdate)
        c_train = np.asarray(apcount)

        # create matrix versions of these arrays
        D_train = d_train[:, np.newaxis]

        d_future = np.asarray(range(184, 244))
        D_future = d_future[:, np.newaxis]

        model = make_pipeline(PolynomialFeatures(degree[j]), Ridge())
        model.fit(D_train, c_train)
        c_future = model.predict(D_future)

        artist_id = artists_rank_to_id[j]
        for idx in range(0, 60):
            date = rank_to_date[d_future[idx]]
            play_num = int(math.ceil(c_future[idx]))
            if play_num < 0:
                play_num = 0
            row = [artist_id, play_num, date]
            print row
            fpwriter.writerow(row)

    fp.close()

def toCSV():
    predict_file_path = "./data/artists_play.csv"
    fp = open(predict_file_path, 'wb')
    fpwriter = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    for j in range(0, 50):
        row = []
        p = artists_play_inday[j]
        for i in p:
            row.append(i[0])
        print row
        fpwriter.writerow(row)

    fp.close()

ap_dfs = {}
d = []
date_rank = range(0, 183)
for i in range(0, 183):
    d.append(np.datetime64(pd.to_datetime(rank_to_date[date_rank[i]], format='%Y%m%d').to_datetime()))
def APtoDF():
    ap_dfs.clear()
    for j in range(0, 50):
        p = artists_play_inday[j]
        c = [0] * 183
        for i in p:
            c[i[1]] = i[0]

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
    print 'len of orig', len(orig)
    stl_w = sm.tsa.seasonal_decompose(orig.tolist(), freq=7)
    stl_w_se = stl_w.seasonal
    stl_w_tr = stl_w.trend
    stl_w_res = stl_w.resid
    stl_w.plot()
    print 'stl_w_se type', type(stl_w_se)
    print len(stl_w_se)
    print 'stl_w_se', stl_w_se
    w_s = stl_w_se[-7:]
    print 'w_s', w_s

    stl_w_rest = orig - stl_w_se
    stl_m = sm.tsa.seasonal_decompose(np.nan_to_num(stl_w_rest).tolist(), freq=30)
    stl_m_se = stl_m.seasonal
    stl_m_tr = stl_m.trend
    stl_m_res = stl_m.resid
    stl_m.plot()
    print 'stl_m_se type', type(stl_m_se)
    print len(stl_m_se)
    print 'stl_m_se', stl_m_se
    m_s = stl_m_se[-30:]
    print 'm_s', m_s

    # rest = stl_m_tr[15:167]
    #rest = stl_m_tr
    rest = stl_w_rest - stl_m_se
    rest_s = pd.Series(rest, index=d, name='artist' + str(j) + 'rest')
    plt.figure(0)
    rest_s.plot()

    order = (2, 0, 1)
    model  = ARIMA(rest_s, order, freq='D')
    model = model.fit()
    model.predict(1, 255).plot()
    rest_pred = model.predict(1, 244)
    print type(rest_pred)
    print len(rest_pred)
    print 'rest_pred', rest_pred

    rest_pred_nda = rest_pred.values
    print 'rest_pred_nda', rest_pred_nda
    # rest_pred_nda = np.insert(rest_pred_nda, 0, 1)
    #rest_pred_nda = np.append(rest_pred_nda, rest_pred_nda)
    print 'rest_pred_nda', rest_pred_nda
    print 'rest_pred_nda len', len(rest_pred_nda)

    for i in range(0, 8):
        stl_w_se = np.append(stl_w_se, w_s)
    stl_w_se = np.append(stl_w_se, w_s[:5])
    print len(stl_w_se)
    print 'stl_w_se', stl_w_se

    stl_m_se = np.append(stl_m_se, 0)
    stl_m_se = np.append(stl_m_se, m_s)
    stl_m_se = np.append(stl_m_se, m_s)
    print len(stl_m_se)
    print 'stl_m_se', stl_m_se

    compose_stl = stl_w_se + stl_m_se + rest_pred_nda
    fit_ap = np.exp(compose_stl)
    s = pd.Series(fit_ap, index=d2, name='artist' + str(j) + 'compose')
    plt.figure(1)
    s.plot()

def predstl():
    predict_file_path = "./data/mars_tianchi_artist_plays_predict.csv"
    fp = open(predict_file_path, 'wb')
    fpwriter = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    for j in range(0, 50):
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

        order = (2, 0, 1)
        model  = ARIMA(rest_s, order, freq='D')
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

        artist_id = artists_rank_to_id[j]
        for idx in range(184, 244):
            date = rank_to_date[idx]
            play_num = int(math.ceil(fit_ap[idx]))
            if play_num < 0:
                play_num = 0
            row = [artist_id, play_num, date]
            print row
            fpwriter.writerow(row)

    fp.close()
    
def aa(j, p, d, q):
    pltf.clf()
    ap_df[j].plot()
    order = (p, d, q)
    model  = ARIMA(ap_df[j], order, freq='D')
    model = model.fit()
    model.predict(1, 255).plot()

def lenOfAP():
    for j in range(0, 50):
        l = len(artists_play_inday[j])
        print j, 'th artist play ', l, 'days'

def cf():
    plt.close('all')
