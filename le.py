########################## fit ##################################################
import numpy as np
import matplotlib.pyplot as pltf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from pp import *
apcount = []
apdate = []
def fitJU(j):
    pltf.clf()
    p = artists_play_inday[j]
    del apcount[:]
    del apdate[:] = range(0, 244)
    for i in p:
        apcount.append(i[0])
        apdate.append(i[1])
    x = np.asarray(apdate)
    y = np.asarray(apcount)
    print x
    print y
    
    x_plot = np.asarray(range(0, 245))
    # create matrix versions of these arrays
    X = x[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    pltf.scatter(x, y, label="training points")
    
    for degree in [1,2,3]:
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_plot = model.predict(X_plot)
        pltf.plot(x_plot, y_plot, label="degree %d" % degree)
    
    pltf.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
    pltf.show()
########################## fit ##################################################
