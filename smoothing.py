import numpy as np
from numpy import random
from numpy.random import randn
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import kf_book.book_plots as bp

def plot_rts(noise, Q=0.001, zs, show_velocity=False):
    random.seed(123)
    fk = KalmanFilter(dim_x=2, dim_z=1)

    fk.x = np.array([0., 1.])      # state (x and dx)

    fk.F = np.array([[1., 1.],
                     [0., 1.]])    # state transition matrix

    fk.H = np.array([[1., 0.]])    # Measurement function
    fk.P = 10.                     # covariance matrix
    fk.R = noise                   # state uncertainty
    fk.Q = Q                       # process uncertainty

    # create noisy data
    # zs = np.asarray([t + randn()*noise for t in range (40)])

    # filter data with Kalman filter, than run smoother on it
    mu, cov, _, _ = fk.batch_filter(zs)
    M,P,C = fk.rts_smoother(mu, cov)

    # plot data
    if show_velocity:
        index = 1
        print('gu')
    else:
        index = 0
    if not show_velocity:
        bp.plot_measurements(zs, lw=1)
    plt.plot(M[:, index], c='b', label='RTS')
    plt.plot(mu[:, index], c='g', ls='--', label='KF output')
    if not show_velocity:
        N = len(zs)
        plt.plot([0, N], [0, N], 'k', lw=2, label='track') 
    plt.legend(loc=4)
    plt.show()


#####################Implemented as below but didn't worked###################
from filterpy.kalman import KalmanFilter
import numpy as np
from auto_mileage.models import *
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
# a = AutoMileage.objects.get(pk='HPTRIP201762281036769225579HP')
a = AutoMileage.objects.get(pk='HPTRIP20176271825306280808395HP')
f = a.coordinate_map
np.set_printoptions(formatter={'float_kind': '{:f}'.format})
c = []
time = []

for k, v in sorted(f.iteritems()):
    if type(v) is dict:
        c.extend(v['gps_point'])

x = []
y = []
for i in c:
    x.append(i.split(",")[0])
    y.append(i.split(",")[1])


plt.plot(x[:666], y[:666])
plt.savefig(str(a.trip_id) + "-01")
plt.clf()
print "done1"

def kalman_smooth(zs,noise=7, Q=0.001):
    fk = KalmanFilter(dim_x=2, dim_z=1)
    dz = np.mean(np.diff(zs[:20]))

    fk.x = np.array([0., dz])      # state (x and dx)

    fk.F = np.array([[1., dz],
                     [0., 1.]])    # state transition matrix

    fk.H = np.array([[1., 0.]])    # Measurement function
    fk.P = 10.                     # covariance matrix
    fk.R = noise                   # state uncertainty
    fk.Q = Q                       # process uncertainty
    index = 0
    # filter data with Kalman filter, than run smoother on it
    mu, cov, _, _ = fk.batch_filter(zs)
    M, P, C = fk.rts_smoother(mu, cov)

    return mu[:, index]

kmx = kalman_smooth(np.asarray(x[:666], dtype=float))
kmy = kalman_smooth(np.asarray(y[:666], dtype=float))
plt.plot(kmx, kmy)
plt.savefig(str(a.trip_id) + "-02")
plt.clf()
print "done2"

