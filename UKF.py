from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter as UKF
import numpy as np
from auto_mileage.models import *
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl

# f = AutoMileage.objects.get(pk='HPTRIP201762281036769225579HP').coordinate_map
f = AutoMileage.objects.get(pk='HPTRIP2017322648358745027062HP').coordinate_map
# f = AutoMileage.objects.first().coordinate_map
A = AutoMileage.objects.all()
# A=f
count = 0
for a in A:
    # A=AutoMileage.objects.first()
    f = AutoMileage.objects.get(
        pk='HPTRIP2017322648358745027062HP').coordinate_map
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})
    c = []
    time = []
    for k, v in sorted(f.iteritems()):
        if type(v) is dict:
            c.extend(v['gps_point'])
            time.extend(v['gps_timestamp'])

    t = np.asarray(time, dtype=float)
    td = np.diff(t / 1000)

    x = []
    y = []
    for i in c:
        x.append(i.split(",")[0])
        y.append(i.split(",")[1])

    print count
    count += 1
    if len(x) <= 10:
        continue
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    np_xd = np.diff(x)
    np_yd = np.diff(y)
    vx = np_xd / td
    vy = np_yd / td

    plt.scatter(x, y)
    plt.savefig(str(a.trip_id) + "-01")
    plt.clf()
    # pl.plot(obs, color='b')

    def f_cv(x, dt):
        """ state transition function for a 
        constant velocity aircraft"""
        F = np.array([[1, 0], [0, 1]])
        return np.dot(F, x)

    def h_cv(x):
        return np.array([x[0], x[1]])

    sigmas = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=1.)
    # sigmas = [np.asarray([i,j], dtype=float) for i in x for j in y]
    ukf = UKF(dim_x=2, dim_z=2, fx=f_cv,
              hx=h_cv, dt=4, points=sigmas)
    ukf.x = np.array([0., 0.])
    ukf.R = np.diag([0.05, 0.05])
    ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=4, var=0.3)
    # ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=4, var=0.3)

    uxs = []
    zs = [np.asarray([i, j], dtype=float) for i in x for j in y]
    for z in zs:
        ukf.predict()
        ukf.update(z)
        uxs.append(ukf.x.copy())

    uxs = np.array(uxs)

    # plt.plot(uxs, color='r')
    plt.scatter(uxs[:, 0], uxs[:, 1])
    plt.savefig(str(a.trip_id) + "-02")
    plt.clf()


#####################################################################

def f_cv(x, dt):
    """ state transition function for a 
    constant velocity aircraft"""
    
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]],dtype=float)
    return np.dot(F, x)

def h_cv(x):
    return np.array([x[0], x[2]])
dt=4
sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
ukf = UKF(dim_x=4, dim_z=2, fx=f_cv,
          hx=h_cv, dt=dt, points=sigmas)
ukf.x = np.array([0., 0., 0., 0.])
ukf.R = np.diag([0.0011, 0.0030]) 
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=4, var=0.02)
ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=4, var=0.02)

uxs = []
zs = [np.asarray([i, j], dtype=float) for i in x[::5] for j in y[::5]]
for z in zs:
    ukf.predict()
    ukf.update(z)
    uxs.append(ukf.x.copy())
uxs = np.array(uxs)

# plt.plot(uxs, color='r')
plt.scatter(uxs[:, 0], uxs[:, 2])
plt.savefig(str(a.trip_id) + "-02")
plt.clf()