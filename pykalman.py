import pylab as pl
from pykalman import UnscentedKalmanFilter
import numpy as np
from auto_mileage.models import *
import matplotlib
import matplotlib.pyplot as plt
# f = AutoMileage.objects.get(pk='HPTRIP2017614113181897636669HP').coordinate_map
f = AutoMileage.objects.first().coordinate_map


c = []
time=[]
for k, v in f.iteritems():
    if type(v) is dict:
        c.extend(v['gps_point'])
        time.extend(v['gps_timestamp'])

t = np.asarray(time, dtype=float)
td=np.diff(t/1000)


x = []
y = []
for i in c:
    x.append(i.split(",")[0])
    y.append(i.split(",")[1])

# initialize parameters
def transition_function(state, noise):
    a = np.sin(state[0]) + state[1] * noise[0]
    b = state[1] + noise[1]
    return np.array([a, b])

def observation_function(state, noise):
    C = np.array([[-1, 0.5], [0.2, 0.1]])
    return np.dot(C, state) + noise

transition_covariance = np.eye(2)
random_state = np.random.RandomState(0)
observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
initial_state_mean = [0, 0]
initial_state_covariance = [[1, 0.1], [-0.1, 1]]

# sample from model
kf = UnscentedKalmanFilter(
    transition_function, observation_function,
    transition_covariance, observation_covariance,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)

observations = [np.asarray([i,j], dtype=float) for i in x for j in y]
filtered_state_estimates = kf.filter(observations)[0]
smoothed_state_estimates = kf.smooth(observations)[0]
states = [np.asarray([i,j], dtype=float) for i in x for j in y]

pl.figure()
lines_true = pl.plot(states, color='b')
lines_filt = pl.plot(filtered_state_estimates, color='r', ls='-')
lines_smooth = pl.plot(smoothed_state_estimates, color='g', ls='-.')
pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
          ('true', 'filt', 'smooth'),
          loc='lower left'
)
pl.show()