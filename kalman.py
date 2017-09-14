from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from numpy import random
from numpy.random import randn

std_x, std_y = 40, 40
dt = 1.0

random.seed(1234)
kf = KalmanFilter(2, 2)
kf.x = np.array([0., 0.])
kf.R = np.diag([std_x**2, std_y**2])
kf.F = np.array([[1, 0],[0, 1]])
kf.H = np.array([[1, 0],[0, 1]])
 
kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
# kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)

zs=[np.asarray([i,j], dtype=float) for i in x for j in y]
# zs = [np.array([i + randn()*std_x, 
#                 i + randn()*std_y]) for i in range(100)] 
print (zs)
xs, _, _, _ = kf.batch_filter(zs)
plt.plot(xs[:, 0], xs[:, 2])