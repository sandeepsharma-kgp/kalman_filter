from filterpy.kalman import KalmanFilter
import numpy as np
from auto_mileage.models import *
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
a = AutoMileage.objects.get(pk='HPTRIP201762281036769225579HP')
f = a.coordinate_map
np.set_printoptions(formatter={'float_kind': '{:f}'.format})

#####################################################################
a = AutoMileage.objects.get(pk='HPTRIP20176271825306280808395HP')
f = a.coordinate_map
np.set_printoptions(formatter={'float_kind': '{:f}'.format})
c = []
time = []
for k, v in sorted(f.iteritems()):
    if type(v) is dict:
        print k
        c.extend(v['gps_point'])
        time.extend(v['gps_timestamp'])

t = np.asarray(time, dtype=float)
td = np.diff(t / 1000)

x = []
y = []
for i in c:
    x.append(i.split(",")[0])
    y.append(i.split(",")[1])


#####################################################################################
a = AutoMileage.objects.get(pk='HPTRIP20176271825306280808395HP')
f=a.smoothed_points
np.set_printoptions(formatter={'float_kind': '{:f}'.format})
c = []
time = []
x=[]
y=[]
for k in sorted(f):
    for i in f[k]:
        x.append(i['location']['latitude'])
        y.append(i['location']['longitude'])
    
lon1, lat1, lon2, lat2 = map(radians,[y[0],x[0],y[-1],x[-1]])
dlon = lon2 - lon1 
dlat = lat2 - lat1 
a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
c = 2 * asin(sqrt(a)) 
r = 6371 # Radius of earth in kilometers. Use 3956 for miles
return c * r

##############################
261
265
233
177
###################################
x=np.asarray(x,dtype=float)
y=np.asarray(y,dtype=float)

lon = np.asarray(map(radians,x),dtype=float)
lat = np.asarray(map(radians,y),dtype=float)
dlon = np.diff(lon[::50])
dlat = np.diff(lat[::50])
sdlat = np.asarray(map(sin,dlat/2),dtype=float)
sdlon = np.asarray(map(sin,dlon/2),dtype=float)
clat
