from auto_mileage.models import *
import matplotlib
import matplotlib.pyplot as plt
# f = AutoMileage.objects.get(pk='HPTRIP2017614113181897636669HP').coordinate_map
f = AutoMileage.objects.get(pk='HPTRIP20173201236438784877257HP').coordinate_map

c = []
for k, v in f.iteritems():
    if type(v) is dict:
        c.extend(v['gps_point'])
x = []
y = []
for i in c:
    x.append(i.split(",")[0])
    y.append(i.split(",")[1])
