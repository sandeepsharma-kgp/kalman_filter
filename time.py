# import os
# import time

# os.environ['TZ'] = 'right/UTC' # TAI scale with 1970-01-01 00:00:10 (TAI) epoch
# time.tzset() # Unix

# from datetime import datetime, timedelta

# gps_timestamp = 1497440172221.0/1000.0 # input
# gps_epoch_as_gps = datetime(1980, 1, 6) 
# # by definition
# gps_time_as_gps = gps_epoch_as_gps + timedelta(seconds=gps_timestamp) 
# gps_time_as_tai = gps_time_as_gps + timedelta(seconds=19) # constant offset
# tai_epoch_as_tai = datetime(1970, 1, 1, 0, 0, 10)
# # by definition
# tai_timestamp = (gps_time_as_tai - tai_epoch_as_tai).total_seconds() 
# print(datetime.utcfromtimestamp(tai_timestamp))

from datetime import datetime, timedelta

# utc = 1980-01-06UTC + (gps - (leap_count(2014) - leap_count(1980)))
utc = datetime(1980, 1, 6) + timedelta(seconds=1497440172221.0/1000 - (35 - 19))
print(utc)