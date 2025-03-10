import harc_plot
import pvlib
import datetime
import pandas as pd
import eclipse_calc

sTime       = datetime.datetime(2024,4,8)
eTime       = datetime.datetime(2024,4,9)
lat = 41.335116
lon = -75.600692
dt_minutes = 1

solarAzEls              = harc_plot.gen_lib.calc_solar_zenith(sTime,eTime,lat,lon,minutes=dt_minutes)
solarAzEls['els']       = 90. - solarAzEls # Make Correction because harc_plot give Solar Zenith angle instead of elevation.
print(solarAzEls)

times = pd.date_range(start=sTime, end=eTime, freq=f'{dt_minutes}min')
solar_position = pvlib.solarposition.get_solarposition(times, lat, lon)
solarAzEls2 = solar_position[['elevation']].rename(columns={'elevation': 'els'})
print(solarAzEls2)
print(solarAzEls2.shape)


# print values at 12:00
print(solarAzEls.loc[sTime + datetime.timedelta(hours=12)])
print(solarAzEls2.loc[sTime + datetime.timedelta(hours=12)])


solar_dt    = datetime.timedelta(minutes=dt_minutes)
solar_times = [sTime]
while solar_times[-1] < eTime:
    solar_times.append(solar_times[-1]+solar_dt)
obsc        = eclipse_calc.calculate_obscuration(solar_times,lat,lon)
df = pd.DataFrame({'obsc':obsc}, index = solar_times)
print(df.loc[sTime + datetime.timedelta(hours=19)])