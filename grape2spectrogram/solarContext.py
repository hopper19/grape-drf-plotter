"""
@authors: HamSCI, Cuong Nguyen KC3UAX
"""
import datetime

import pandas as pd
import pvlib

import eclipse_calc

class solarTimeseries(object):
    def __init__(self,sTime=None,eTime=None,lat=None,lon=None,dt_minutes=1):
        """
        Class for overlaying solar elevation angle and eclipse obscuration on timeseries axis objects.
        sTime:      Datetime to start solar calculations.
        eTime:      Datetime to end solar calculations.
        lat:        Latitude of observer.
        lon:        Longitude of observer.
        dt_minutes: Time resolution in minutes of solar calculations.
        """

        self.sTime      = sTime
        self.eTime      = eTime
        self.lat        = lat
        self.lon        = lon
        self.dt_minutes = dt_minutes
        self.data       = {}

        if not self.__check_parameters__():
            print('WARNING: Incomplete inputs to solarTimeseries(); will not perform calculations')

    def __check_parameters__(self):
        """
        Returns False if any of the sTime, eTime, lat, or lon are None.
        """
        if self.sTime is None or self.eTime is None \
                or self.lat is None or self.lon is None:
            return False
        else:
            return True

    def __calcSolarAzEls__(self):
        """
        Compute solar azimuths and elevations using parameters defined when object was created.

        Results are stored in a dataframe in self.data['solarAzEls']
        """
        sTime       = self.sTime
        eTime       = self.eTime
        lat         = self.lat
        lon         = self.lon
        dt_minutes  = self.dt_minutes

        # Create time range
        times = pd.date_range(start=sTime, end=eTime, freq=f'{dt_minutes}min')
        
        # Calculate solar position using pvlib
        solar_position = pvlib.solarposition.get_solarposition(times, lat, lon)
        
        # Extract zenith angles
        solarAzEls = pd.DataFrame(index=times)
        solarAzEls['els'] = solar_position['elevation']
        
        self.data['solarAzEls'] = solarAzEls

    def __calcSolarEclipse__(self):
        """
        Compute solar eclipse obscurations using parameters defined when object was created.

        Results are stored in a dataframe in self.data['solarEclipse']
        """
        sTime       = self.sTime
        eTime       = self.eTime
        lat         = self.lat
        lon         = self.lon
        dt_minutes  = self.dt_minutes

        solar_dt    = datetime.timedelta(minutes=dt_minutes)
        solar_times = [sTime]
        while solar_times[-1] < eTime:
            solar_times.append(solar_times[-1]+solar_dt)
        obsc        = eclipse_calc.calculate_obscuration(solar_times,lat,lon)
        df = pd.DataFrame({'obsc':obsc}, index = solar_times)
        self.data['solarEclipse'] = df

    def overlaySolarElevation(self,ax,ylim=(0,90),
            ylabel='Solar Elevation Angle',
            grid=False,color='0.6',ls='-.',lw=4,**kwargs):
        """
        Overplot the solar elevation on a timeseries axis object. The new data
        will be plotted on a twin axis created by ax.twinx()

        ax: Axis object of original timeseries. X-axis should use UTC datetimes.
        """
        if not self.__check_parameters__():
            print('WARNING: Incomplete inputs to solarTimeseries().')
            print('         Cannot overlay solar elevations angles.')
            return

        if 'solarAzEls' not in self.data:
            self.__calcSolarAzEls__()

        azEls   = self.data['solarAzEls']
        sza_xx  = azEls.index
        sza_yy  = azEls['els']

        ax_sza = ax.twinx()
        ax_sza.plot(sza_xx,sza_yy,color=color,ls=ls,lw=lw,**kwargs)
        ax_sza.set_ylabel(ylabel)
        ax_sza.set_ylim(ylim)
        ax_sza.grid(grid)

    def overlayEclipse(self,ax,ylim=(1.,0.),
            ylabel='Eclipse Obscuration',
            color='b',alpha=0.5,ls=':',lw=4,grid=False,
            spine_position=1.06,**kwargs):
        """
        Overplot the eclipse obscuration on a timeseries axis object. The new data
        will be plotted on a twin axis created by ax.twinx()

        ax: Axis object of original timeseries. X-axis should use UTC datetimes.
        spine_position: Position of spine in transAxes coordinates so. Default set to
            1.10 so that this spine does not overlap with Solar Elevation Angle.
        """
        if not self.__check_parameters__():
            print('WARNING: Incomplete inputs to solarTimeseries().')
            print('         Cannot overlay solar eclipse obscurations.')
            return

        if 'solarEclipse' not in self.data:
            self.__calcSolarEclipse__()

        solar_times = self.data['solarEclipse'].index
        obsc        = self.data['solarEclipse']['obsc']

        ax_ecl = ax.twinx()
        ax_ecl.plot(solar_times,obsc,color=color,alpha=alpha,ls=ls,lw=lw,**kwargs)
        ax_ecl.set_ylabel(ylabel)
        ax_ecl.set_ylim(ylim)
        ax_ecl.grid(grid)
        if spine_position is not None:
            ax_ecl.spines.right.set_position(("axes", spine_position))
