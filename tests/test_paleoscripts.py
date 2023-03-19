#!/usr/bin/env python

"""Tests for `paleoscripts` package."""


import pytest
import paleoscripts
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def create_month_latlon_data(nlat, nlon):

    nlat1, nlon1 = nlat + 1, nlon + 1

    # create month coordinate
    mon = np.arange(1, 13)

    # create lat coordinate
    lat = np.linspace(-90., 90, nlat1)

    # create a lon coordinate, the last value (ie 360 deg) is missing
    dlon = 360/nlon
    lon = np.linspace(0., 360, nlon1)


    # create some data
    data = np.random.rand(12, nlat1, nlon1)
    da = xr.DataArray(data, coords=[mon, lat, lon], dims=['month', 'latitude', 'longitude'], name='noise')

    return da    



def create_latlon_data(nlat, nlon):
    nlat1, nlon1 = nlat + 1, nlon + 1

    # create a data array with lat-lon cooordinates
    lat = np.linspace(-90., 90, nlat1)

    # create a lon coordinate, the last value (ie 360 deg) is missing
    dlon = 360/nlon
    lon = np.linspace(0., 360 - dlon, nlon)

    # create some data
    data = np.arange(0, nlat1 * nlon).reshape((nlat1, nlon)) / (nlat1 * nlon)
    da = xr.DataArray(data, coords=[lat, lon], dims=['latitude', 'longitude'], name='temperature')

    return da

def test_apply_cyclic_padding():

    nlat, nlon = 3, 4
    da = create_latlon_data(nlat, nlon)
    x_da = paleoscripts.apply_cyclic_padding(da)

    assert np.all(x_da[..., 0] == x_da[..., -1])


def test_create_contourf_plot():
    
    nlat, nlon = 10, 20
    da = create_latlon_data(nlat, nlon)
    x_da = paleoscripts.apply_cyclic_padding(da)
    p = paleoscripts.create_contourf_plot(da, title='toto',\
        levels=np.linspace(0., 1., 11),\
        xlim=(-180, 180), ylim=(-90,90))
    plt.savefig('toto.png')
    #plt.show()


def test_find_points_where_field_is_max():

    nlat, nlon = 10, 20
    da = create_latlon_data(nlat, nlon)
    lon_min, lon_max = -30., None # None means max value, whatever it is
    lat_min, lat_max = -40., 50.
    xlim = (lon_min, lon_max)
    ylim = (lat_min, lat_max)

    xy_points = paleoscripts.find_points_where_field_is_max(da,\
        xlim=xlim, ylim=ylim)

    # check that the points lie within the box
    for lo, la in xy_points:
        assert lon_min <= lo
        if lon_max:
            assert lo <= lon_max
        assert lat_min <= la
        if lat_max:
            assert la <= lat_max

    # subset the data
    da2 = da.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))

    # check that we found the max value along lats for each lon
    for i in range(xy_points.shape[0]):
        lo, la = xy_points[i, :]
        val = da.sel(longitude=lo, latitude=la)
        da3 = da2.sel(longitude=lo)
        assert np.all(da3.data <= val.data)


def test_extract_season():

    da = create_month_latlon_data(nlat=4, nlon=8)
    da_djf = paleoscripts.extract_season(da, season='djf')
    assert da_djf.shape[0] == 3
    assert np.all(da_djf[0,...] == da[11, ...])
    assert np.all(da_djf[1,...] == da[0, ...])
    assert np.all(da_djf[2,...] == da[1, ...])

    for season in 'mam', 'jja', 'son':
        da_season = paleoscripts.extract_season(da, season=season)
        assert da_season.shape[0] == 3


    





