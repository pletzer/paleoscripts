#!/usr/bin/env python

"""Tests for `paleoscripts` package."""


import pytest
import paleoscripts
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


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
    lon_min, lon_max = -30., None
    lat_min, lat_max = -40., 50.
    xy_points = paleoscripts.find_points_where_field_is_max(da,\
        low_point=(lon_min, lat_min),\
        high_point=(lon_max, lat_max))

    # check
    da2 = da.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
    for i in range(xy_points.shape[0]):
        lo, la = xy_points[i, :]
        val = da.sel(longitude=lo, latitude=la)
        assert val >= da.sel(longitude=lo).max()




