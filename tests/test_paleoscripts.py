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
        levels=np.linspace(0., 1., 11))
    plt.show()
    plt.savefig('toto.png')


