#!/usr/bin/env python

"""Tests for `paleoscripts` package."""


import pytest
import paleoscripts
import xarray as xr
import numpy as np

def test_apply_cyclic_padding():

    # number of cells
    nlat, nlon = 3, 4
    nlat1, nlon1 = nlat + 1, nlon + 1

    # create a data array of lat-lon
    lat = np.linspace(-90., 90, nlat1)

    dlon = 360/nlon
    lon = np.linspace(0., 360 - dlon, nlon)

    data = np.arange(0, nlat1 * nlon).reshape((nlat1, nlon))

    da = xr.DataArray(data, coords=[lat, lon], dims=['latitude', 'longitude'], name='temperature')

    x_da = paleoscripts.apply_cyclic_padding(da)

    assert np.all(x_da[..., 0] == x_da[..., -1])
