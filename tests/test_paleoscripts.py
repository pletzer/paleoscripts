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
    xx, yy = np.meshgrid(lon, lat)

    data = (np.sin((xx - 48)*np.pi/180.) * np.cos((yy - 0.05*xx)*np.pi/180.))**2


    da = xr.DataArray(data, coords=[lat, lon], \
                      dims=['latitude', 'longitude'], name='fake_data')

    return da


def test_area_weighted_average():

    nlat, nlon = 6, 12
    da = create_month_latlon_data(nlat, nlon)
    xlim = (10., 30.)
    ylim = (50., 80.)
    da_weighted = paleoscripts.area_weighted_average(da, xlim=xlim, ylim=ylim, nx1=11, ny1=21)
    assert da_weighted.min() >= da.min()
    assert da_weighted.max() <= da.max()


def test_rain_colormap():

    nlat, nlon = 180, 360
    da = create_latlon_data(nlat, nlon)
    da = paleoscripts.apply_cyclic_padding(da)
    x = da['longitude'][:]
    y = da['latitude'][:]
    xx, yy = np.meshgrid(x, y)

    cm = paleoscripts.rain_colormap()
    p = plt.contourf(xx, yy, da, cmap=cm)
    plt.colorbar(p)
    plt.savefig('test_rain_colormap.png')


def test_plot_linefit():

    nlat, nlon = 180, 360
    da = create_latlon_data(nlat, nlon)
    da = paleoscripts.apply_cyclic_padding(da)
    x = da['longitude'][:]
    y = da['latitude'][:]
    xx, yy = np.meshgrid(x, y)

    ax = paleoscripts.plot_linefit(da, central_longitude=180.,
                         xlim=(100, 300), ylim=(-60., 40.),
                         fitxlim=(130., 190.), fitylim=(3., 25.),
                         cmap='bwr', figsize=(12, 8))
    plt.savefig('test_plot_linefit.png')


def test_apply_cyclic_padding():

    nlat, nlon = 3, 4
    da = create_latlon_data(nlat, nlon)
    x_da = paleoscripts.apply_cyclic_padding(da)

    assert np.all(x_da[..., 0] == x_da[..., -1])


def test_plot_contour():
    
    nlat, nlon = 10, 20
    da = create_latlon_data(nlat, nlon)
    x_da = paleoscripts.apply_cyclic_padding(da)
    p = paleoscripts.plot_contour(x_da, title='',\
        levels=np.linspace(da.min(), da.max(), 11),\
        xlim=(-180, 180), ylim=(-90,90))
    plt.savefig('test_plot_contour.png')


def test_find_points_where_field_is_extreme():

    nlat, nlon = 10, 20
    da = create_latlon_data(nlat, nlon)
    lon_min, lon_max = -30., None # None means max value, whatever it is
    lat_min, lat_max = -40., 50.

    # subset the data
    da_box = da.sel(longitude=slice(lon_min, lon_max),
                 latitude=slice(lat_min, lat_max))

    xy_points = paleoscripts.find_points_where_field_is_extreme(da_box)

    # check that the points lie within the box
    for lo, la in xy_points:
        assert lon_min <= lo
        if lon_max:
            assert lo <= lon_max
        assert lat_min <= la
        if lat_max:
            assert la <= lat_max

    # check that we found the max value along lats for each lon
    for i in range(xy_points.shape[0]):
        lo, la = xy_points[i, :]
        val = da.sel(longitude=lo, latitude=la)
        da3 = da_box.sel(longitude=lo)
        assert np.all(da3.data <= val.data)


def test_linear_regression1():
    
    # nothing should be removed
    xy = np.array([(0., 1.), (1., 3.), (2., 5.)])
    res = paleoscripts.linear_regression_coeffs(xy, cooks_tol=2.0)
    assert res.intercept == 1.0
    assert res.slope == 2.0


def test_linear_regression2():
    
    # remove the last point
    xy = np.array([(0., 1.), (1., 3.), (2., 5.), (3., 0.)])
    res = paleoscripts.linear_regression_coeffs(xy, cooks_tol=8.0)
    assert res.intercept == 1.0
    assert res.slope == 2.0


def test_linear_regression3():

    # lots of points
    n = 100
    a = 1.0
    b = 2.0
    xy = np.array([(x, a + b*x) for x in np.linspace(0., 10., n)])

    # create some outliers
    xy[0, :] = xy[0,0], 2.0 + 3*xy[0, 0]
    res = paleoscripts.linear_regression_coeffs(xy, cooks_tol=4.0)
    
    assert abs(res.intercept - a) < 1.e-10
    assert abs(res.slope - b) < 1.e-10


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


    





