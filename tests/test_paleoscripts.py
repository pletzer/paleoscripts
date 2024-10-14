#!/usr/bin/env python

"""Tests for `paleoscripts` package."""

import sys
print(sys.executable)
import pytest
import paleoscripts
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

DATA_DIR = Path(__file__).absolute().parent.parent / Path('data')


# Convenience functions used by some tests

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


def create_latlon_data_curvilinear(nlat, nlon):
    nlat1, nlon1 = nlat + 1, nlon + 1

    # create a data array with lat-lon cooordinates
    lat = np.linspace(-90., 90, nlat1)

    # create a lon coordinate, the last value (ie 360 deg) is missing
    dlon = 360/nlon
    lon = np.linspace(0., 360 - dlon, nlon)

    # create some data
    xx, yy = np.meshgrid(lon, lat)

    data = (np.sin((xx - 48)*np.pi/180.) * np.cos((yy - 0.05*xx)*np.pi/180.))**2


    da = xr.DataArray(data,
                      coords={'latitude': (['y', 'x'], yy),
                              'longitude': (['y', 'x'], xx)},
                      dims=['y', 'x'], name='fake_data')

    return da


# Tests start here
##################

def test_linear_regression_sklearn1():
    x = [1, 2, 3]
    y = [4, 5, 6]
    res = paleoscripts.linear_regression_coeffs_sklearn(x, y)
    # intercept
    assert abs(res['intercept'] - 3.0) < 1.e-10
    # slope
    assert abs(res['linear_coef'] - 1.0) < 1.e-10
    

def test_linear_regression_sklearn2():
    x = np.linspace(100, 150, 11)
    y = -20 - 0.01*x
    res = paleoscripts.linear_regression_coeffs_sklearn(x, y, method='LinearRegression', poly_degree=2)
    # intercept
    assert abs(res['intercept'] - (-20)) < 1.e-3
    # slope
    assert abs(res['linear_coef'] - (-0.01)) < 1.e-3



def test_gridded_data_to_excel():
    da = create_latlon_data(4, 8)
    paleoscripts.gridded_data_to_excel(da, 'fake_data.xlsx')



def test_area_weighted_average():

    nlat, nlon = 6, 12
    da = create_month_latlon_data(nlat, nlon)
    xlim = (10., 30.)
    ylim = (50., 80.)
    da_weighted = paleoscripts.area_weighted_average(da, xlim=xlim, ylim=ylim, nx1=11, ny1=21)
    assert da_weighted.min() >= da.min()
    assert da_weighted.max() <= da.max()


def test_area_weighted_average2():

    nlat, nlon = 6, 12
    da = create_latlon_data(nlat, nlon)
    xlim = (10., 30.)
    ylim = (50., 80.)
    da_weighted = paleoscripts.area_weighted_average(da, xlim=xlim, ylim=ylim, nx1=11, ny1=21)
    assert da_weighted.min() >= da.min()
    assert da_weighted.max() <= da.max()


def test_correlation():

    nlat, nlon = 6, 12
    da = create_month_latlon_data(nlat, nlon)
    xlim = (10., 30.)
    ylim = (50., 80.)
    res = paleoscripts.correlation(da, xlim, ylim, da, dim='month')
    assert np.all(res.shape == da.shape[1:])
    assert res.min() >= -1.0
    assert res.max() <= 1.0

    res2 = paleoscripts.pearson_r(da, xlim, ylim, da, dim='month')
    assert abs((res - res2).sum()) < 1.e-10
    pval = paleoscripts.pearson_p(da, xlim, ylim, da, dim='month')


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

@pytest.mark.xfail()
def test_apply_cyclic_padding_curvilinear():

    nlat, nlon = 3, 4
    da = create_latlon_data_curvilinear(nlat, nlon)
    # expected to fail
    x_da = paleoscripts.apply_cyclic_padding(da)




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

def test_hadley_cell():
    fnames = glob.glob(str(DATA_DIR) + '/sv*.nc')
    psi = paleoscripts.hadley_cell(fnames, season='djf', lon_min=0., lon_max=360.)
    assert len(psi.shape) == 2
    test_val = 2.1946e+10
    print(f'Hadley test value: {test_val:.4e} got {psi.sum().data:.4e}')
    assert abs(psi.sum() - test_val) < 1.e-3*abs(test_val)
    
    #import matplotlib.pyplot as plt
    # import xarray as xr
    # xr.plot.contourf(psi)
    # plt.gca().invert_yaxis()
    # plt.savefig('hadley_cell.png')

def test_hadley_cell2():
    fnames = glob.glob(str(DATA_DIR) + '/sv*.nc')
    psi = paleoscripts.hadley_cell(fnames, season='djf', lon_min=120., lon_max=280.)
    assert len(psi.shape) == 2
    test_val = 6.5794e+09
    print(f'Hadley test value: {test_val:.4e} got {psi.sum().data:.4e}')
    assert abs(psi.sum() - test_val) < 1.e-3*abs(test_val)


def test_walker_cell():
    fnames = glob.glob(str(DATA_DIR) + '/su*.nc')
    psi = paleoscripts.walker_cell(fnames, season='djf', lat_min=-90, lat_max=90)
    assert len(psi.shape) == 2
    test_val = 1.5325e+12
    print(f'Walker test value: {test_val:.4e} got {psi.sum().data:.4e}')
    assert abs(psi.sum() - test_val) < 1.e-3*abs(test_val)
    
    # import matplotlib.pyplot as plt
    # import xarray as xr
    # xr.plot.contourf(psi)
    # plt.gca().invert_yaxis()
    # plt.savefig('walker_cell.png')

def test_walker_cell2():
    fnames = glob.glob(str(DATA_DIR) + '/su*.nc')
    psi = paleoscripts.walker_cell(fnames, season='djf', lat_min=-5, lat_max=5.)
    assert len(psi.shape) == 2
    test_val = -4.5946e+11
    print(f'Walker test value: {test_val:.4e} got {psi.sum().data:.4e}')
    assert abs(psi.sum() - test_val) < 1.e-3*abs(test_val)

