import xarray as xr
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geocat.viz as gv
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import xskillscore as xs
import pandas as pd
import re
import os
import time


def gridded_data_to_excel(data_array, file_name, lon_name='longitude', lat_name='latitude'):
    """
    Save lon-lat gridded data to an excel file
    :param file_name: Excel file name
    :param lon_name: name of the longitude axis
    :param lat_name: name of the latitude axis
    """

    # convert the axes to 2d arrays
    lon2d, lat2d = np.meshgrid(data_array[lon_name], data_array[lat_name], indexing='xy')

    # create a data frame
    df = pd.DataFrame({lon_name: np.ravel(lon2d), lat_name: np.ravel(lat2d), data_array.name: np.ravel(data_array)})

    # save the data frame to an excel file
    df.to_excel(file_name)



def area_weighted_average(data_array: xr.DataArray,
                          xlim: tuple=(0., 360.), ylim: tuple=(-90., 90),
                          nx1: int=101, ny1: int=101) -> xr.DataArray:
    """
    Compute the area weighted average of a (time, latitude, longitude) or (latitude, longitude) field
    :param data_array: field
    :param xlim: tuple of (min, max) longitudes of the box
    :param ylim: tuple of (min, max) latitudes of the box
    :param nx1: number of target interpolation longitude points in longitudes
    :param ny1: number of target interpolation longitude points in latitudes
    :returns the area averaged value of the field 
    """

    # the target coordinates
    xs = np.linspace(xlim[0], xlim[1], nx1)
    ys = np.linspace(ylim[0], ylim[1], ny1)

    # remap the array to the target points
    data = data_array.interp(longitude=xs, latitude=ys)

    # compute the area elements
    area = np.tile(np.cos(xs * np.pi/180), (ys.shape[0], 1))
    total_area = area.sum()

    if len(data_array.shape) == 2:
        res = (data * area).sum()/total_area
        return res
    
    elif len(data_array.shape) == 3:
        
        nt = data_array.shape[0] # first index is time axis
        # the resulting array
        res = np.empty((nt,), data_array.dtype)

        # first axis is assumed to be time-like
        for itime in range(nt):
            res[itime] = (data[itime, ...] * area).sum()/total_area

        dim_name = data_array.dims[0]
        return xr.DataArray(res, dims=(dim_name,), coords=[data_array.coords[dim_name]])
    else:
        raise RuntimeError('array must be lat, lon or time, lat, lon')



def pearson_r(data_array1, xlim, ylim, data_array2, dim='year'):
    """
    Compute the Pearson correlation between the area averaged data_array1 in box (xlim, ylim) and data_array2
    :param data_arra1: reference array with axes (time, latitude, longitude) to be area averaged within (xlim, ylim) box
    :param xlim: tuple of (min, max) longitudes of the box
    :param ylim: tuple of (min, max) latitudes of the box
    :param data_array2: other array with axes (time, latitude, longitude)
    :param dim: dimension along which the correlation should be computed
    :returns an array of the same size as data_array2 representing the Pearson coefficient in the range -1 to 1
    """
    # compute the area weighted average over the box
    ref_values = area_weighted_average(data_array1, xlim, ylim)
    return xs.pearson_r(data_array2, ref_values, dim=dim)

def pearson_p(data_array1, xlim, ylim, data_array2, dim='year'):
    """
    Compute the Pearson p-value between the area averaged data_array1 in box (xlim, ylim) and data_array2
    :param data_arra1: reference array with axes (time, latitude, longitude) to be area averaged within (xlim, ylim) box
    :param xlim: tuple of (min, max) longitudes of the box
    :param ylim: tuple of (min, max) latitudes of the box
    :param data_array2: other array with axes (time, latitude, longitude)
    :param dim: dimension along which the correlation should be computed
    :returns an array of the same size as data_array2 representing the Pearson p-value in the range -1 to 1
    """
    # compute the area weighted average over the box
    ref_values = area_weighted_average(data_array1, xlim, ylim)
    return xs.pearson_r_p_value(data_array2, ref_values, dim=dim)

def correlation(data_array1, xlim, ylim, data_array2, dim='year'):
    """
    Compute the Pearson correlation between the area averaged data_array1 in box (xlim, ylim) and data_array2
    :param data_arra1: reference array with axes (time, latitude, longitude)
    :param xlim: tuple of (min, max) longitudes of the box
    :param ylim: tuple of (min, max) latitudes of the box
    :param data_array2: other array with axes (time, latitude, longitude)
    :param dim: dimension along which the correlation should be computed
    :returns an array of the same size as data_array2 representing the Pearson coefficient in the range -1 to 1
    """
    # compute the area weighted average over the box
    return pearson_r(data_array1, xlim, ylim, data_array2, dim=dim)


def rain_colormap(n = 32):
    n1 = n + 1
    x = np.linspace(0., 1., n1)
    cmap = np.empty((n1, 4), np.float32)
    # red
    cmap[:, 0] = 0.95*(1. - x)
    # green
    cmap[:, 1] = 1. - x**2
    # blue
    cmap[:, 2] = 0.95*(1. - x) + 0.5*x
    # opacity
    cmap[:, 3] = 1.
    return matplotlib.colors.ListedColormap(cmap)


def apply_cyclic_padding(data_array: xr.DataArray, coord_name: str='longitude', period: float=360.) -> xr.DataArray:
    """
    Apply cyclic padding to a data array along a given coordinate
    :param data_array: instance of xarray.DataArray
    :param coord_name: the coordinate name
    :param period: periodicity length
    :returns a new DataArray with an additional row, containing the first row values
    """

    # only supported for rectilinear coordinates
    nd = len(data_array.coords[coord_name].dims)
    if nd > 1:
        msg = f'Not a rectilinear grids, coordinate {coord_name} has {nd} dimensions (should have 1)'
        raise RuntimeError(msg)

    # find the index of the coordinate
    index_coord = data_array.dims.index(coord_name)

    coord = data_array.coords[coord_name]

    # the extended coordinate adds a periodicity length
    xcoord = xr.DataArray(list(coord) + [coord[0] + period], name=coord_name)

    ncoords = len(data_array.dims)

    # the shape of the original data needs to be extended by 1 in the direction of coord_name
    x_shape = list(data_array.shape)
    x_shape[index_coord] += 1

    # create a new array data object with the extended shape
    x_data = np.empty(x_shape, data_array.dtype)
    new_coords = [data_array.coords[data_array.dims[i]] for i in range(index_coord)] + \
                 [xcoord] + \
                 [data_array.coords[data_array.dims[i]] for i in range(index_coord + 1, ncoords)]
    x_data_array = xr.DataArray(x_data, coords=new_coords,\
                                dims=data_array.dims,\
                                name=data_array.name,\
                                attrs=data_array.attrs)

    # fill in the values of the new data array
    slab = [slice(None, None) for i in range(index_coord)] + \
           [slice(0, len(coord))] + \
           [slice(None, None) for i in range(index_coord + 1, ncoords)]
    # copy
    x_data_array[tuple(slab)] = data_array.data

    # fill in the last row
    slab_beg = [slice(None, None) for i in range(index_coord)] + \
           [0] + \
           [slice(None, None) for i in range(index_coord + 1, ncoords)]

    slab_end = [slice(None, None) for i in range(index_coord)] + \
           [-1] + \
           [slice(None, None) for i in range(index_coord + 1, ncoords)]
    # apply periodicity
    x_data_array[tuple(slab_end)] = x_data_array[tuple(slab_beg)]

    return x_data_array


def plot_contour(data_array: xr.DataArray,
                         central_longitude: float=0.,
                         title: str='',
                         levels=None,
                         xlim: tuple=(0., 360.),
                         ylim: tuple=(-90., 90.),
                         cmap: str='bwr',
                         figsize: tuple=(12, 8),
		         ticks=np.linspace(-1,1,11),
                         cbarorient: str='horizontal') -> None:
    """
    Create contour plot
    :param data_array: instance of xarray.DataArray
    :param central_longitude: mid longitude
    :param title: title
    :param levels: contour levels
    :param xlim: min/max longitude limits
    :param ylim: min/max latitude limits
    :param cmap: colormap name
    :param figsize: figure size
    :param ticks: ticks of the colorbar
    :param cbarorient: "vertical" or "horizontal"
    """

    fig = plt.figure(figsize=figsize) 

    proj = ccrs.PlateCarree(central_longitude=central_longitude)

    ax = plt.subplot(1, 1, 1, projection=proj)

    ax.set_extent(list(xlim) + list(ylim), crs=ccrs.PlateCarree())
    ax.coastlines()

    data = data_array.data
    if levels is None:
        levels = np.linspace(data.min(), data.max(), 21)

    cs = plt.contourf(data_array['longitude'], data_array['latitude'], data,
        transform=ccrs.PlateCarree(), levels=levels, cmap=cmap)


    # bounds = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]				 
    #plt.colorbar(orientation = cbarorient)

    # mycolormap = getattr(plt.cm, cmap)

    # Add colorbar for the  plot
    # norm = mcolors.BoundaryNorm(bounds, mycolormap.N)
    # mappable = plt.cm.ScalarMappable(norm=norm, cmap=mycolormap)

    plt.colorbar(cs,
	          ax=ax,
                  cmap=cmap,
                  label= '',
                  extendrect=True,
                  extendfrac=True,
                  ticks=ticks,
                  spacing='uniform',
                  orientation =cbarorient,
                  drawedges=False,
                  fraction=0.047) 
				 

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='black', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.left_labels = True
    gl.xlines = True

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 15, 'color': 'black'}
    gl.xlabel_style = {'size': 15, 'color': 'black', 'rotation': 0}

    if not title:
        title = data_array.name
    plt.title(title)


def linear_regression_coeffs(xy_pts: np.ndarray,
                             cooks_tol: float=4): 
    """
    Find the linear regression coefficients after removing outliers
    :param xy_pts: 2d array of [(x,y), ..] points
    :param cooks_tol: Cook's distance tolerance, points that are cooks_tol/n distance 
                      away will be removed. A good value is about 4.
    """

    x, y = xy_pts[:, 0], xy_pts[:, 1]

    # add constant to predictor variables
    x = sm.add_constant(x)

    # fit linear regression model
    model = sm.OLS(y, x).fit()

    # create instance of influence
    influence = model.get_influence()

    # obtain Cook's distance for each observation
    cooks = influence.cooks_distance[0]

    # remove the outliers 
    msk = (cooks > cooks_tol/len(x))
    xy_pts_filtered = xy_pts[~msk, :]

    # recompute the linear regressioon coefficients without the outliers
    res = linregress(xy_pts_filtered)

    return res

def linear_regression_coeffs_sklearn(x: np.array, y: np.array, poly_degree: int=1, 
                                     method: str='LinearRegression', **kw) -> dict:
    """
    Perform a linear regression using Sklearn
    :param x: x points
    :param y: y points
    :param poly_degree: degree of the polynomial
    :param method: regression method, e.g. LinearRegression, Ridge, Huber, etc.
    :param kw: additional keyword arguments to pass to the model's creator
    :returns a dictionary with entries 'linear_coef', 'intercept' and others
    """
    import importlib
    from sklearn.preprocessing import PolynomialFeatures
    
    # normalize the input
    xori = np.array(x)
    xmin = min(xori)
    xmax = max(xori)
    
    # make the feature run between 0 and 1
    xnrm = (xori - xmin)/(xmax - xmin)
    
    # create the polynomial features
    poly = PolynomialFeatures(degree=poly_degree)
    Xpoly = poly.fit_transform(np.array(xnrm).reshape(-1, 1))
    
    # import the requested model
    Model = getattr( importlib.import_module('sklearn.linear_model'), method )
    
    
    # create the model and fit it
    model = Model(fit_intercept=False, **kw)
    model.fit(Xpoly, y)
    yreg = model.predict(Xpoly)
    
    # rescale the coefficient. Note we're using fit_intercept=False so
    # coefficient [0] is the intercept and [1] the slope
    linear_coef = model.coef_[1]/(xmax - xmin)   
        
    res = {'reg_points': np.array(list(zip(x, yreg))),
           'MSE': np.mean((yreg - y)**2),
           'linear_coef': linear_coef,
           'intercept': model.coef_[0] - linear_coef*xmin,
           'ypredict': yreg,
    }
    return res


def find_points_where_field_is_extreme(data_array: xr.DataArray,\
				   extremum='max', 
                   lon_name='longitude', lat_name='latitude') -> np.ndarray:
    """
    Find the points where the field is either min or max
    :param data_array: instance of xarray.DataArray
    :param extremum: either 'min' or 'max'
    :param lon_name: name of the longitude coordinate
    :param lat_name: name of the latitude coordinate
    :returns a numpy array of [(lon, lat), ...] points
    """
    argextrem = np.argmax
    if extremum == 'min':
        argextrem = np.argmin
    lon = data_array.coords[lon_name].data
    lat = data_array.coords[lat_name].data

    xy_points = []
    for lo in lon:
        data = data_array.sel({lon_name: lo}).data
        j = argextrem(data)
        xy_points.append( (lo, lat[j],) )
    
    return np.array(xy_points)


def extract_season(data_array: xr.DataArray, season: str):
    """
    Return the data for a season
    :param data_array: instance of xarray.DataArray
    :param season: either 'djf', 'mam', 'jja' or 'son'
    """
    season2months = {'djf': np.array((12, 1, 2)),
                     'mam': np.array((3, 4, 5)),
                     'jja': np.array((6, 7, 8)),
                     'son': np.array((9, 10, 11))}
    if not season in season2months:
        raise RuntimeError(f"ERROR: {season} must be 'djf', 'mam', 'jja' or 'son'")
    da = data_array.sel(month=season2months[season])
    return da



def plot_linefit(data_array: xr.DataArray,
                 central_longitude: float=0.,
                 levels=None,
                 xlim: tuple=(0., 360.),
                 ylim: tuple=(-90., 90.),
                 fitxlim: tuple=(120., 200.),
                 fitylim: tuple=(-30, 0.),
                 cmap: str='bwr',
                 figsize: tuple=(12, 8)) -> matplotlib.axes.Axes:
    """
    Create contour plot with line fit
    :param data_array: 2D field,
    :param central_longitude: mid longitude
    :param levels: levels (automatic if None)
    :param xlim: x-bounds for the contour plot
    :param ylim: y-bounds for the contour plot
    :param fitxlim: x-bounds for the fitted line
    :param fitylim: y-bounds for the fitted line
    :param cmap: colour map
    :param figsize: figure size
    """
    if len(data_array.shape) != 2:
        raise RuntimeError(f'ERROR: in plot_linefit, array should have only two axes (got {len(data_array.shape)}!)')


    fig = plt.figure(figsize=figsize) 

    proj = ccrs.PlateCarree(central_longitude=central_longitude)

    ax = plt.subplot(1, 1, 1, projection=proj)

    ax.set_extent(list(xlim) + list(ylim), crs=ccrs.PlateCarree())
    ax.coastlines()

    data = data_array.data
    if levels is None:
        levels = np.linspace(data.min(), data.max(), 21)

    cs = plt.contourf(data_array['longitude'], data_array['latitude'], data,
        transform=ccrs.PlateCarree(), levels=levels, cmap=cmap)

    plt.colorbar(orientation = 'horizontal')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='black', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.left_labels = True
    gl.xlines = True

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'size': 15, 'color': 'gray', 'rotation': 45}

    #
    # linear regression
    #

    nx = int((fitxlim[1] - fitxlim[0])/1.)
    ny = int((fitylim[1] - fitylim[0])/1.)
    data_fit = data_array.interp(longitude=np.linspace(fitxlim[0], fitxlim[1], nx),
                                 latitude=np.linspace(fitylim[0], fitylim[1], ny))
    
    xy = find_points_where_field_is_extreme(data_fit, extremum='max')
    reg = linear_regression_coeffs(xy, cooks_tol=4)

    # plot the points
    ax.plot(xy[:, 0], xy[:, 1], 'k.', transform=ccrs.PlateCarree(), markersize=20)

    # plot the regression line
    ax.plot(fitxlim, reg.intercept + reg.slope * np.array(fitxlim), 'r--', transform=ccrs.PlateCarree())

    return ax


def extract_v_wind_at_pressure_levels(filenames: list, season: str, last_years, lon_min, lon_max, max_wind=1000.) -> np.ndarray:

    # check that the list of files is not empty
    if len(filenames) == 0:
        raise RuntimeError('ERROR: empty file list')

    # longitude resolution
    dlon = 1.0
    nlon = int( (lon_max - lon_min) / dlon )
    lon_vals = np.linspace(lon_min - dlon/2., lon_max + dlon/2., nlon)
    
    filenames.sort()
    lat = None
    pressures = np.zeros((len(filenames),), float)
    v_wind = []
    for fl in filenames:
        
        # extract the level
        bfl = os.path.basename(fl)
        m = re.search(r'sv(\d+)\_', bfl)
        level = -1
        if m:
            level = m.group(1)
        else:
            raise RuntimeError(f'ERROR: could not infer the pressure level from file name {bfl}')
        
        # extract the long_name of the variable v<level>
        ds = xr.open_dataset(fl)
        vname = 'v' + level
        
        if not hasattr(ds[vname], 'long_name'):
            raise RuntimeError(f'ERROR: variable {ds[vname]} must have a long_name attribute')
        
        m = re.search(r'wind at pressure\s*\=\s*(\d+\.\d+)', ds[vname].long_name)
        if m:
            pressure_value = float(m.group(1)) # in hPa or mbar
        else:
            raise RuntimeError(f'ERROR: could not infer the pressure value from {ds[vname].long_name}')
        
        pressures[int(level) - 1] = pressure_value
        
        # mask out invalid values
        v = (np.fabs(ds[vname]) < max_wind) * ds[vname]

        # I don't think we should apply periodic BCs as this would cause
        # the start /end points to be represented twice when computing the mean
        #v = apply_cyclic_padding(v, coord_name='longitude', period=360.0)

        # select values in lon_min, lon_max interval
        v = v.interp(longitude=lon_vals, method='cubic')
        #v = v.sel(longitude=lon_vals, method='nearest')
        
        v = extract_season(v, season)
        
        # assume order is year, month, latitude, longitude
        if isinstance(last_years, int) and last_years > 0:
            # average over the last 25 years only
            last_years = min(v.shape[0], last_years)
            vmean_level = v[-last_years:, ...].mean(dim=['longitude', 'month', 'year'])
        else:
            # use all the years
            vmean_level = v.mean(dim=['longitude', 'month', 'year'])

        v_wind.append(vmean_level)
        
        lat = v.latitude.data
        
    return np.array(v_wind), pressures, lat
   

def extract_u_wind_at_pressure_levels(filenames: list, season: str, last_years, lat_min, lat_max, max_wind=1000.) -> np.ndarray:

    # check that the list of files is not empty
    if len(filenames) == 0:
        raise RuntimeError('ERROR: empty file list')


    # latitude resolution
    dlat = 1.0
    nlat = int( (lat_max - lat_min) / dlat )
    lat_vals = np.linspace(lat_min - dlat/2., lat_max + dlat/2., nlat)
    
    filenames.sort()
    lat = None
    pressures = np.zeros((len(filenames),), float)
    u_wind = []
    for fl in filenames:
        
        # extract the level
        bfl = os.path.basename(fl)
        m = re.search(r'su(\d+)\_', bfl)
        level = -1
        if m:
            level = m.group(1)
        else:
            raise RuntimeError(f'ERROR: could not infer the pressure level from file name {bfl}')
        
        # extract the long_name of the variable u<level>
        ds = xr.open_dataset(fl)
        uname = 'u' + level
        
        if not hasattr(ds[uname], 'long_name'):
            raise RuntimeError(f'ERROR: variable {ds[uname]} must have a long_name attribute')
        
        m = re.search(r'wind at pressure\s*\=\s*(\d+\.\d+)', ds[uname].long_name)
        if m:
            pressure_value = float(m.group(1)) # in hPa or mbar
        else:
            raise RuntimeError(f'ERROR: could not infer the pressure value from {ds[uname].long_name}')
        
        pressures[int(level) - 1] = pressure_value
        
        # mask out invalid values
        u = (np.fabs(ds[uname]) < max_wind) * ds[uname]

        u = apply_cyclic_padding(u, coord_name='longitude', period=360.0)

        # select values in lat_min, lat_max interval
        u = u.interp(latitude=lat_vals, method='cubic')
        
        u = extract_season(u, season)
        
        # assume order is year, month, latitude, longitude
        if isinstance(last_years, int) and last_years > 0:
            # average over the last 25 years only
            last_years = min(u.shape[0], last_years)
            umean_level = u[-last_years:, ...].mean(dim=['latitude', 'month', 'year'])
        else:
            # use all the years
            umean_level = u.mean(dim=['latitude', 'month', 'year'])

        u_wind.append(umean_level)
        
        lon = u.longitude.data
        
    return np.array(u_wind), pressures, lon


def hadley_cell(filenames: list, season: str='djf', last_years=None,
                lon_min: float=120., lon_max:float=280., 
                aradius: float=6371e3, g: float=9.8) -> xr.DataArray:
    """
    Compute the Hadley cell
    :param filenames: list of file name paths containing the meridional velocity for each elevation, 
                      e.g. ["sv01_hosv1.nc.gz", "sv02_hosv1.nc.gz", ...]
    :param season: string, eg. 'djf'
    :param last_years: last number of years (None if taking all the years)
    :param lon_min: min longitude
    :param lon_max: max longitude
    :param aradius: earth's radius in m
    :param g: gravitional acceleration in m/s^2
    :returns an xarray DataArray with pressure levels and latitudes as axes
    """
    
    v_wind, pressures, lat = extract_v_wind_at_pressure_levels(filenames, season=season,
                                                             last_years=last_years,
                                                             lon_min=lon_min, lon_max=lon_max)
        
    # from the top of the atmosphere downwards
    pressures = np.flip(pressures)
    v_wind = np.flip(v_wind, axis=0) # CHECK that pressure is the first axis
    
    # compute dp from one level to the next
    dp = pressures[1:] - pressures[:-1]
    
    # compute the wind at mid pressure levels
    v_mid = 0.5*(v_wind[1:, :] + v_wind[:-1, :]) # pressure is axis 0
    
    # multiply wind by dp over each vertical interval
    for i in range(len(dp)):
        v_mid[i, :] *= dp[i]
    
    # integrate over levels, starting from the top and going downwards
    integral = np.cumsum( v_mid, axis=0, dtype=float )
    
    psi = np.empty(integral.shape, integral.dtype)
    
    # Hadley circulation, Eq(1) in https://wcd.copernicus.org/articles/3/625/2022/
    for j in range(len(lat)):
        psi[:, j] = (2 * np.pi * aradius * np.cos(lat[j]*np.pi/180.) / g) * integral[:, j]

    
    # create the DataArray and return it
    psia = xr.DataArray(data=psi, dims=('pressure', 'latitude'), \
        coords={'pressure': pressures[1:], # pressures are boundary values
                'latitude': lat,
                }
    )
    psia.coords['pressure'].attrs['units'] = 'hPa'
    psia.coords['latitude'].attrs['units'] = 'degree north'
    psia.attrs['history'] = f'Produced by paleoscripts.hadley_cell on {time.asctime()}'
    
    return psia

def walker_cell(filenames: list, season: str='djf', last_years=None,
                lat_min: float=-5, lat_max:float=5., 
                aradius: float=6371e3, g: float=9.8) -> xr.DataArray:
    """
    Compute the Walker cell
    :param filenames: list of file name paths containing the zonal velocity for each elevation, 
                      e.g. ["su01_hosv1.nc.gz", "su02_hosv1.nc.gz", ...]
    :param season: string, eg. 'djf'
    :param last_years: last number of years (None if taking all the years)
    :param lat_min: min latitude
    :param lat_max: max latitude
    :param aradius: earth's radius in m
    :param g: gravitional acceleration in m/s^2
    :returns an xarray DataArray with pressure levels and longitudes as axes
    """
    
    u_wind, pressures, lon = extract_u_wind_at_pressure_levels(filenames, season=season,
                                                             last_years=last_years,
                                                             lat_min=lat_min, lat_max=lat_max)
        
    # from the top of the atmosphere downwards
    pressures = np.flip(pressures)
    u_wind = np.flip(u_wind, axis=0) # CHECK that pressure is the first axis
    
    # compute dp from one level to the next
    dp = pressures[1:] - pressures[:-1]
    
    # compute the wind at mid pressure levels
    u_mid = 0.5*(u_wind[1:, :] + u_wind[:-1, :]) # pressure is axis 0
    
    # multiply wind by dp over each vertical interval
    for i in range(len(dp)):
        u_mid[i, :] *= dp[i]
    
    # integrate over levels, starting from the top and going downwards
    integral = np.cumsum( u_mid, axis=0, dtype=float )
    
    psi = np.empty(integral.shape, integral.dtype)
    
    # Walker circulation, Eq(4) in https://www.mdpi.com/2073-4433/14/2/397
    for i in range(len(lon)):
        psi[:, i] = (2 * np.pi * aradius / g) * integral[:, i]
    
    # create the DataArray and return it
    psia = xr.DataArray(data=psi, dims=('pressure', 'longitude'), \
        coords={'pressure': pressures[1:], # pressures are boundary values
                'longitude': lon,
                }
    )
    psia.coords['pressure'].attrs['units'] = 'hPa'
    psia.coords['longitude'].attrs['units'] = 'degree east'
    psia.attrs['history'] = f'Produced by paleoscripts.walker_cell on {time.asctime()}'
    
    return psia


def get_subtropical_high(lons: np.ndarray, lats: np.ndarray, psi_pacific: np.ndarray) -> float:
    """
    Given a positive function with a maximum in the domain:
    1) compite the zonal cross-section where the array is maximum
    2) select the longtiudes where the field is 90% of the maximum in the cross section
    3) return the mean longitudes where the field is > 90% of maximum in the cross section
    """
    psi_pacific_min = np.min(psi_pacific)
    psi_pacific_max = np.max(psi_pacific)
    # find the lat index of the max
    
    print(f'*** psi_pacific_max = {psi_pacific_max}')
    # latitude index where field is highest
    j_max = np.argmax(psi_pacific, axis=0)[0]
    print(f'*** j_max = {j_max}')
    # all the longitude indices in the zonal cross-section where field > 0.9 max
    i_90percent = np.where( psi_pacific[j_max, :] > psi_pacific_min + 0.9*(psi_pacific_max - psi_pacific_min) )[0]
    print(f'*** i_90percent = {i_90percent}')
    # mean of the selected longitudes
    mean_lon_max = np.mean(lons[i_90percent])
    
    return mean_lon_max, lats[j_max]
    
    