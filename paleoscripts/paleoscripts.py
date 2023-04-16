import xarray as xr
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geocat.viz as gv
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter


def apply_cyclic_padding(data_array: xr.DataArray, coord_name: str='longitude', period: float=360.) -> xr.DataArray:
    """
    Apply cyclic padding to a data array along a given coordinate
    :param data_array: instance of xarray.DataArray
    :param coord_name: the coordinate name
    :param period: periodicity length
    :returns a new DataArray with an additional row, containing the first row values
    """

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


def create_contourf_plot(data_array: xr.DataArray,
                         central_longitude: float=0.,
                         title: str='Temperature',
                         levels: np.array=np.linspace(200,320,60),
                         xlim: tuple=(0., 360.),
                         ylim: tuple=(-90., 90.),
                         cmap: str='bwr',
                         figsize: tuple=(12, 8)) -> None:
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
    """

    fig = plt.figure(figsize=figsize)
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax = plt.axes(projection=proj)

    # add land feature
    ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=1)


    gv.set_titles_and_labels(ax,
                             maintitle="",
                             maintitlefontsize=18,
                             lefttitle="",
                             lefttitlefontsize=18,
                             righttitle="",
                             righttitlefontsize=18)

    # Format tick labels as latitude and longitudes
    gv.add_lat_lon_ticklabels(ax=ax)

    # Use geocat-viz utility function to customize tick marks
    gv.set_axes_limits_and_ticks(ax,
                                 xlim=xlim,
                                 ylim=ylim,
                                 xticks=np.linspace(xlim[0], xlim[1], 7),
                                 yticks=np.linspace(ylim[0], ylim[1], 5))

    # Remove degree symbol from tick labels
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))

    # Add minor tick marks
    gv.add_major_minor_ticks(ax,
                             x_minor_per_major=4,
                             y_minor_per_major=4,
                             labelsize=14)

    # Make sure that tick marks are only on the left and bottom sides of subplot
    ax.tick_params('both', which='both', top=False, right=False)


    vmo_plot = data_array.plot.contourf(
                                 ax=ax,
                                 transform=proj,
                                 levels=levels,
                                 cmap=cmap,
                                 add_colorbar=False)  

    # Add and customize colorbar
    units = ''
    if hasattr(data_array, 'units'):
        units = data_array.units

    cbar_ticks = np.linspace(min(levels), max(levels), 11)

    plt.colorbar(ax=ax,
             mappable=vmo_plot,
             extendrect=False,
             extendfrac='Auto',
             label=units,
             ticks=cbar_ticks,
             drawedges=False,
             aspect=15,
             shrink=0.55)

    ax.coastlines()
    ax.set(xlabel=None)
    ax.set(ylabel=None)

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


def find_points_where_field_is_extreme(data_array: xr.DataArray,\
				   extremum='max') -> np.ndarray:
    """
    Find the points where the field is either min or max
    :param data_array: instance of xarray.DataArray
    :param extremum: either 'min' or 'max'
    :returns a numpy array of [(lon, lat), ...] points
    """
    argextrem = np.argmax
    if extremum == 'min':
	    argextrem = arg.argmin
    lon = data_array.coords['longitude'].data
    lat = data_array.coords['latitude'].data

    xy_points = []
    for lo in lon:
        data = data_array.sel(longitude=lo).data
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
                 levels=[],
                 xlim: tuple=(0., 360.),
                 ylim: tuple=(-90., 90.),
                 fitxlim: tuple=(120., 200.),
                 fitylim: tuple=(0., 30.),
                 cmap: str='bwr',
                 figsize: tuple=(12, 8)) -> matplotlib.axes.Axes:
    """
    Create contour plot with line fit
    :param data_array: 2D field,
    :param central_longitude: mid longitude
    :param levels: levels (automatic if empty array)
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
    if not levels:
        levels = np.linspace(data.min(), data.max(), 21)
    cs = plt.contourf(data_array['longitude'], data_array['latitude'], data,
        transform=ccrs.PlateCarree(), levels=levels, cmap=cmap)

    plt.colorbar(orientation = 'horizontal')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='black', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.left_labels = True
    gl.xlines = True

    nx = int( (xlim[1] - xlim[0]) / 4 )
    xticks = np.linspace(np.floor(xlim[0]), np.ceil(xlim[1]), nx + 1)
    # subtract 360 if larger than 180
    xticks = (xticks > 180)*(xticks - 360) + (xticks <= 180)*xticks
    gl.xlocator = mticker.FixedLocator(xticks)

    ny = int( (ylim[1] - ylim[0]) / 4 )
    yticks = np.linspace(np.floor(ylim[0]), np.ceil(ylim[1]), ny + 1)
    gl.ylocator = mticker.FixedLocator(yticks)

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'size': 15, 'color': 'gray'}

    #
    # linear regression
    #

    nx = int((fitxlim[1] - fitxlim[0])/1.)
    ny = int((fitylim[1] - fitylim[0])/1.)
    data_fit = data_array.interp(longitude=np.linspace(fitxlim[0], fitxlim[1], nx),
                                 latitude=np.linspace(fitylim[0], fitylim[1], ny))
    

    xy = find_points_where_field_is_extreme(data_fit, extremum='max')
    reg = linear_regression_coeffs(xy, cooks_tol=4)

    print(xy)

    # plot the points
    ax.plot(xy[:, 0], xy[:, 1], 'k.', transform=ccrs.PlateCarree(), markersize=20)

    # plot the regression line
    ax.plot(fitxlim, reg.intercept + reg.slope * np.array(fitxlim), 'r--', transform=ccrs.PlateCarree())

    return ax

