import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
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


def create_contourf_plot(data_array: xr.DataArray,\
                         central_longitude: float=0.,
                         title: str='Temperature',\
                         levels: np.array=np.linspace(200,320,60),
                         xlim: tuple=(0., 360.),
                         ylim: tuple=(-90., 90.),
                         cmap: str='bwr',\
                         figsize: tuple=(12, 8)) -> None:
    """
    Create contour plot
    :param data_array: instance of xarray.DataArray
    :param central_longitude mid longitude
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


def find_points_where_field_is_max(data_array: xr.DataArray,\
                                   xlim: tuple=(0., 360.),\
                                   ylim: tuple=(-90, 90.)) -> np.ndarray:
    """
    Find the points where the field is max
    :param data_array: instance of xarray.DataArray
    :param xlim: min/max values of longitudes
    :param ylim: min/max values of latitudes
    :returns a numpy array of [(lon, lat), ...] points
    """
    da = data_array.sel(\
         longitude = slice(xlim[0], xlim[1]),\
         latitude = slice(ylim[0], ylim[1]) \
         )
    lon = da.coords['longitude'].data
    lat = da.coords['latitude'].data

    xy_points = []
    for lo in lon:
        data = da.sel(longitude=lo).data
        j = np.argmax(data)
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





