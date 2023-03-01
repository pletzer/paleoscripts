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


def create_contourf_plot(data_array, title='', units='', figsize=(12, 8)):

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude= -60))

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
                                 xlim=(-90, 120),
                                 ylim=(0, 90),
                                 xticks=(-90, -60, -30, 0, 30, 60, 90, 120),
                                 yticks=(0, 30, 60, 90))

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
                                 transform=ccrs.PlateCarree(),
                                 levels=np.linspace(200,320,60),
                                 cmap="bwr",
                                 add_colorbar=False)  

    # Add and customize colorbar
    cbar_ticks = np.arange(200, 320, 20)
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

    return vmo_plot

