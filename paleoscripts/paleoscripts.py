import xarray as xr
import numpy as np

def apply_cyclic_padding(data_array: xr.DataArray, coord_name: str='longitude', period: float=360.) -> xr.DataArray:
    """
    Apply cyclic padding to the data along a coordinate
    :param data_array: instance of xarray.DataArray
    :param cooord_name: the coordinate name
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

