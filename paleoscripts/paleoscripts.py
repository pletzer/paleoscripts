import xarray as xr
import numpy as np

def pad_data(data_array):
    """
    Pad the data along the last axis (longitudes)
    :param data_array: instance of xarray.DataArray
    :returns a new DataArray with an additional row, containing the first row values
    """

    # extended shape, last dimension is lon
    x_shape = list(data_array.shape)
    x_shape[-1] += 1

    x_data = np.empty(x_shape, data_array.dtype())
    x_data[..., :-2] = data_array[:]
    x_data[..., -1] = x_data[..., 0]

    x_data_array = xr.DataArray(x_data, coords=data_array.coords[:-1] + [x_lon], \
                                dims=data_array.dims, name=data_array.name, attrs=data_array.attrs)
    return x_data_array

