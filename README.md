# paleoscripts

A python module that contains a collection of postprocessing scripts

## Prerequisites

 - python
 - numpy
 - pytest
 - xarray
 - matplotlib
 - geocat-viz
 - cartopy

## How to access the source code

In a terminal, type
```
git clone https://github.com/pletzer/paleoscripts
```
 
## How to install the package

See below if you need to install this package within a Jupyter kernel running on NeSI.

In the top `paleoscripts` directory, type

```
pip install .
```

## How to install `paleoscripts` within a Jupyter kernel running on NeSI

### Identify the path to the conda environment

```
grep "conda activate" ~/.local/share/jupyter/kernels/PYTHON_KERNEL-kernel/wrapper.sh
```
where PYTHON_KERNEL is the name of the python kernel (e.g. pygeo). This command will return something like
```
conda activate /nesi/project/PROJECT/USER/envs/PYTHON_KERNEL
```
where PROJECT is your project number and USER your user name. Take note of the third string.


### Activate the environment

```
module purge
module load Miniconda3/4.8.2
conda deactivate
conda activate /nesi/project/PROJECT/USER/envs/PYTHON_KERNEL
```

### Install the package

Navigate to the top directory `paleoscripts` and type
```
pip install .
```

## How to test the package

Check that you can import the package

```
import paleoscripts
```

More extensive tests can be run inside the `paleoscripts` directory by typing
```
pytest
```


