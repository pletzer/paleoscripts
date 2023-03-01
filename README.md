# paleoscripts

A python module that contains a collection of postprocessing scripts

## How to access the source code

In a terminal, type
```
git clone https://github.com/pletzer/paleoscripts
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
where PROJECT is your project number and USER your user name. Copy the third string.

### Activate the environment

```
module purge
module load Miniconda3/4.8.2
conda deactivate
conda activate /nesi/project/PROJECT/USER/envs/PYTHON_KERNEL
```
(Note: your Miniconda version might be different, check the version in `~/.local/share/jupyter/kernels/PYTHON_KERNEL-kernel/wrapper.sh`.)

Then proceed to the next section.

## Install the package

Navigate to the top directory `paleoscripts` and type
```
pip install .
```

## Updating the package

Make sure you have the latest version installed. In the top `paleoscripts` directory type,
```
git pull origin main
```

Then, activate if need be the `conda` environment following the steps detailed [above](#activate-the-environment) and [reinstall](#install-the-package) the package

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


