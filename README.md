# paleoscripts

A python module that contains a collection of postprocessing scripts

![paleoscripts tests](https://github.com/pletzer/paleoscripts/actions/workflows/python-app.yml/badge.svg)

## How to access the source code

In a terminal, type
```
git clone https://github.com/pletzer/paleoscripts
```
 
## How to install `paleoscripts` within a Jupyter kernel running on NeSI

### Load your environment in the Terminal

Let's assume your environment is `pygeo`. You should have a file,
```
~/.local/share/jupyter/kernels/pygeo-kernel/wrapper.sh
```

This file contains the list of commands needed to load your environment. In my case, the command 
```
cat ~/.local/share/jupyter/kernels/pygeo-kernel/wrapper.sh
```
returns
```
#!/usr/bin/env bash

# load required modules here
module purge
module load Miniconda3

export PYTHONNOUSERSITE=True
unset I_MPI_PMI_LIBRARY

# activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh 
conda deactivate  # workaround for https://github.com/conda/conda/issues/9392
conda activate /nesi/project/nesi99999/pletzera/envs/pygeo

# run the kernel
exec python $@
```

Copy-paste the commands from `module purge` to `conda activate ...` in your Terminal. You should now have your environment activated.

### Install the package

Navigate to the top directory `paleoscripts` and type
```
pip install -e .
```
Option `-e` means editable, that any code changes will be picked without the need to rerun this command.

## Updating the package

Make sure you have the latest version installed. In the top `paleoscripts` directory type,
```
git pull origin main
```
(There is no need to re-install as we used `pip install -e .`.)

## How to test the package

Check that you can import the package

```
python -c "import paleoscripts"
```

More extensive tests can be run inside the `paleoscripts` directory by typing
```
pip install pytest # run this only once
pytest
```


