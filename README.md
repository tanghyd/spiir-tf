# SPIIR Machine Learning (ML) Environments

## Setup

First ensure Python is installed on your system.

For example, we might be running a simple Linux (Ubuntu) system where we install via apt:

    sudo apt install python3.7 python3.7-venv

We recommend installing Python in your working environment according to your user needs.

### OzStar

Below are the steps to reproduce the installation of this TensorFlow virtual environment on the OzStar system.

If you're working in a local environment with CUDA and Python3 already installed, skip to the [Virtual Environment](#venv_setup) section.

#### Load OzStar Environment Modules

    module load gcc/9.2.0 openmpi/4.0.2
    module load cudnn/8.0.4-cuda-11.0.2
    module load python/3.7.4

### <a href="venv_setup"></a>Install Virtual Environment

We can install a virtual environment using Python's virtualenv system as follows.

    python -m venv venv                         # we used python3.7.4
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel  # check `which pip` path
    pip install -r requirements.txt
    
#### Add Virtual Environment to Jupyter Notebook Kernels

In order to use this environment in a Jupyter notebook, we can add the virtual environment
to the list of user Jupyter kernels (this works remotely as well).

    python -m ipykernel install --user --name=venv  # consider changing --name

Sometimes issues may arise with some IDEs in remote SSH contexts where the environment is not
immediately added to the list of available kernels. In these instances we recommend resetting
the remote host instance and connecting again. For example the Command Pallete in VSCode
(Cntrl+Shift+P) can be used to Kill VS Sever on Host to force kill the remote instance.

Installed jupyter kernels can be viewed by calling `jupyter kernelspec list`.

### Symbolic Links

After setting up your virtual environment, a symbolic link can be created as follows:

    ln -s <installed_dir> <linked_dir>

For example, if you installed a virtual environment in a shared drive (e.g. /fred/oz016/<user> on OzStar)
and you want to create a link in your home directory (or vice versa), we might run the following command:

    ln -s /fred/oz016/<user>/project/venv ~/project/venv 
