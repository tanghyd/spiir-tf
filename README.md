# SPIIR Machine Learning (ML) Environments

## TensorFlow Environment Setup

First ensure Python is installed on your system.

For example, we might be running a simple Linux (Ubuntu) system where we install via apt:

    sudo apt install python3.7 python3.7-venv

We recommend installing Python in your working environment according to your user needs.

### <a href="venv_setup"></a>Virtual Environment

    python -m venv venv         # we used python3.7
    source venv/bin/activate    # ensure "which pip" returns "venv/bin/pip"
    pip install --upgrade pip setuptools wheel    # not required but ideal
    pip install -r requirements.txt
    
##### Add Virtual Environment to Jupyter Kernels

In order to use this environment in a Jupyter notebook, we can add the virtual environment
to the list of user Jupyter kernels (this works remotely as well).

    python -m ipykernel install --user --name=venv  # consider changing --name

Sometimes issues may arise with some IDEs in remote SSH contexts where the environment is not
immediately added to the list of available kernels. In these instances we recommend resetting
the remote host instance and connecting again. For example the Command Pallete in VSCode
(Cntrl+Shift+P) can be used to Kill VS Sever on Host to force kill the remote instance.

### OzStar

Below are the steps reproduce the installation of this TensorFlow ML HPC virtual environment on the OzStar system.

#### Load Environment Modules

    module load gcc/9.2.0 openmpi/4.0.2
    module load cudnn/8.0.4-cuda-11.0.2
    module load nccl/2.9.6                # probably optional
    module load python/3.7.4

#### Install Virtual Environment

To install the Python virtual environment, follow the same commands outlined in the above [Virtual Environment](#venv_setup) section.

#### Symbolic Link

After setting up your virtual environment, a symbolic link can be created as follows:

    ln -s <installed_dir> <linked_dir>

For example, if you installed a virtual environment in a shared drive (e.g. /fred/oz016)
and you want to create a link in your home directory where environments are stored in folders named `env/` as below, we might run the following command.

    ln -s /fred/oz016/<user>/project/venv ~/project/venv 
