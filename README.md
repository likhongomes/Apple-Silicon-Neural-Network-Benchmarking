# Apple-Silicon-Neural-Network-Benchmarking
Final Project for MSML605 Computational System in Machine Learning

## How to run the project

### Before opting to run the project, please make sure the Docker Daemon is running on your computer. Apple Silicon is mandatory to run this code.

1. Build the docker container first using the command
``docker build -t nn_mps_runner .``

2. Then run the docker container using the command
``docker run --rm -v "$PWD:/app" nn_mps_runner``

# CORRECTION
MPS is not available through docker in MacOS, therefore this project can't be run on the container, instead install the python library locally and run it

1. Run the command below to setup the environment
``python3 -m venv path/to/venv
    source path/to/venv/bin/activate
    python3 -m pip install -r requirements.txt``

2. Run the python script using the command below
``python3 NN_Mps.py``