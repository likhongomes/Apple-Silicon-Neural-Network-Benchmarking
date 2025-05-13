# Apple-Silicon-Neural-Network-Benchmarking
Final Project for MSML605 Computational System in Machine Learning

## How to run the project

### Before opting to run the project, please make sure the Docker Daemon is running on your computer. Apple Silicon is mandatory to run this code.

Build the docker container first using the command
``docker build -t nn_mps_runner .``

Then run the docker container using the command
``docker run --rm -v "$PWD:/app" nn_mps_runner``