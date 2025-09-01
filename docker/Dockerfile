# Use the official Python 3.12 slim image based on Debian
FROM python:3.12-slim

# Install GCC 12.1.0 and related build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 12 as the default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Verify GCC installation
RUN gcc --version && g++ --version

# Copy your local file into the image at the specified path
COPY txl-3.4.0-cp312-cp312-linux_x86_64.whl /root/

# Use a base image with Conda installed. The Miniconda3 image is a good choice.
#FROM quay.io/condaforge/linux-anvil-x86_64:alma9

# Set the working directory
#WORKDIR /home/conda/

# Create the Conda environment from a YAML file.
# We'll use a file named 'environment.yml'.
#COPY environment.yml .
#RUN conda env create -f environment.yml

# Activate the Conda environment
#ENV PYTHONPATH=/home/conda/txl/python/
#ENV LD_LIBRARY_PATH=/opt/conda/envs/txl/lib

# Copy your Python code into the container
#COPY requirements.txt .
#COPY txl-3.4.0-cp312-cp312-linux_x86_64.whl .
#COPY 02-flash-attention.py .

# The command to run your Python script
# Make sure to replace your_script.py with the name of your file
#CMD ["python", "python/txl/tutorials/02-flash-attention.py"]

#RUN sudo dnf install -y python3-pip
# Switch to conda user
#USER conda

# Set bash as shell and ensure conda is available
#SHELL ["/bin/bash", "-c"]

#CMD ["/bin/bash"]

## Initialize conda and use it
#RUN source /opt/conda/etc/profile.d/conda.sh && \
#    conda activate base && \
#    pip install --no-cache-dir -r requirements.txt && \
#    pip uninstall -y triton && \
#    pip install *.whl && \
#    python 02-flash-attention.py
