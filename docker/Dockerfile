# Use a base image with Conda installed. The Miniconda3 image is a good choice.
FROM quay.io/condaforge/linux-anvil-x86_64:alma9

# Set the working directory
WORKDIR /home/conda/

# Create the Conda environment from a YAML file.
# We'll use a file named 'environment.yml'.
#COPY environment.yml .
#RUN conda env create -f environment.yml

# Activate the Conda environment
#ENV PYTHONPATH=/home/conda/txl/python/
#ENV LD_LIBRARY_PATH=/opt/conda/envs/txl/lib

# Copy your Python code into the container
COPY requirements.txt .
COPY txl-3.4.0-cp312-cp312-linux_x86_64.whl .
COPY 02-flash-attention.py .

# The command to run your Python script
# Make sure to replace your_script.py with the name of your file
#CMD ["python", "python/txl/tutorials/02-flash-attention.py"]

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install *.whl

CMD ["python", "02-fused-attention.py"]
