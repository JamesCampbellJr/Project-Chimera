# /Dockerfile

# Use an official NVIDIA CUDA runtime image as a parent image.
# This ensures that CUDA and NVIDIA drivers are correctly set up.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including Python 3.11, pip, and audio libraries
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# We install torch separately to ensure the correct CUDA version is used.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code
COPY . .

# Command to run the application
CMD ["python3", "main.py"]