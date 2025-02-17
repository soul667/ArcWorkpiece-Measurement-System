# Dockerfile for ArcWorkpiece Measurement System
FROM ubuntu:latest

# Install system dependencies and Python tools
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Upgrade pip and install Python dependencies
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

RUN /app/venv/bin/pip --no-cache-dir install pybind11
# 单独处理pyblind11

# Default command (can be adjusted as needed)
CMD ["bash"]
