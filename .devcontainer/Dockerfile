# Dockerfile for ArcWorkpiece Measurement System
FROM ascendai/python:3.10-ubuntu22.04

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    libpcl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace


RUN python -m venv /workspace/venv

# Copy current directory contents into the container

# Install Python dependencies
RUN pip install --no-cache-dir  pybind11

COPY ./requirements.txt /tmp/requirements.txt
# 安装依赖，这一层只有在 requirements.txt 变化时才会重建
RUN . /workspace/venv/bin/activate && pip install --no-cache-dir -r /tmp/requirements.txt

RUN echo "source /workspace/venv/bin/activate" >> /root/.bashrc
