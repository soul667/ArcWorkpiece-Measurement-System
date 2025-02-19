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
    curl 
# Set the working directory
WORKDIR /workspace


RUN python -m venv /workspace/venv

# Copy current directory contents into the container

# Install Python dependencies
RUN pip install --no-cache-dir  pybind11

COPY ./requirements.txt /tmp/requirements.txt
# 安装依赖，这一层只有在 requirements.txt 变化时才会重建
RUN . /workspace/venv/bin/activate && pip install --no-cache-dir -r /tmp/requirements.txt

# RUN apt install nodejs npm -y && npm install -D vite
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN echo "source /workspace/venv/bin/activate" >> /root/.bashrc 

# 安装Typst

