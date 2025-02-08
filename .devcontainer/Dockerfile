# 使用官方的python运行时作为父映像
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# 将当前目录内容复制到 /工作区的容器中
COPY .. /workspace

# 安装其他任何python依赖性
RUN pip install --no-cache-dir -r requirements.txt
