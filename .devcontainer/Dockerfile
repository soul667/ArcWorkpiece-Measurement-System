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
    curl \
    unzip \
    fontconfig \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*


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

# Install Typst
RUN curl -LO https://github.com/typst/typst/releases/latest/download/typst-x86_64-unknown-linux-musl.tar.xz \
    && tar -xf typst-x86_64-unknown-linux-musl.tar.xz \
    && mv typst-x86_64-unknown-linux-musl/typst /usr/local/bin/ \
    && rm -rf typst-x86_64-unknown-linux-musl*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install some fonts
RUN mkdir -p /usr/share/fonts/truetype/noto \
    && curl -L https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf \
        -o /usr/share/fonts/truetype/noto/NotoSans-Regular.ttf \
    && curl -L https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf \
        -o /usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf \
    && fc-cache -f -v \
    && apt-get install fonts-noto-cjk

# RUN chown -R $USERNAME:$USERNAME /workspace
# # Switch to vscode user
# USER $USERNAME

# Set the working directory
WORKDIR /workspace
# Add venv activation to vscode user's bashrc
RUN echo "source /workspace/venv/bin/activate" >> ~/.bashrc

# 安装Typst

# pip install PyQt5

CMD ["bash"]
