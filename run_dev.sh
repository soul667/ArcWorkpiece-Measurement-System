#!/bin/bash

echo "=== 启动开发环境 ==="

# 创建必要的目录
echo "创建必需的目录..."
mkdir -p UserInterface/assets/temp

# 设置脚本权限
echo "设置脚本权限..."
chmod +x make_executable.sh
./make_executable.sh

# 检查并启动MySQL
echo "检查MySQL状态..."
if ! docker ps | grep -q mysql; then
    echo "启动MySQL..."
    docker-compose up -d mysql
    echo "等待MySQL启动完成..."
    sleep 10
else
    echo "MySQL已在运行中"
fi

# 初始化数据库（如果需要）
echo "检查数据库初始化状态..."
if ! python -c "
from UserInterface.auth.db import Database
db = Database()
if db.connect():
    print('数据库连接正常')
else:
    exit(1)
"; then
    echo "初始化数据库..."
    cd UserInterface
    python -m auth.init_db
    cd ..
fi

# 安装前端依赖
echo "检查前端依赖..."
cd antd-demo
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    ./install_deps.sh
fi
cd ..

# 启动开发服务器
echo "
=== 启动开发服务器 ===

在新的终端窗口中运行前端开发服务器:
cd antd-demo && npm start

当前终端将启动后端服务器...
"

# 启动后端服务器
cd UserInterface
if [ ! -d "assets/temp" ]; then
    mkdir -p assets/temp
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/..
python fastapi_test.py
