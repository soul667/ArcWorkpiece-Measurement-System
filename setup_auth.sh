#!/bin/bash

echo "=== 设置认证系统 ==="

# 安装后端依赖
echo "正在安装后端依赖..."
pip install -r requirements_auth.txt

# 启动MySQL容器
echo "启动MySQL数据库..."
docker-compose up -d mysql

# 等待MySQL启动
echo "等待MySQL启动..."
sleep 10

# 初始化数据库
echo "初始化数据库..."
cd UserInterface
python -m auth.init_db
cd ..

# 安装前端依赖
echo "安装前端依赖..."
cd antd-demo
chmod +x install_deps.sh
./install_deps.sh

echo "=== 设置完成 ==="
echo
echo "现在你可以:"
echo "1. 启动后端服务: cd UserInterface && python fastapi_test.py"
echo "2. 启动前端服务: cd antd-demo && npm start"
echo
echo "默认管理员账号:"
echo "用户名: admin"
echo "密码: admin123"
