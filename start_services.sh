#!/bin/bash

# Make scripts executable
chmod +x setup_auth.sh
chmod +x antd-demo/install_deps.sh

echo "=== 启动圆弧测量系统服务 ==="

# 检查数据库是否运行
echo "检查MySQL状态..."
if ! docker ps | grep -q mysql; then
    echo "MySQL未运行，正在启动..."
    docker-compose up -d mysql
    echo "等待MySQL启动..."
    sleep 10
fi

# 启动后端服务
echo "启动后端服务..."
cd UserInterface
python fastapi_test.py &
BACKEND_PID=$!
cd ..

# 启动前端服务
echo "启动前端服务..."
cd antd-demo
npm start &
FRONTEND_PID=$!
cd ..

echo "
=== 服务已启动 ===

前端服务: http://localhost:3000
后端服务: http://localhost:9304

API文档: http://localhost:9304/docs

默认管理员账号:
用户名: admin
密码: admin123

按 Ctrl+C 停止所有服务
"

# 等待任意子进程终止
wait -n

# 清理进程
kill $BACKEND_PID 2>/dev/null
kill $FRONTEND_PID 2>/dev/null

echo "
服务已停止
"
