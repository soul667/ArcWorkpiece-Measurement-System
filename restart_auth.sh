#!/bin/bash

echo "=== 重启认证系统 ==="

# 停止正在运行的服务
echo "停止现有服务..."
pkill -f "python.*fastapi_test.py"
pkill -f "npm.*start"

# 等待服务完全停止
sleep 2

# 清理旧的令牌
echo "清理旧的认证数据..."
rm -f UserInterface/fastapi.log
rm -f auth_init.log

# 初始化数据库
echo "初始化数据库..."
cd UserInterface
python -m auth.init_db
cd ..

# 启动后端服务
echo "启动后端服务..."
cd UserInterface
python fastapi_test.py &
cd ..

# 等待后端服务启动
echo "等待后端服务启动..."
sleep 5

# 启动前端服务
echo "启动前端服务..."
cd antd-demo
npm start &
cd ..

echo """
=== 服务已启动 ===

前端服务: http://localhost:3000
后端服务: http://localhost:9304

默认管理员账号:
用户名: admin
密码: admin123

提示：如遇到登录问题，请检查：
1. 数据库连接是否正常
2. 后端服务是否正常运行
3. 前端配置是否正确
"""

# 监控日志
echo "监控日志输出(Ctrl+C退出)..."
tail -f UserInterface/fastapi.log
