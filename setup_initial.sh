#!/bin/bash

echo "=== 初始化权限和基本设置 ==="

# 使所有脚本可执行
chmod +x make_executable.sh
chmod +x setup_auth.sh
chmod +x start_services.sh
chmod +x check_system.sh
chmod +x run_dev.sh
chmod +x restart_auth.sh
chmod +x antd-demo/install_deps.sh

# 创建必要的目录
mkdir -p UserInterface/assets/temp
chmod 755 UserInterface/assets/temp

# 复制环境配置文件
cp .env.example .env 2>/dev/null || true
cp antd-demo/.env.example antd-demo/.env 2>/dev/null || true

echo """
=== 初始化完成 ===

现在您可以运行以下命令：
1. 初始化系统：./setup_auth.sh
2. 启动服务：./start_services.sh
3. 检查状态：./check_system.sh

如需重启认证系统：
./restart_auth.sh

默认管理员账号：
用户名：admin
密码：admin123
"""

# 显示文件权限验证
echo -e "\n=== 权限验证 ==="
ls -l *.sh
ls -l antd-demo/*.sh
ls -ld UserInterface/assets/temp
