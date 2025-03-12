#!/bin/bash

echo "=== 设置脚本权限 ==="

# 主要脚本
chmod +x setup_auth.sh
chmod +x start_services.sh
chmod +x check_system.sh
chmod +x run_dev.sh
chmod +x restart_auth.sh

# 前端脚本
chmod +x antd-demo/install_deps.sh

# 修改文件夹权限
chmod 755 UserInterface/assets/temp 2>/dev/null || mkdir -p UserInterface/assets/temp && chmod 755 UserInterface/assets/temp

echo """
=== 权限设置完成 ===

可执行脚本:
- setup_auth.sh    : 初始化认证系统
- start_services.sh: 启动所有服务
- check_system.sh  : 检查系统状态
- run_dev.sh       : 启动开发环境
- restart_auth.sh  : 重启认证系统

前端脚本:
- antd-demo/install_deps.sh: 安装前端依赖

目录权限:
- UserInterface/assets/temp: 755 (读写执行)

使用方法:
1. 首次设置: ./setup_auth.sh
2. 启动服务: ./start_services.sh
3. 检查状态: ./check_system.sh
"""

# 验证权限
echo "验证脚本权限..."
for script in setup_auth.sh start_services.sh check_system.sh run_dev.sh restart_auth.sh antd-demo/install_deps.sh; do
    if [ -x "$script" ]; then
        echo "✓ $script 可执行"
    else
        echo "✗ $script 权限设置失败"
    fi
done

echo "验证目录权限..."
if [ -d "UserInterface/assets/temp" ] && [ $(stat -c "%a" UserInterface/assets/temp) = "755" ]; then
    echo "✓ UserInterface/assets/temp 权限正确"
else
    echo "✗ UserInterface/assets/temp 权限设置失败"
fi
