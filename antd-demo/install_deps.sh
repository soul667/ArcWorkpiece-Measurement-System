#!/bin/bash

echo "=== 安装前端依赖 ==="

# 检查是否安装了npm
if ! command -v npm &> /dev/null; then
    echo "错误: 未找到npm. 请先安装Node.js和npm."
    exit 1
fi

# 清理旧的依赖
echo "清理旧的依赖..."
rm -rf node_modules
rm -f package-lock.json

# 安装依赖
echo "安装依赖..."
npm install --legacy-peer-deps

# 安装开发工具
echo "安装开发工具..."
npm install --save-dev \
    @welldone-software/why-did-you-render \
    cross-env \
    prettier \
    source-map-explorer \
    serve

# 验证安装
echo "验证安装..."
if [ -d "node_modules" ]; then
    echo "✓ node_modules 目录存在"
    if [ -f "node_modules/react/package.json" ]; then
        echo "✓ React 已安装"
        if [ -f "node_modules/antd/package.json" ]; then
            echo "✓ Ant Design 已安装"
            if [ -f "node_modules/axios/package.json" ]; then
                echo "✓ Axios 已安装"
                echo "=== 安装成功 ==="
                echo "
你现在可以运行:
  npm start     - 启动开发服务器
  npm run build - 构建生产版本
  npm test     - 运行测试
"
                exit 0
            fi
        fi
    fi
fi

echo "安装似乎出现问题，请检查错误信息"
exit 1
