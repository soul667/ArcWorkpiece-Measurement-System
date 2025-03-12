#!/bin/bash

echo "=== 圆弧测量系统状态检查 ==="



# 检查后端服务
echo -n "检查后端服务: "
if curl -s http://localhost:9304/docs > /dev/null; then
    echo "运行中 ✓"
else
    echo "未运行 ✗"
    exit 1
fi

# 检查前端服务
echo -n "检查前端服务: "
if curl -s http://localhost:3000 > /dev/null; then
    echo "运行中 ✓"
else
    echo "未运行 ✗"
    exit 1
fi

# 检查数据库连接
echo -n "检查数据库连接: "
cd UserInterface
if python -c "
from auth.db import Database
db = Database()
if db.connect():
    print('连接成功 ✓')
else:
    print('连接失败 ✗')
    exit(1)
"; then
    true
else
    cd ..
    exit 1
fi
cd ..

# 验证认证服务
echo -n "验证认证服务: "
TOKEN=$(curl -s -X POST http://localhost:9304/auth/token \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=admin&password=admin123" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

if [ ! -z "$TOKEN" ]; then
    echo "正常 ✓"
else
    echo "异常 ✗"
    exit 1
fi

echo "
=== 系统状态 ===
✓ 所有服务运行正常

可用服务:
- 前端界面: http://localhost:3000
- 后端API: http://localhost:9304
- API文档: http://localhost:9304/docs
"

exit 0
