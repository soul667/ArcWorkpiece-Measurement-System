#!/bin/bash

# 启动前端服务
start_frontend() {
    export NODE_OPTIONS=--openssl-legacy-provider && cd antd-demo && npm run start
}

# 启动后端服务
start_backend() {
    uvicorn UserInterface.fastapi_test:app --reload --host 0.0.0.0 --port 9304
}

# 设置脚本为可执行
chmod +x "$0"

# 静默模式运行（无界面输出）
run_silent() {
    nohup bash -c "export NODE_OPTIONS=--openssl-legacy-provider && cd $(dirname "$0")/antd-demo && npm run start" > /dev/null 2>&1 &
    nohup bash -c "cd $(dirname "$0") && uvicorn UserInterface.fastapi_test:app --reload --host 0.0.0.0 --port 9304" > /dev/null 2>&1 &
    echo "服务已静默启动，前端运行于3000端口，后端运行于9304端口"
}

# 显示模式运行（新窗口显示输出）
run_display() {
    if command -v xterm >/dev/null 2>&1; then
        xterm -e "export NODE_OPTIONS=--openssl-legacy-provider && cd $(dirname "$0")/antd-demo && npm run start" &
        xterm -e "cd $(dirname "$0") && uvicorn UserInterface.fastapi_test:app --reload --host 0.0.0.0 --port 9304" &
        echo "已在新窗口启动服务："
        echo "前端 - http://localhost:3000"
        echo "后端 - http://localhost:9304"
    else
        # 无xterm时在当前终端运行
        echo "未找到xterm，将在当前终端运行" >&2
        (cd $(dirname "$0")/antd-demo && export NODE_OPTIONS=--openssl-legacy-provider && npm run start) &
        (cd $(dirname "$0") && uvicorn UserInterface.fastapi_test:app --reload --host 0.0.0.0 --port 9304) &
    fi
}

# 根据参数选择运行模式
if [ "$1" == "silent" ]; then
    run_silent
else
    run_display
fi

# 使用说明提示
echo ""
echo "操作提示："
echo "1. 静默模式启动：$0 silent"
echo "2. 显示模式启动：$0"
echo "3. 前端地址：http://localhost:3000"
echo "4. 后端地址：http://localhost:9304/docs"