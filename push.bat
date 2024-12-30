@echo off
REM 设置代理
set http_proxy=http://127.0.0.1:7890
set https_proxy=http://127.0.0.1:7890

REM 验证代理是否生效
echo 当前 http_proxy 设置为：%http_proxy%
echo 当前 https_proxy 设置为：%https_proxy%

REM 推送代码到远程仓库
git push origin main

REM 等待用户查看结果后关闭窗口
pause
