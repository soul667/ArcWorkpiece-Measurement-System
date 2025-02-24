#!/bin/bash

# 查找并删除匹配的文件
echo "正在删除匹配的文件: axis_detection_random_test_*.png"
rm -f axis_detection_random_test_*.png
rm -f projection_*.png
rm -f axis_*.png


# 检查是否删除成功
if [ $? -eq 0 ]; then
    echo "文件删除成功。"
else
    echo "文件删除失败或未找到匹配的文件。"
fi