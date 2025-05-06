# 开发环境故障排除指南

## 常见错误及解决方案

### 1. 找不到 node_modules/fsevents 目录
```
FileNotFoundError: [Errno 2] No such file or directory: '.../node_modules/fsevents'
```

解决方案:
```bash
# 重新安装前端依赖
cd antd-demo
rm -rf node_modules
./install_deps.sh
```

### 2. 认证失败 (401 Unauthorized)
```
INFO: 127.0.0.1:45828 - "POST /auth/token HTTP/1.1" 401 Unauthorized
```

可能原因和解决方案:
1. 数据库未正确初始化
   ```bash
   cd UserInterface
   python -m auth.init_db
   ```

2. 用户名密码错误
   - 检查是否使用默认账号：admin/admin123
   - 检查数据库中的用户表是否正确

3. JWT密钥问题
   - 检查 UserInterface/auth/service.py 中的 SECRET_KEY

### 3. 模块导入错误

```python
ModuleNotFoundError: No module named 'UserInterface'
```

解决方案:
```bash
# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 4. 数据库连接错误

```
Error connecting to MySQL
```

检查步骤:
1. 确认MySQL容器运行状态
   ```bash
   docker ps | grep mysql
   ```

2. 检查数据库日志
   ```bash
   docker-compose logs mysql
   ```

3. 验证数据库配置
   - 检查 UserInterface/auth/db.py 中的配置信息
   - 确保配置与 docker-compose.yml 中的设置匹配

### 5. 前端构建错误

```
Cannot find module 'axios'
```

解决方案:
```bash
cd antd-demo
npm install axios @ant-design/icons antd
```

### 6. 开发服务器启动失败

后端服务器:
```bash
# 检查端口占用
lsof -i :12345
# 如果端口被占用，终止相关进程
kill -9 <PID>
```

前端服务器:
```bash
# 检查端口占用
lsof -i :3000
# 如果端口被占用，终止相关进程
kill -9 <PID>
```

### 7. 跨域问题

检查项:
1. 后端CORS设置
   - 检查 UserInterface/fastapi_test.py 中的 CORS 中间件配置

2. 前端API请求
   - 检查 antd-demo/src/utils/axios.js 中的 baseURL 配置

### 8. 热重载不工作

后端:
1. 检查 reload_excludes 设置
   ```python
   # fastapi_test.py
   reload_excludes = ["node_modules", ".git", "__pycache__", ".pytest_cache"]
   ```

前端:
1. 确认 package.json 中的 scripts 配置
2. 检查是否有语法错误阻止了重新编译

## 开发工具

### 1. VS Code 扩展推荐
- Python
- React
- ESLint
- Prettier
- Docker

### 2. 调试工具
1. React Developer Tools
2. FastAPI Swagger UI (/docs)
3. Docker Desktop
4. MySQL Workbench

### 3. 日志查看
```bash
# 后端日志
tail -f UserInterface/fastapi.log

# Docker日志
docker-compose logs -f

# 前端日志
npm run start # 控制台输出
```

## 数据库维护

### 1. 连接数据库
```bash
docker exec -it mysql mysql -udev -pdevpass arcworkpiece
```

### 2. 备份数据
```bash
docker exec mysql mysqldump -udev -pdevpass arcworkpiece > backup.sql
```

### 3. 恢复数据
```bash
docker exec -i mysql mysql -udev -pdevpass arcworkpiece < backup.sql
```

## 注意事项

1. 代码提交前：
   - 运行所有测试
   - 检查代码格式
   - 更新文档

2. 环境变量：
   - 检查 .env 文件配置
   - 不要提交敏感信息

3. 安全性：
   - 定期更新依赖
   - 检查安全漏洞
   - 保护敏感数据
