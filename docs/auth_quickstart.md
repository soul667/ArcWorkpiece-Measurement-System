# 认证系统快速启动指南

## 自动安装 (推荐)

使用一键安装脚本：
```bash
# 使脚本可执行
chmod +x setup_auth.sh

# 运行安装脚本
./setup_auth.sh
```

脚本会自动：
- 安装所有依赖
- 启动MySQL数据库
- 初始化数据库和默认用户
- 准备前端环境

## 手动安装

如果需要手动安装，请按以下步骤操作：

### 1. 启动数据库
确保MySQL容器正在运行：
```bash
docker-compose up -d mysql
```

### 2. 初始化数据库
创建默认管理员账户：
```bash
cd UserInterface
python -m auth.init_db
```

### 3. 安装依赖

#### 后端依赖
在项目根目录下：
```bash
pip install -r requirements_auth.txt
```

#### 前端依赖
在antd-demo目录下：
```bash
cd antd-demo
chmod +x install_deps.sh
./install_deps.sh
```

## 启动服务

### 启动后端服务
在项目根目录下：
```bash
cd UserInterface
python fastapi_test.py
```

### 启动前端服务
新开一个终端，在antd-demo目录下：
```bash
npm start
```

## 登录系统

默认管理员账号：
- 用户名：admin
- 密码：admin123

访问：http://localhost:3000

## 功能验证

1. 登录功能：
   - 使用默认管理员账号登录
   - 验证登录成功后跳转到主界面
   - 验证右上角显示用户名

2. 会话保持：
   - 刷新页面后应保持登录状态
   - 验证localStorage中存储了token

3. 登出功能：
   - 点击用户名下拉菜单中的"退出登录"
   - 验证跳转回登录页面
   - 验证localStorage中的token被清除

## 常见问题

1. 模块未找到错误：
   ```
   Cannot find module 'axios'
   ```
   解决方案：重新运行依赖安装
   ```bash
   cd antd-demo
   ./install_deps.sh
   ```

2. 数据库连接问题：
   ```
   Error connecting to MySQL
   ```
   解决方案：检查MySQL容器状态
   ```bash
   docker-compose ps
   docker-compose logs mysql
   ```

3. 登录失败：
   - 检查数据库连接是否正常
   - 确认用户名和密码是否正确
   - 查看后端日志中的错误信息

4. 跨域问题：
   - 确认后端CORS设置正确
   - 检查请求URL是否正确

## 开发调试

1. 前端开发
   - 修改代码后自动热重载
   - 在浏览器开发工具中检查网络请求
   - 在Console面板查看错误信息

2. 后端开发
   - 查看fastapi.log日志文件
   - 访问 http://localhost:12345/docs 查看API文档
   - 使用数据库管理工具检查用户表

3. 环境变量
   - JWT_SECRET_KEY: JWT签名密钥
   - MYSQL_HOST: 数据库主机名
   - MYSQL_PORT: 数据库端口
   - MYSQL_USER: 数据库用户名
   - MYSQL_PASSWORD: 数据库密码
