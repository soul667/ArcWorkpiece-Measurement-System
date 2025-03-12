# 圆弧工件测量系统

## 项目概述

圆弧工件测量系统是一个集成了用户认证、点云处理和测量功能的工业应用系统。

## 快速开始

### 1. 初始化设置

```bash
# 首次运行，设置权限和基本配置
chmod +x setup_initial.sh
./setup_initial.sh
```

### 2. 启动系统

```bash
# 启动所有服务
./start_services.sh
```

### 3. 访问系统

打开浏览器访问：http://localhost:3000

默认管理员账号：
- 用户名：admin
- 密码：admin123

## 系统功能

### 1. 用户认证
- 安全登录/登出
- JWT令牌认证
- 会话管理
- 权限控制

### 2. 点云处理
- 点云文件上传
- 点云数据预处理
- 区域选择和裁剪
- 点云可视化

### 3. 测量功能
- 圆弧参数测量
- 实时数据分析
- 结果导出

## 项目结构

```
.
├── UserInterface/        # 后端代码
│   ├── auth/            # 认证模块
│   ├── assets/          # 静态资源
│   └── fastapi_test.py  # 主服务器
│
├── antd-demo/           # 前端代码
│   ├── src/             # 源代码
│   ├── public/          # 公共资源
│   └── package.json     # 项目配置
│
├── docs/                # 文档
│   ├── auth_README.md   # 认证文档
│   └── auth_setup.md    # 认证设置指南
│
└── scripts/             # 工具脚本
    ├── setup_initial.sh # 初始化脚本
    ├── setup_auth.sh    # 认证设置脚本
    └── restart_auth.sh  # 认证重启脚本
```

## 开发指南

### 前端开发
```bash
cd antd-demo
npm start
```

### 后端开发
```bash
cd UserInterface
python fastapi_test.py
```

## 文档

- [认证系统说明](docs/auth_README.md)
- [快速启动指南](docs/auth_quickstart.md)
- [开发故障排除](docs/dev_troubleshooting.md)
- [标定说明](docs/标定.md)

## 常用命令

```bash
# 初始化系统
./setup_initial.sh

# 启动所有服务
./start_services.sh

# 检查系统状态
./check_system.sh

# 重启认证系统
./restart_auth.sh

# 启动开发环境
./run_dev.sh
```

## 系统要求

- Python 3.8+
- Node.js 14+
- MySQL 8.0+
- Docker & Docker Compose

## 故障排除

### 1. 权限问题
```bash
# 重新设置权限
./make_executable.sh
```

### 2. 认证问题
```bash
# 重启认证系统
./restart_auth.sh
```

### 3. 服务问题
```bash
# 检查系统状态
./check_system.sh
```

## 维护指南

1. 日志文件位置
   - 后端：`UserInterface/fastapi.log`
   - 认证：`auth_init.log`

2. 临时文件
   - 点云：`UserInterface/assets/temp/`
   - 前端：`antd-demo/build/`

3. 数据备份
   - 数据库：`docker exec mysql mysqldump...`
   - 配置文件：`.env`, `antd-demo/.env`

## 安全建议

1. 首次登录后修改默认密码
2. 定期更新系统和依赖
3. 启用HTTPS
4. 配置防火墙
5. 监控异常访问

## 许可证

版权所有 © 2025
