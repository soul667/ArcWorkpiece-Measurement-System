# 圆弧工件测量系统认证模块

## 快速开始

### 一键设置（推荐）

1. 设置权限并初始化
```bash
# 设置脚本权限
chmod +x make_executable.sh
./make_executable.sh

# 初始化系统
./setup_auth.sh
```

2. 启动服务
```bash
./start_services.sh
```

### 手动设置

详细的手动设置步骤请参考 [auth_quickstart.md](auth_quickstart.md)。

## 重启或重置系统

如果需要重启认证系统：
```bash
./restart_auth.sh
```

## 默认账户

系统初始化时会自动创建管理员账户：
- 用户名：admin
- 密码：admin123

建议在首次登录后修改默认密码。

## 系统结构

### 后端组件
- FastAPI认证服务
- MySQL用户数据库
- JWT令牌管理
- bcrypt密码加密

### 前端组件
- React登录界面
- JWT令牌管理
- 会话状态维护
- 请求拦截器

## 开发指南

### 前端开发

1. 安装依赖
```bash
cd antd-demo
./install_deps.sh
```

2. 启动开发服务器
```bash
npm start
```

### 后端开发

1. 安装依赖
```bash
pip install -r requirements_auth.txt
```

2. 启动开发服务器
```bash
cd UserInterface
python fastapi_test.py
```

## 配置说明

### 环境变量

项目使用两个环境配置文件：
- `.env`：后端配置
- `antd-demo/.env`：前端配置

配置示例请参考各自目录下的 `.env.example` 文件。

### 关键配置项

1. 后端配置：
   - JWT密钥和过期时间
   - 数据库连接信息
   - 服务器端口

2. 前端配置：
   - API地址
   - 令牌存储键名
   - 超时设置

## 故障排除

### 登录问题
1. 密码验证失败
   - 检查输入是否正确
   - 确认数据库连接正常
   - 查看后端日志

2. 无法连接服务器
   - 检查服务是否运行
   - 验证端口配置
   - 检查网络连接

### 开发问题
1. 依赖安装失败
   ```bash
   # 重新安装前端依赖
   cd antd-demo
   rm -rf node_modules
   ./install_deps.sh
   ```

2. 数据库问题
   ```bash
   # 重置数据库
   cd UserInterface
   python -m auth.init_db
   ```

## 安全建议

1. 生产环境配置
   - 使用强密钥
   - 启用HTTPS
   - 设置适当的CORS策略

2. 密码策略
   - 后端强制密码复杂度
   - 定期更换密码
   - 限制登录失败次数

3. 数据保护
   - 加密敏感信息
   - 定期备份数据
   - 监控异常访问

## 维护指南

1. 日常维护
   - 检查日志文件
   - 清理临时文件
   - 更新依赖包

2. 性能优化
   - 监控响应时间
   - 优化数据库查询
   - 清理过期会话

3. 安全更新
   - 更新安全补丁
   - 轮换密钥
   - 审查访问日志

## 贡献指南

1. 提交代码
   - 遵循代码规范
   - 添加单元测试
   - 更新文档

2. 问题报告
   - 提供复现步骤
   - 附加相关日志
   - 描述环境信息

## 许可证

版权所有 © 2025
