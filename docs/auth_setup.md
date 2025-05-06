# 认证系统设置指南

## 认证系统概述

认证系统提供了基于JWT的用户认证机制，支持：
- 用户登录/登出
- 会话管理
- API访问控制
- 密码加密存储

## 默认账户

系统初始化时会创建一个默认管理员账户：
- 用户名：admin
- 密码：admin123

这个账户可以用来首次登录系统。出于安全考虑，建议在首次登录后修改密码。

## 系统配置

### 配置文件

主要配置文件位于：
- 前端：`antd-demo/.env`
- 后端：`.env`

### 环境变量

后端环境变量：
```bash
# JWT配置
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# 数据库配置
MYSQL_DATABASE=arcworkpiece
MYSQL_USER=dev
MYSQL_PASSWORD=devpass
```

前端环境变量：
```bash
# API配置
REACT_APP_API_URL=http://localhost:12345
REACT_APP_TOKEN_KEY=arc_workpiece_token
```

## 认证流程

1. 用户登录
   - 访问登录页面
   - 输入用户名和密码
   - 系统验证凭据
   - 成功后返回JWT令牌

2. API认证
   - 前端存储JWT令牌
   - 请求时自动附加令牌
   - 后端验证令牌有效性

3. 会话管理
   - 令牌30分钟后过期
   - 自动登出过期会话
   - 清理登录状态

## 安全措施

### 密码安全
- bcrypt加密存储
- 禁止明文传输
- 密码策略由后端强制执行

### 令牌安全
- 签名验证
- 过期时间控制
- 安全存储

### API安全
- CORS保护
- 令牌验证
- 错误处理

## 开发指南

### 添加新用户
```python
from auth.service import AuthService

auth_service = AuthService()
auth_service.create_user("username", "password")
```

### 验证用户
```python
user = auth_service.authenticate_user("username", "password")
if user:
    print("认证成功")
```

### 创建令牌
```python
token = auth_service.create_access_token({"sub": username})
```

## 故障排除

### 登录失败

1. 检查用户名和密码
   - 确认输入正确
   - 区分大小写

2. 检查数据库连接
   ```bash
   docker-compose ps
   docker-compose logs mysql
   ```

3. 检查日志
   ```bash
   tail -f UserInterface/fastapi.log
   ```

### 令牌问题

1. 令牌过期
   - 检查系统时间同步
   - 验证过期时间设置

2. 令牌无效
   - 检查JWT密钥配置
   - 验证令牌格式

## 生产环境部署

1. 修改默认密码
   - 立即修改默认管理员密码
   - 设置强密码策略

2. 配置HTTPS
   - 启用SSL/TLS
   - 配置安全证书

3. 环境隔离
   - 使用环境变量
   - 分离配置文件

## 维护指南

1. 定期维护
   - 更新密钥
   - 清理过期会话
   - 审查访问日志

2. 安全更新
   - 更新依赖包
   - 应用安全补丁
   - 监控安全公告

## 备注

- 密码验证：由后端完成，前端不做强制验证
- 登录限制：可配置登录失败次数限制
- 日志记录：记录所有认证相关操作
