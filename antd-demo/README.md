# 圆弧工件测量系统 - 前端

## 项目概述

本项目是圆弧工件测量系统的前端部分，基于React和Ant Design构建，提供了用户认证、点云处理和测量功能的界面实现。

## 技术栈

- React 18
- Ant Design 5
- Axios
- Three.js
- JWT认证
- CSS-in-JS

## 开发环境设置

### 1. 安装依赖

```bash
# 设置脚本可执行权限
chmod +x install_deps.sh

# 运行安装脚本
./install_deps.sh
```

### 2. 环境配置

```bash
# 创建环境配置文件
cp .env.example .env

# 根据需要修改配置
vim .env
```

### 3. 启动开发服务器

```bash
npm start
```

服务器将在 http://localhost:3000 启动

## 项目结构

```
src/
├── components/           # React组件
│   ├── AuthenticatedApp.js    # 认证根组件
│   ├── ErrorBoundary.js       # 错误边界
│   ├── LoadingScreen.js       # 加载界面
│   └── LoginComponent.js      # 登录组件
├── utils/               # 工具函数
│   └── axios.js        # API请求配置
├── config.js           # 全局配置
├── App.js             # 应用入口
└── index.js           # 渲染入口
```

## 可用脚本

- `npm start` - 启动开发服务器
- `npm run build` - 构建生产版本
- `npm test` - 运行测试
- `npm run lint` - 代码检查
- `npm run format` - 代码格式化

## 开发指南

### 代码风格

项目使用Prettier进行代码格式化，配置文件位于`.prettierrc`。运行以下命令格式化代码：

```bash
npm run format
```

### 环境变量

开发时可用的环境变量在`.env.example`中列出，包括：

- `REACT_APP_API_URL` - 后端API地址
- `REACT_APP_VERSION` - 应用版本
- `REACT_APP_DEBUG` - 调试模式开关

### 认证系统

认证使用JWT实现，主要文件：
- `AuthenticatedApp.js` - 认证状态管理
- `LoginComponent.js` - 登录界面
- `axios.js` - API请求拦截器

### 组件开发

1. 组件应遵循以下原则：
   - 单一职责
   - 可复用性
   - 可测试性

2. 组件文档：
   - 导出Props类型定义
   - 添加JSDoc注释
   - 包含使用示例

3. 错误处理：
   - 使用ErrorBoundary包装组件
   - 实现合适的错误状态UI
   - 记录错误日志

## 测试

### 单元测试

```bash
# 运行所有测试
npm test

# 运行特定测试
npm test -- ComponentName.test.js

# 生成测试覆盖报告
npm test -- --coverage
```

### E2E测试

```bash
# 运行E2E测试
npm run test:e2e
```

## 构建和部署

### 开发环境

```bash
npm start
```

### 生产环境

```bash
# 构建生产版本
npm run build

# 预览生产构建
npm run serve
```

### Docker部署

```bash
# 构建镜像
docker build -t arc-workpiece-frontend .

# 运行容器
docker run -p 3000:80 arc-workpiece-frontend
```

## 故障排除

常见问题及解决方案参见：`docs/dev_troubleshooting.md`

## 性能优化

1. 代码分割
   - 使用React.lazy进行组件懒加载
   - 路由级别的代码分割

2. 性能监控
   - 使用React DevTools分析组件性能
   - 监控页面加载性能指标

## 安全注意事项

1. 认证
   - 使用HTTPS
   - 实现CSRF保护
   - XSS防护

2. 数据保护
   - 敏感信息加密存储
   - 清理会话数据

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交变更
4. 推送到分支
5. 创建Pull Request

## 许可证

Copyright © 2025
