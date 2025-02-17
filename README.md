# 弧形工件测量系统

该项目旨在使用各种算法和点云处理技术来测量弧形工件。

## 开发环境

该项目使用开发容器来确保一致的开发环境。开发容器包括项目所需的所有依赖项和工具。

### 先决条件

- Docker
- Visual Studio Code
- Visual Studio Code 的 Remote - Containers 扩展

### 设置

1. 克隆存储库：
    ```sh
    git clone https://github.com/your-repo/arcworkpiece-measurement-system.git
    cd arcworkpiece-measurement-system
    ```

2. 在 Visual Studio Code 中打开项目：
    ```sh
    code .
    ```

3. 当提示时，在开发容器中重新打开项目。

### 包含的扩展

开发容器包括以下 Visual Studio Code 扩展：

- Python
- C++ 工具
- 拼写检查
- Docker
- Pylance
- Jupyter
- GitHub Copilot
- GitLens
- Error Lens
- ESLint
- Prettier
- ES7 React/Redux/GraphQL/React-Native 代码片段
- 自动重命名标签
- 自动关闭标签
- Typst LSP

### 依赖项

开发容器安装以下 Python 依赖项：

- numpy
- opencv-python-headless
- pyyaml
- open3d
- matplotlib
- scikit-learn
- scipy
- robpy
- py_cylinder_fitting
- skspatial
- fastapi
- uvicorn
- pyblind

### 用法

您可以在开发容器中开始开发和测试代码。所有必要的工具和依赖项都已预安装。

### 许可证

该项目根据 MIT 许可证授权。


``` shell
uvicorn UserInterface.fastapi_test:app --reload --host 0.0.0.0 --port 9304
```

<!-- pip install python-multipart -->