# 轴向投影器（AxisProjector）类文档

## 简介

`AxisProjector`类用于将三维点云投影到垂直于指定轴向量的平面上。这在处理圆柱等轴对称物体的点云数据时特别有用，可以帮助我们获得物体的横截面视图。

## 类方法

### normalize(v)

静态方法，用于向量标准化。

**参数:**
- `v`: 输入向量

**返回:**
- 标准化后的单位向量

### get_projection_basis(v)

静态方法，用于构建投影平面上的正交基向量。

**参数:**
- `v`: 轴向量（将被标准化）

**返回:**
- 包含两个正交单位基向量的元组 (u,w)

### project_points(points, axis_vector, axis_point)

将三维点云投影到垂直于轴向量的平面上。

**参数:**
- `points`: Nx3数组，包含N个三维点
- `axis_vector`: 定义投影方向的三维向量
- `axis_point`: 轴线经过的三维点

**返回:**
返回一个元组，包含：
- `projected_points`: Nx3数组，投影后的三维点
- `planar_coords`: Nx2数组，投影平面上的二维坐标

## 使用示例

```python
import numpy as np
from algorithm.axis_projection import AxisProjector

# 创建测试点云（例如：圆柱体表面的点）
theta = np.linspace(0, 2*np.pi, 50)
z = np.linspace(0, 5, 50)
theta, z = np.meshgrid(theta, z)
r = 2.0
x = r * np.cos(theta)
y = r * np.sin(theta)
points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

# 定义投影轴
axis_vector = np.array([1.0, 1.0, 1.0])  # 45度角的轴
axis_point = np.array([0.0, 0.0, 0.0])   # 经过原点

# 创建投影器并执行投影
projector = AxisProjector()
projected_points, planar_coords = projector.project_points(points, axis_vector, axis_point)

# projected_points 包含投影后的三维坐标
# planar_coords 包含投影平面上的二维坐标
```

## 工作原理

1. **投影平面确定**
   - 投影平面垂直于给定的轴向量
   - 平面通过指定的轴点

2. **点投影过程**
   - 对每个输入点，计算其到投影平面的投影点
   - 投影使用公式：Q = P + t*v，其中
     - P 是原始点
     - v 是轴向量
     - t 是投影系数：t = ((x-P)·v)/(v·v)
     - Q 是投影点

3. **坐标转换**
   - 在投影平面上构建正交基向量(u,w)
   - 将投影点相对于轴点的位移向量投影到这些基向量上
   - 获得最终的二维坐标

## 注意事项

1. 输入的轴向量不能为零向量
2. 函数会自动将输入转换为numpy数组
3. 如果轴向量与y轴对齐，会使用特殊的基向量处理
4. 所有计算都是矢量化的，可以高效处理大量点
