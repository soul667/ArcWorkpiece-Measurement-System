import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
# 设置 matplotlib 使用宋体（SimSun）
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class CircleArc:
    def __init__(self):
        """
        初始化 CircleArc 类。

        center: 圆心坐标 (x, y)
        radius: 圆的半径
        angle_range: 弧线的角度范围 (起始角度, 结束角度)，单位为度
        num_points: 生成的点的数量
        noise_level: 添加到点的高斯噪声的标准差
        """
        # self.points = points

    # def generate_arc_points(self):
    #     """
    #     生成弧线上的点，并添加可选的高斯噪声。
    #     """
    #     # 将角度范围转换为弧度
    #     angles = np.linspace(np.radians(self.angle_range[0]), np.radians(self.angle_range[1]), self.num_points)
    #     # 计算点的 x 和 y 坐标
    #     x = self.center[0] + self.radius * np.cos(angles)
    #     y = self.center[1] + self.radius * np.sin(angles)
        
    #     # 添加高斯噪声
    #     x += np.random.normal(0, self.noise_level, self.num_points)
    #     y += np.random.normal(0, self.noise_level, self.num_points)
        
    #     return np.column_stack((x, y))


    def generate_arc_points(self,center, radius, angle_range, num_points, noise_level=0.0):
        angles = np.linspace(np.radians(angle_range[0]), np.radians(angle_range[1]), num_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        x += np.random.normal(0, noise_level, num_points)
        y += np.random.normal(0, noise_level, num_points)
        
        return np.column_stack((x, y))

    def calculate_radius(self,center, points):
        x, y = points[:, 0], points[:, 1]
        return np.mean(np.sqrt((x - center[0])**2 + (y - center[1])**2))

    def cost_function(self,center, points):
        x, y = points[:, 0], points[:, 1]
        radius = self.calculate_radius(center, points)
        return np.sum((np.sqrt((x - center[0])**2 + (y - center[1])**2) - radius)**2)

    def fit_circle_arc_with_minimize(self,points, initial_center):
        result = minimize(self.cost_function, initial_center, args=(points,), method='BFGS')
        center = result.x
        radius = self.calculate_radius(center, points)
        return center[0], center[1], radius
    # def fit_circle_arc_hyper(self,points):
    def hyper_circle_fit(self,points):
        """
        用“Hyper”方法拟合圆的参数。
        
        参数：
            XY (numpy.ndarray): n x 2 的数组，表示点的坐标 (x, y)。
        
        返回：
            tuple: (a, b, R) - 圆的中心 (a, b) 和半径 R。
        """
        # 提取 x 和 y 坐标
        self.points=points
        X = self.points[:, 0]
        Y = self.points[:, 1]
        # X = XY[:, 0]
        # Y = XY[:, 1]
        
        # 计算 Z = x^2 + y^2
        Z = X**2 + Y**2
        
        # 构造矩阵 ZXY1
        ZXY1 = np.column_stack((Z, X, Y, np.ones_like(Z)))
        
        # 计算 M 和 ZXY1 的均值
        M = ZXY1.T @ ZXY1
        S = np.mean(ZXY1, axis=0)
        
        # 定义约束矩阵 N
        N = np.array([
            [8*S[0], 4*S[1], 4*S[2], 2],
            [4*S[1], 1,       0,       0],
            [4*S[2], 0,       1,       0],
            [2,       0,       0,       0]
        ])
        
        # 解广义特征值问题
        NM = np.linalg.inv(N) @ M
        D, E = np.linalg.eig(NM)
        
        # 对特征值排序并选择第二小的特征值对应的特征向量
        idx = np.argsort(D)
        if D[idx[0]] > 0:
            raise ValueError("错误：最小特征值为正数，说明数据异常。")
        if D[idx[1]] < 0:
            raise ValueError("错误：第二小特征值为负数，说明数据异常。")
        A = E[:, idx[1]]
        
        # 计算圆的参数
        a, b = -A[1:3] / (2 * A[0])
        R = np.sqrt((A[1]**2 + A[2]**2 - 4*A[0]*A[3]) / (4 * A[0]**2))
        
        return (a, b, R)
    def fit_circle_arc(self,points):
        """
        使用特征值分解方法拟合圆形弧线。
        """
        # 构造设计矩阵
        self.points=points
        x = self.points[:, 0]
        y = self.points[:, 1]
        # print(x)
        # print(y)
        A = np.column_stack((x**2 + y**2, x, y, np.ones_like(x)))
        
        # 构造 Q 矩阵
        Q = np.array([[0, 0, 0, -2],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [-2, 0, 0, 0]])
        
        # 求解广义特征值问题 A.T A P = η Q P
        M = A.T @ A
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Q) @ M)
        
        # 找到最小非负特征值对应的特征向量
        min_eigenvalue_index = np.argmin(np.abs(eigenvalues))
        P = eigenvectors[:, min_eigenvalue_index]
        
        # 提取参数
        A, B, C, D = P
        center_x = -B / (2 * A)
        center_y = -C / (2 * A)
        radius = np.sqrt((B**2 + C**2 - 4 * A * D) / (4 * A**2))
        self.first_param=(center_x, center_y, radius)
        return center_x, center_y, radius

    def plot_arc(self, fitted_circle=True, point_size=20):
        """
        绘制生成的弧线点和拟合的圆形。

        fitted_circle: 是否绘制拟合的圆形
        point_size: 点的大小，默认为 20
        """
        plt.figure(figsize=(8, 8))
        # 绘制生成的弧线点
        plt.scatter(self.points[:, 0], self.points[:, 1], label="生成的弧线点", color='blue', alpha=0.6, s=point_size)
        
        if fitted_circle:
            # 拟合圆形
            center_x, center_y, radius_fitted = self.fit_circle_arc()
            # 绘制拟合的圆形
            circle = plt.Circle((center_x, center_y), radius_fitted, color='red', fill=False, label="拟合的圆形")
            plt.gca().add_patch(circle)
            # 绘制拟合的圆心
            plt.scatter(center_x, center_y, color='red', marker='x', label="拟合的圆心")
        print(f"拟合的圆心: ({center_x:.2f}, {center_y:.2f})")
        print(f"拟合的半径: {radius_fitted:.2f}")
        # 设置图形属性
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("弧线点及拟合圆形")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # 保证 x 和 y 轴比例相同
        plt.show()

# 示例用法
if __name__ == "__main__":
    # 定义参数
    center = (0, 0)
    radius = 5
    angle_range = (0, 15)  # 角度范围
    num_points = 10000
    noise_level = 0.05

    # 创建 CircleArc 实例
    arc = CircleArc()
    points=arc.generate_arc_points(center, radius, angle_range, num_points, noise_level)
    # show points
    plt.scatter(points[:, 0], points[:, 1], s=0.1)
    plt.axis('equal')
    plt.show()
    # arc.fit_circle_arc
    # 绘制弧线点和拟合的圆形，并设置点的大小
    arc.fit_circle_arc(points)
    print(arc.first_param)

    print(arc.hyper_circle_fit(points))
    # print(arc.)
    # arc.plot_arc(fitted_circle=True, point_size=5)
    # print()