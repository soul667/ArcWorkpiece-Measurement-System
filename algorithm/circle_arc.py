import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eig
from scipy.optimize import least_squares

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
        使用特征值分解方法拟合圆形弧线。
        """
        # 构造设计矩阵
        self.points=points
        x = self.points[:, 0]
        y = self.points[:, 1]
        # print(x)
        # print(y)
        A = np.column_stack((x**2 + y**2, x, y, np.ones_like(x)))
        S= np.mean(A, axis=0)
        # 构造 Q 矩阵
        N = np.array([
            [8*S[0], 4*S[1], 4*S[2], 2],
            [4*S[1], 1,       0,       0],
            [4*S[2], 0,       1,       0],
            [2,       0,       0,       0]
        ])
        
        # 求解广义特征值问题 A.T A P = η Q P
        M = A.T @ A
        # eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Q) @ M)
        eigenvalues, eigenvectors = eig(M,N)

        
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
    def hyper_taubin_fit(self,points):
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
        S= np.mean(A, axis=0)
        # 构造 Q 矩阵
        N = np.array([
            [4*S[0], 2*S[1], 2*S[2], 0],
            [2*S[1], 1,       0,       0],
            [2*S[2], 0,       1,       0],
            [0,       0,       0,       0]
        ])
        
        # 求解广义特征值问题 A.T A P = η Q P
        M = A.T @ A
        eigenvalues, eigenvectors = eig(M,N)

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

    def _numerical_jacobian(self, f, p, eps=1e-8):
        """数值计算雅可比矩阵"""
        f0 = f(p)
        n = len(f0)
        m = len(p)
        J = np.zeros((n, m))
        
        for i in range(m):
            p_perturbed = p.copy()
            p_perturbed[i] += eps
            f_perturbed = f(p_perturbed)
            J[:, i] = (f_perturbed - f0) / eps
            
        return J

    def _circle_residuals(self, p, points):
        """计算圆拟合的残差向量"""
        a, b, r = p  # 圆心坐标(a,b)和半径r
        return np.sqrt((points[:,0]-a)**2 + (points[:,1]-b)**2) - r
    def fit_circle_arc_scipy(self, points):
        """使用scipy的least_squares实现LM算法拟合圆弧
        
        Args:
            points: 输入点集 shape=(n,2)
        Returns:
            center_x, center_y, radius: 圆心坐标和半径
        """
        # 残差函数
        def residual_func(p):
            a, b, r = p
            return np.sqrt((points[:,0]-a)**2 + (points[:,1]-b)**2) - r
            
        # 获取初始估计
        cx, cy, r = self.hyper_circle_fit(points)
        # cx, cy, r = 1,1,40
        p0 = np.array([cx, cy, r])
        print(f"初始参数: {p0}")
        # 使用LM算法优化
        result = least_squares(
            residual_func,
            p0,
            method='lm',        # 使用Levenberg-Marquardt算法
            ftol=1e-12,        # 函数收敛容差
            xtol=1e-12,        # 参数收敛容差
            gtol=1e-12,        # 梯度收敛容差
            max_nfev=1000,     # 最大函数评估次数
            verbose=2          # 输出详细信息
        )
        print(f"优化结果: {result.x}")
        print(f"优化状态: {result.message}")
        if not result.success:
            print(f"优化未收敛: {result.message}")
        self.first_param=(result.x[0], result.x[1], result.x[2])
        return result.x[0], result.x[1], result.x[2]
    def _levenberg_marquardt(self, f, x, p0, jac=None, max_iter=10000,
                          eps1=1e-16, eps2=1e-16, eps3=1e-16,
                          mu0=1e-6, tol=1e-8):
        """
                Levenberg-Marquardt算法实现
                
                参数:
                    f : callable(p) -> 向量函数输出，形状(n,)
                    x : 目标向量，形状(n,)
                    p0 : 初始参数估计，形状(m,)
                    jac : 雅可比矩阵计算函数，callable(p) -> 矩阵(n, m)
                    max_iter : 最大迭代次数
                    eps1-3 : 停止条件阈值
                    mu0 : 初始阻尼因子
                    tol : 数值微分的小量

                返回:
                    p_opt : 优化后的参数向量
        """
        # 初始化参数
        p = p0.copy().astype(float)
        n_params = len(p0)
        k = 0
        v = 2
        mu = mu0
        
        # 数值雅可比计算（如果未提供）
        if jac is None:
            def jac(p):
                return self._numerical_jacobian(f, p, tol)
        
        # 初始计算
        phi = x - f(p)
        J = jac(p)
        A = J.T @ J
        g = J.T @ phi
        stop = False
        
        # 记录残差历史
        residual = 0.5 * np.sum(phi**2)
        residuals = [residual]
        min_iter = 100
        while not stop and k < max_iter:
            try:
                # 解线性方程 (A + μI)δp = -g
                delta_p = np.linalg.solve(A + mu * np.eye(n_params), -g)
            except np.linalg.LinAlgError:
                delta_p = -g / (mu + 1e-12)  # 退化为梯度下降

            # 停止条件1: 更新量足够小
            if np.linalg.norm(delta_p) <= eps2:
                stop = True
                print(np.linalg.norm(delta_p))
                print("更新量足够小，停止迭代")
                break

            # 计算候选参数
            p_new = p + delta_p
            phi_new = x - f(p_new)
            residual_new = 0.5 * np.sum(phi_new**2)
            print(f"当前残差: {residual_new:.4f}, 迭代次数: {k}")
            # 计算实际/预测残差变化量
            actual_reduction = residual - residual_new
            predicted_reduction = -delta_p.T @ (g + 0.5 * A @ delta_p)
            rho = actual_reduction / (predicted_reduction + 1e-12)

            # 更新判断
            if residual_new < residual and rho > 0:
                # 接受更新
                print(f"迭代次数: {k}, rho: {rho:.4f}, 更新量: {np.linalg.norm(delta_p):.4f}")
                p = p_new
                phi = phi_new
                residual = residual_new
                J = jac(p)
                A = J.T @ J
                g = J.T @ phi
                residuals.append(residual)
                
                # 调整阻尼因子
                mu *= max(1/3, 1 - (2*rho - 1)**3)
                v = 2
                
                # 停止条件检查
                if (np.max(np.abs(g)) <= eps1 or 
                    np.sum(phi**2) <= eps3):
                    print("收敛到最优解")
                    stop = True
            else:
                # 拒绝更新，增大阻尼
                mu *= v
                v *= 2

            k += 1

        return p, np.array(residuals)

    def fit_circle_arc_lm(self, points):
        """使用Levenberg-Marquardt算法拟合圆弧
        Args:
            points: 输入点集 shape=(n,2)
        Returns:
            center_x, center_y, radius: 圆心坐标和半径
        """
        self.points = points
        
        # 使用hyper_circle_fit获取初始估计
        # center_x, center_y, radius = self.hyper_circle_fit(points)
        # center_x, center_y, radius = self.hyper_circle_fit(points)
        center_x, center_y, radius = 1,4,40

        p0 = np.array([center_x, center_y, radius])
        
        # 定义残差函数闭包
        def residual_func(p):
            return self._circle_residuals(p, points)
        
        # 执行LM优化
        x = np.zeros_like(residual_func(p0))  # 目标残差为0
        p_opt, residuals = self._levenberg_marquardt(
            f=residual_func,
            x=x,
            p0=p0,
            max_iter=100000,
            eps1=1e-18,
            eps2=1e-18,
            eps3=1e-18,
            mu0=1e-8,
            tol=1e-4
        )
        # 输出每一次的residuals
        return p_opt[0], p_opt[1], p_opt[2]  # 返回优化后的圆心坐标和半径


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
        N = np.array([[0, 0, 0, -2],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [-2, 0, 0, 0]])
        
        # 求解广义特征值问题 A.T A P = η Q P
        M = A.T @ A
        eigenvalues, eigenvectors = eig(M,N)

        
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
    center = (56, 5)
    radius = 14.67
    angle_range = (100, 106)  # 角度范围
    num_points = 10000
    noise_level = 0.001

    # 创建 CircleArc 实例并生成测试数据
    arc = CircleArc()
    points = arc.generate_arc_points(center, radius, angle_range, num_points, noise_level)
    
    # 展示原始数据点
    # plt.figure(figsize=(10, 10))
    # plt.scatter(points[:, 0], points[:, 1], s=0.5, label='数据点')
    # plt.axis('equal')
    # plt.title('原始数据点')
    # plt.grid(True)
    # plt.show()

    # 对比不同拟合方法
    methods = [
        ('Pratt', arc.fit_circle_arc),
        ('Hyper', arc.hyper_circle_fit),
        ('Taubin', arc.hyper_taubin_fit),
        # ('LM ', arc.fit_circle_arc_lm)
        ('LM', arc.fit_circle_arc_lm),
        ('Scipy', arc.fit_circle_arc_scipy)
    ]

    plt.figure(figsize=(12, 12))
    plt.scatter(points[:, 0], points[:, 1], s=0.5, c='gray', label='数据点')
    colors = ['red', 'blue', 'green', 'purple']

    print("\n各方法拟合结果对比:")
    print("=" * 50)
    
    # for (name, method), color in zip(methods, colors):
    #     cx, cy, r = method(points)
    #     print(f"\n{name}:")
    #     print(f"圆心: ({cx:.6f}, {cy:.6f})")
    #     print(f"半径: {r:.6f}")
    #     print(f"与真实参数偏差:")
    #     print(f"圆心偏差: {((cx-center[0])**2 + (cy-center[1])**2)**0.5:.6f}")
    #     print(f"半径偏差: {abs(r-radius):.6f}")
        
    #     # 绘制拟合圆
    #     circle = plt.Circle((cx, cy), r, fill=False, color=color, label=name)
    #     plt.gca().add_patch(circle)
    #     plt.plot(cx, cy, 'x', color=color, markersize=10)

    # plt.title('DUIBI')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # print(methods)
    # for (name, method), color in zip(methods, colors):
    #     cx, cy, r = method(points)
    #     print(f"\n{name}:")
    #     print(f"圆心: ({cx:.6f}, {cy:.6f})")
    #     print(f"半径: {r:.6f}")
    #     print(f"与真实参数偏差:")
    #     print(f"圆心偏差: {((cx-center[0])**2 + (cy-center[1])**2)**0.5:.6f}")
    #     print(f"半径偏差: {abs(r-radius):.6f}")
    arc.hyper_circle_fit(points)
    print(arc.first_param)
    arc.fit_circle_arc_scipy(points)
    print(arc.first_param)
    # arc.fit_circle_arc_lm(points)
    # print(arc.first_param)