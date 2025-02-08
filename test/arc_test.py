import numpy as np
import matplotlib.pyplot as plt
# import numpy as np\
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# 将相对路径转换为绝对路径
relative_path = "../"  # 相对路径，假设 "../test" 是你想添加的路径
absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
# 添加到 sys.path
sys.path.append(absolute_path)

from algorithm.circle_arc import CircleArc
arc = CircleArc()


# import matplotlib.pyplot as plt
from matplotlib import rcParams  # 用于设置字体

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 生成样本数据
# 真实圆的参数
a_true, b_true, r_true = 0, 0, 1

# 生成0到5度范围内的短弧上的点
theta = np.linspace(0, np.pi*(10/180), 10000)  # 5度转换为弧度
x_true = a_true + r_true * np.cos(theta)
y_true = b_true + r_true * np.sin(theta)

# 先给出一个大概的 A, B, C, D的初始值根据真实值
pre_r=0.9
pre_a=0.05
pre_b=0.05
pre_A = 1/(pre_r**2)
pre_B = -2*pre_A*pre_a
pre_C = -2*pre_A*pre_b
pre_D = pre_A*(pre_a**2 + pre_b**2) - 1

# 添加高斯噪声
noise = 0.001
x_obs = x_true + np.random.normal(0, noise, len(theta))
y_obs = y_true + np.random.normal(0, noise, len(theta))

# 定义目标函数和梯度
def objective(A, B, C, D, lam, eta, x, y):
    f = A*(x**2 + y**2) + B*x + C*y + D
    sum_term = np.sum(f**2)
    constraint = lam * (4*A*D - B**2 - C**2 + eta**2)
    return sum_term + constraint

def gradients(A, B, C, D, lam, eta, x, y):
    f = A*(x**2 + y**2) + B*x + C*y + D
    grad_A = 2 * np.sum(f * (x**2 + y**2)) + 4*lam*D
    grad_B = 2 * np.sum(f * x) - 2*lam*B
    grad_C = 2 * np.sum(f * y) - 2*lam*C
    grad_D = 2 * np.sum(f) + 4*lam*A
    grad_lam = 4*A*D - B**2 - C**2 + eta**2
    grad_eta = 2*lam*eta
    return np.array([grad_A, grad_B, grad_C, grad_D, grad_lam, grad_eta])

# 实现Adam优化算法的更新规则
def adam_update(m, v, grad, t, beta1=0.9, beta2=0.999, alpha=0.00001, eps=1e-8):
    m = beta1 * m + (1 + beta1) * grad
    v = beta2 * v + (1 + beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    theta_update = theta - alpha * m_hat / (np.sqrt(v_hat) + eps)
    return theta_update, m, v

# 优化循环
# 初始化参数
A = pre_A
B = pre_B
C = pre_C
D = pre_D
lam = 0.0
eta = 0.0
theta = np.array([A, B, C, D, lam, eta])
m = np.zeros_like(theta)
v = np.zeros_like(theta)
t = 1
converged = False
max_iter = 100000
delta = 1e-3
n=len(x_obs)
x_use=[]
y_use=[]

while not converged and t < max_iter:
    # print(f"第{t}次迭代，参数={theta}")
    x_use.append(t)
    grad = gradients(*theta, x=x_obs, y=y_obs)/n
    theta_new, m, v = adam_update(m, v, grad, t)
    A, B, C, D, lam, eta = theta
    a = -B / (2*A)
    b = -C / (2*A)
    r = np.sqrt((B**2 + C**2 - 4*A*D)/(4*A**2))

    # 求解全局Loss （所有点到中心点的距离的方差）
    loss= np.sum(((y_obs-b)**2+(x_obs-a)**2-r**2))/n
    loss=np.sqrt(np.abs(loss))
    # print( )
    # loss=np.linalg.norm(theta_new - theta)
    y_use.append(loss)
    print(f"第{t}次迭代，参数={theta},loss={loss},delta={delta}")
    # if np.linalg.norm(theta_new - theta) < delta:
    #     converged = True
    theta = theta_new
    t += 1
    if loss < delta:
        converged = True
# plot x_use y_use
plt.plot(x_use,y_use)
plt.show()
# 将优化得到的参数转换回标准圆的参数
A, B, C, D, lam, eta = theta
a = -B / (2*A)
b = -C / (2*A)
r = np.sqrt((B**2 + C**2 - 4*A*D)/(4*A**2))

print(f"拟合得到的圆：圆心=({a}, {b})，半径={r}")
points=np.stack([x_obs,y_obs],axis=1)
arc.fit_circle_arc(points)
# plt.scatter(points_use[:,0], points_use[:,1], s=0.1)
# 设置xy比例相同
# plt.axis('equal')

center = arc.first_param[0:2]
radius = arc.first_param[2]
print(f"直接拟合法得到的圆：圆心=({center[0]}, {center[1]})，半径={radius}")

# 可选：可视化验证
# 生成拟合圆
theta_fit = np.linspace(0, 2*np.pi, 100)
x_fit = a + r * np.cos(theta_fit)
y_fit = b + r * np.sin(theta_fit)

# 绘制观测点和拟合圆
plt.scatter(x_obs, y_obs, label='观测点')
plt.plot(x_fit, y_fit, label='拟合圆', color='red')
plt.title('使用ICF方法拟合圆弧')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.axis('equal')
plt.show()