import numpy as np
from scipy.interpolate import splprep, splev
from enum import Enum

class State(Enum):
    UNDETERMINED = 0  # 还没判断
    POSITIVE = 1      # 正
    NEGATIVE = -1     # 负
    
def smooth_points(points, smoothing_factor=0.1):
    """
    使用 B 样条拟合平滑一组二维点。

    参数:
    points: numpy.ndarray
        一个 Nx2 的 (x, y) 坐标数组。
    smoothing_factor: float
        控制平滑程度的非负值。较大的值会产生更平滑的曲线（默认值为 0，表示插值点）。

    返回:
    smoothed_points: numpy.ndarray
        一个 Mx2 的平滑 (x, y) 坐标数组。
    """
    points = np.asarray(points)
    if points.shape[1] != 2:
        raise ValueError("输入点必须是一个 Nx2 的 (x, y) 坐标数组。")

    # 提取 x 和 y 坐标
    x, y = points[:, 0], points[:, 1]

    # 使用平滑拟合 B 样条
    tck, u = splprep([x, y], s=smoothing_factor)

    # 生成平滑点
    u_fine = np.linspace(0, 1, len(points) * 10)  # 将分辨率提高 10 倍
    x_smooth, y_smooth = splev(u_fine, tck)

    smoothed_points = np.column_stack((x_smooth, y_smooth))
    return smoothed_points

def get_inflection_points(self, points: np.ndarray, window_size: int = 6, consecutive_threshold: int = 5) -> list:
    """
    计算曲线的拐点（极值点）
    
    Args:
        points: np.ndarray - 输入点云数据
        window_size: int - 计算梯度的窗口大小（必须是偶数）
        consecutive_threshold: int - 判定为趋势变化的连续点数量
        
    Returns:
        list - 极值点的x坐标列表
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    data_length = len(x_coords)
    half_window = window_size // 2
    
    # 初始化
    gradients = np.zeros(data_length)
    current_state = State.UNDETERMINED
    consecutive_count = 0
    inflection_points = []
    temp_inflection_x = 0
    
    # 计算梯度并查找拐点
    for i in range(half_window, data_length - half_window):
        # 计算梯度
        gradients[i] = (y_coords[i + half_window] - y_coords[i - half_window]) / window_size
        
        if gradients[i] == 0:  # 跳过零梯度点
            continue
            
        if current_state == State.UNDETERMINED:
            # 初始状态判定
            consecutive_count += 1 if gradients[i] > 0 else -1
            if abs(consecutive_count) == consecutive_threshold:
                current_state = State.POSITIVE if consecutive_count > 0 else State.NEGATIVE
                
        elif current_state == State.POSITIVE:
            if gradients[i] < 0:
                consecutive_count -= 1
                if consecutive_count == consecutive_threshold - 1:
                    temp_inflection_x = x_coords[i]
                elif consecutive_count == 0:
                    current_state = State.NEGATIVE
                    consecutive_count = -consecutive_threshold
                    inflection_points.append(temp_inflection_x)
            elif consecutive_count < consecutive_threshold and gradients[i] != 0:
                consecutive_count += 1
                
        elif current_state == State.NEGATIVE:
            if gradients[i] > 0:
                consecutive_count += 1
                if consecutive_count == -consecutive_threshold + 1:
                    temp_inflection_x = x_coords[i]
                elif consecutive_count == 0:
                    current_state = State.POSITIVE
                    consecutive_count = consecutive_threshold
                    inflection_points.append(temp_inflection_x)
            elif consecutive_count > -consecutive_threshold and gradients[i] != 0:
                consecutive_count -= 1
    
    return inflection_points