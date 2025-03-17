import numpy as np
from scipy.interpolate import interp1d
from typing import List, Union

def normalize_sequence(points: Union[List[float], np.ndarray], target_length: int = 500) -> np.ndarray:
    """将点序列插值到指定长度
    
    Args:
        points: 输入点序列
        target_length: 目标序列长度
        
    Returns:
        插值后的等长序列
        
    Raises:
        ValueError: 输入序列为空时抛出
    """
    points = np.array(points)
    current_length = len(points)
    
    if current_length == 0:
        raise ValueError("Empty point sequence")
        
    if current_length == target_length:
        return points
        
    # 创建原始序列的索引
    x_original = np.linspace(0, 1, current_length)
    x_target = np.linspace(0, 1, target_length)
    
    # 使用线性插值
    interpolator = interp1d(x_original, points, kind='linear')
    interpolated = interpolator(x_target)
    
    return interpolated

def normalize_input(x: Union[List[float], np.ndarray]) -> np.ndarray:
    """标准化输入数据
    
    Args:
        x: 输入数据序列
        
    Returns:
        标准化后的数据序列
    """
    x = np.array(x)
    return (x - np.mean(x)) / np.std(x)
