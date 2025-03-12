import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.neighbors import KDTree

class PointCloudGrouper:
    @staticmethod
    def get_all_groups(points: np.ndarray, axis: str = 'x') -> List[np.ndarray]:
        """
        按指定轴对点云进行分组，返回所有组
        
        Args:
            points (np.ndarray): 点云数据，Nx3数组
            axis (str): 分组轴 ('x', 'y', 或 'z')
            
        Returns:
            List[np.ndarray]: 分组后的点列表
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
            
        if points.shape[1] != 3:
            raise ValueError("Points must be a Nx3 array")
            
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis_map:
            raise ValueError(f"Invalid axis: {axis}. Must be one of: x, y, z")
            
        axis_idx = axis_map[axis]
        
        # 获取唯一坐标
        unique_coords = np.unique(points[:, axis_idx])
        
        # 对每个坐标值获取对应的点
        groups = []
        for coord in unique_coords:
            mask = np.isclose(points[:, axis_idx], coord, rtol=1e-5)
            group_points = points[mask]
            groups.append(group_points)
            
        return groups

    @staticmethod
    def find_neighbors_kdtree(points: np.ndarray, k: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        使用KDTree为每个点找到k个最近邻居
        
        Args:
            points (np.ndarray): 点云数据，Nx3数组
            k (int): 需要查找的邻居数量
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: 
                - 每个点的邻居点列表
                - 每个点的邻居距离列表
        """
        # 创建KDTree
        tree = KDTree(points)
        
        # 查询k个最近邻
        distances, indices = tree.query(points, k=k+1)  # k+1 因为包含点本身
        
        # 移除每个点本身（索引为0的邻居）
        neighbor_indices = indices[:, 1:]
        neighbor_distances = distances[:, 1:]
        
        # 获取邻居点坐标
        neighbors = []
        for idx_set in neighbor_indices:
            neighbor_points = points[idx_set]
            neighbors.append(neighbor_points)
            
        return neighbors, [d for d in neighbor_distances]

    @staticmethod
    def group_by_axis(points: np.ndarray, axis: str = 'x', index: int = 0) -> Dict[str, Any]:
        """
        返回指定索引的线条数据
        
        Args:
            points (np.ndarray): 点云数据，Nx3数组
            axis (str): 分组轴 ('x', 'y', 或 'z')
            index (int): 要返回的线条索引
            
        Returns:
            Dict containing:
                - group: Dict with coordinate value and associated points
                - axis: The axis used for grouping
                - total_groups: Total number of unique groups
                - current_index: Current index
                - coordinate_range: [min, max] coordinate values for context
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
            
        if points.shape[1] != 3:
            raise ValueError("Points must be a Nx3 array")
            
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis_map:
            raise ValueError(f"Invalid axis: {axis}. Must be one of: x, y, z")
            
        axis_idx = axis_map[axis]
        
        # Get unique coordinates along the specified axis
        unique_coords = np.unique(points[:, axis_idx])
        
        # 获取指定位置的坐标值
        if index < 0 or index >= len(unique_coords):
            raise ValueError(f"索引超出范围: {index}，总线条数: {len(unique_coords)}")
            
        coord = unique_coords[index]
        
        # 找到共享该坐标的点
        mask = np.isclose(points[:, axis_idx], coord, rtol=1e-5)
        group_points = points[mask].tolist()
        
        # 计算坐标范围，用于显示进度上下文
        coord_range = [float(unique_coords[0]), float(unique_coords[-1])]
        
        return {
            'group': {
                'coordinate': float(coord),
                'points': group_points
            },
            'axis': axis,
            'total_groups': len(unique_coords),
            'current_index': index,
            'coordinate_range': coord_range
        }

    @staticmethod
    def remove_groups(points: np.ndarray, group_indices: List[int], axis: str = 'x') -> np.ndarray:
        """
        移除指定索引的线条组
        
        Args:
            points (np.ndarray): 点云数据，Nx3数组
            group_indices (List[int]): 要删除的线条索引列表
            axis (str): 分组轴 ('x', 'y', 或 'z')
            
        Returns:
            np.ndarray: 移除指定线条后的点云数据
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
            
        if points.shape[1] != 3:
            raise ValueError("Points must be a Nx3 array")
            
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis_map:
            raise ValueError(f"Invalid axis: {axis}. Must be one of: x, y, z")
            
        axis_idx = axis_map[axis]
        
        # 获取所有唯一坐标
        unique_coords = np.unique(points[:, axis_idx])
        
        # 验证索引有效性
        invalid_indices = [i for i in group_indices if i < 0 or i >= len(unique_coords)]
        if invalid_indices:
            raise ValueError(f"无效的线条索引: {invalid_indices}")
        
        # 收集要删除的坐标值
        coords_to_remove = unique_coords[group_indices]
        
        # 创建掩码，标识要保留的点
        mask = np.ones(len(points), dtype=bool)
        for coord in coords_to_remove:
            mask &= ~np.isclose(points[:, axis_idx], coord, rtol=1e-5)
        
        # 返回保留的点
        return points[mask]

    @staticmethod
    def remove_duplicate_neighbors(neighbors: List[np.ndarray]) -> List[np.ndarray]:
        """
        移除邻居点中的重复点
        
        Args:
            neighbors (List[np.ndarray]): 邻居点列表
            
        Returns:
            List[np.ndarray]: 去重后的邻居点列表
        """
        unique_neighbors = []
        for neighbor_set in neighbors:
            # 使用structured array来去重
            dtype = [('x', float), ('y', float), ('z', float)]
            unique = np.unique(neighbor_set.view(dtype))
            unique_neighbors.append(unique.view(float).reshape(-1, 3))
        return unique_neighbors
