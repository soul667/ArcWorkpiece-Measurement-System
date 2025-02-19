import numpy as np
import open3d as o3d
import pypcl_algorithms as pcl_algo

def generate_cylinder_points(point_count=1000, radius=0.5, height=2.0, noise_std=0.01):
    """Generate synthetic cylinder point cloud with noise."""
    # Generate random angles and heights
    thetas = np.random.uniform(0, 2*np.pi, point_count)
    heights = np.random.uniform(-height/2, height/2, point_count)
    
    # Generate points on cylinder surface
    x = radius * np.cos(thetas)
    y = radius * np.sin(thetas)
    z = heights
    
    # Combine into points array
    points = np.column_stack([x, y, z])
    
    # Add random noise
    noise = np.random.normal(0, noise_std, (point_count, 3))
    points += noise
    
    return points

def visualize_points_with_normals(points, normals, subsample=5):
    """Visualize points and their normal vectors using Open3D."""
    # Create point cloud for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray

    # Create lines to visualize normal vectors (subsample for clarity)
    normal_scale = 0.1  # Scale factor for normal vector visualization
    idx = np.arange(0, len(points), subsample)  # Subsample points
    normal_points = np.vstack([
        points[idx],
        points[idx] + normals[idx] * normal_scale
    ])
    lines = np.array([[i*2, i*2+1] for i in range(len(idx))])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(normal_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 1, 0])  # Green

    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    
    # Visualize everything
    o3d.visualization.draw_geometries([pcd, line_set, coord_frame])

def visualize_cylinder_fit(points, point_on_axis, axis_direction, radius):
    """Visualize points and fitted cylinder axis using Open3D."""
    # Create point cloud for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    
    # Create line set for axis visualization
    # Extend the axis line beyond the points
    max_range = np.max(np.abs(points))
    line_points = np.array([
        point_on_axis - axis_direction * max_range * 1.5,
        point_on_axis + axis_direction * max_range * 1.5
    ])
    lines = np.array([[0, 1]])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # Red
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=point_on_axis
    )
    
    # Visualize everything
    o3d.visualization.draw_geometries([pcd, line_set, coord_frame])

def test_normal_computation():
    """Test and visualize normal computation."""
    print("\nTesting normal computation...")
    
    # Generate synthetic cylinder data
    points = generate_cylinder_points(point_count=1000, radius=0.5, height=2.0, noise_std=0.01)
    
    # Compute normals
    normals = pcl_algo.compute_normals(points, k_neighbors=30)
    
    print("Normal computation completed.")
    print(f"Number of points: {len(points)}")
    print(f"Number of normals: {len(normals)}")
    print("\nFirst few normals:")
    print(normals[:3])
    
    # Visualize points with normals
    visualize_points_with_normals(points, normals)

def test_cylinder_fitting():
    """Test and visualize cylinder fitting with all parameters."""
    print("\nTesting cylinder fitting...")
    
    # Generate synthetic cylinder data
    points = generate_cylinder_points(point_count=1000, radius=0.5, height=2.0, noise_std=0.01)
    
    # Fit cylinder using RANSAC with all parameters
    point_on_axis, axis_direction, radius = pcl_algo.fit_cylinder_ransac(
        points,
        distance_threshold=0.01,    # Maximum distance from point to cylinder surface
        max_iterations=1000,        # RANSAC iterations
        k_neighbors=30,             # Number of neighbors for normal estimation
        normal_distance_weight=0.1, # Weight for normal distance in fitting
        min_radius=0.1,            # Minimum allowed cylinder radius
        max_radius=1.0             # Maximum allowed cylinder radius
    )
    
    # Print results
    print("\nFitted Cylinder Parameters:")
    print(f"Point on axis: [{point_on_axis[0]:.3f}, {point_on_axis[1]:.3f}, {point_on_axis[2]:.3f}]")
    print(f"Axis direction: [{axis_direction[0]:.3f}, {axis_direction[1]:.3f}, {axis_direction[2]:.3f}]")
    print(f"Radius: {radius:.3f}")
    
    # Visualize results
    visualize_cylinder_fit(points, point_on_axis, axis_direction, radius)

if __name__ == "__main__":
    # Test normal computation
    test_normal_computation()
    
    # Test cylinder fitting
    test_cylinder_fitting()
