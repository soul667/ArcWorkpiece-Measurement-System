import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import pypcl_algorithms as pcl_algo
import matplotlib

# Configure matplotlib display settings
matplotlib.rcParams['font.family'] = "DejaVu Sans Mono"  # Set font
def generate_cylinder_points(point_count=1000, radius=0.5, height=2.0, noise_std=0.01, 
                           axis_direction=np.array([0, 0, 1])):
    """Generate cylinder point cloud data with specified axis direction"""
    # Normalize axis vector
    axis = axis_direction / np.linalg.norm(axis_direction)
    
    # Create rotation matrix to align [0,0,1] with target axis direction
    if np.allclose(axis, [0, 0, 1]):
        R = np.eye(3)
    else:
        # Calculate rotation axis (obtained through cross product)
        rot_axis = np.cross([0, 0, 1], axis)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        
        # Calculate rotation angle
        cos_angle = np.dot([0, 0, 1], axis)
        angle = np.arccos(cos_angle)
        
        # Rodriguez rotation formula
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                     [rot_axis[2], 0, -rot_axis[0]],
                     [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * K.dot(K)
    
    # First generate cylinder point cloud in Z-axis direction
    thetas = np.random.uniform(0, 2*np.pi, point_count)
    heights = np.random.uniform(-height/2, height/2, point_count)
    
    # Generate points on cylinder surface
    x = radius * np.cos(thetas)
    y = radius * np.sin(thetas)
    z = heights
    
    # Combine into point cloud array
    points = np.column_stack([x, y, z])
    
    # Rotate point cloud to align with target axis
    points = points.dot(R.T)
    
    # Add random noise
    noise = np.random.normal(0, noise_std, (point_count, 3))
    points += noise
    
    return points, axis

def generate_test_cases():
    """Generate test cases for different directions"""
    test_cases = []
    
    # Case 1: Vertical cylinder (Z-axis)
    points1, axis1 = generate_cylinder_points(axis_direction=np.array([0, 0, 1]))
    test_cases.append(("Vertical Cylinder (Z-axis)", points1, axis1))
    
    # Case 2: Horizontal cylinder (X-axis)
    points2, axis2 = generate_cylinder_points(axis_direction=np.array([1, 0, 0]))
    test_cases.append(("Horizontal Cylinder (X-axis)", points2, axis2))
    
    # Case 3: Diagonal cylinder
    points3, axis3 = generate_cylinder_points(axis_direction=np.array([1, 1, 1]))
    test_cases.append(("Diagonal Cylinder", points3, axis3))
    
    # Case 4: Random direction
    random_axis = np.random.randn(3)
    points4, axis4 = generate_cylinder_points(axis_direction=random_axis)
    test_cases.append(("Random Direction Cylinder", points4, axis4))
    
    return test_cases

def save_o3d_visualization(geometries, filename, width=1920, height=1080):
    """Save visualization results using Open3D off-screen rendering"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # Add all geometries
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Optimize view
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([1, 1, 1])
    
    # Auto-align camera to scene
    vis.get_view_control().set_zoom(0.8)
    vis.update_renderer()
    
    # Capture and save image
    image = vis.capture_screen_float_buffer(True)
    plt.imsave(filename, np.asarray(image))
    
    vis.destroy_window()

def save_points_with_normals(points, normals, subsample=5, filename="normals.png"):
    """Save point cloud visualization results with normal vectors"""
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    
    # Create normal vector visualization lines
    normal_scale = 0.1  # Normal vector display scale
    idx = np.arange(0, len(points), subsample)  # Sampling points
    normal_points = np.vstack([
        points[idx],
        points[idx] + normals[idx] * normal_scale
    ])
    lines = np.array([[i*2, i*2+1] for i in range(len(idx))])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(normal_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 1, 0])  # Green
    
    # Create coordinate system
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    
    # Save visualization results
    save_o3d_visualization([pcd, line_set, coord_frame], filename)

def visualize_results(points, true_axis, ransac_result, svd_axis, title, filename):
    """Visualize and save cylinder fitting results"""
    point_on_axis, axis_ransac, radius = ransac_result
    
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title)
    
    # Plot point cloud and axis lines
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='gray', alpha=0.5, s=1)
    
    # Plot true axis line
    center = np.mean(points, axis=0)
    line_points = np.array([center - true_axis * 2, center + true_axis * 2])
    ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], 'b-', 
            linewidth=2, label='True Axis')
    
    # Plot RANSAC results
    line_points = np.array([
        point_on_axis - axis_ransac * 2,
        point_on_axis + axis_ransac * 2
    ])
    ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], 'r--', 
            linewidth=2, label='RANSAC')
    
    # Plot SVD results
    line_points = np.array([center - svd_axis * 2, center + svd_axis * 2])
    ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], 'g--', 
            linewidth=2, label='SVD')
    
    ax.legend()
    ax.set_title('Axis Comparison')
    
    # Plot angle errors
    ax = fig.add_subplot(122)
    ransac_angle = np.arccos(np.abs(np.dot(axis_ransac, true_axis))) * 180 / np.pi
    svd_angle = np.arccos(np.abs(np.dot(svd_axis, true_axis))) * 180 / np.pi
    
    bars = ax.bar(['RANSAC', 'SVD'], [ransac_angle, svd_angle])
    ax.set_ylabel('Angle Error (degrees)')
    ax.set_title('Axis Direction Error')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}°',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return ransac_angle, svd_angle

def test_cylinder_methods():
    """Test cylinder axis fitting methods"""
    print("\nStarting axis fitting test for cylinders in different directions...")
    
    test_cases = generate_test_cases()
    ransac_errors = []
    svd_errors = []
    
    for case_name, points, true_axis in test_cases:
        print(f"\nTest case: {case_name}")
        print("True axis direction:", true_axis)
        
        # Calculate normal vectors and save visualization results
        normals = pcl_algo.compute_normals(points, k_neighbors=30)
        save_points_with_normals(points, normals, filename=f'normals_{case_name.replace(" ", "_").lower()}.png')
        
        # Method 1: RANSAC fitting
        ransac_result = pcl_algo.fit_cylinder_ransac(
            points,
            distance_threshold=0.01,
            max_iterations=1000,
            k_neighbors=30,
            normal_distance_weight=0.1,
            min_radius=0.1,
            max_radius=1.0
        )
        
        # Method 2: SVD normal vector analysis
        svd_axis = pcl_algo.find_cylinder_axis_svd(normals)
        
        # Visualize results
        ransac_error, svd_error = visualize_results(
            points, true_axis, ransac_result, svd_axis, case_name,
            f'cylinder_test_{case_name.replace(" ", "_").lower()}.png')
        
        ransac_errors.append(ransac_error)
        svd_errors.append(svd_error)
        
        print("RANSAC axis direction:", ransac_result[1])
        print("SVD axis direction:", svd_axis)
        print(f"RANSAC error: {ransac_error:.2f}°")
        print(f"SVD error: {svd_error:.2f}°")
    
    # Create summary comparison plot
    plt.figure(figsize=(10, 6))
    x = range(len(test_cases))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], ransac_errors, width, label='RANSAC')
    plt.bar([i + width/2 for i in x], svd_errors, width, label='SVD')
    
    plt.xlabel('Test Cases')
    plt.ylabel('Angle Error (degrees)')
    plt.title('Comparison of Cylinder Axis Fitting Methods for Different Directions')
    plt.xticks(x, [case[0] for case in test_cases], rotation=45)
    plt.legend(['RANSAC Method', 'SVD Method'])
    plt.tight_layout()
    plt.savefig('cylinder_test_summary.png')
    plt.close()
    
    print("\nTest Results Summary:")
    print("----------------------------------------")
    for i, (case_name, _, _) in enumerate(test_cases):
        print(f"\n{case_name}:")
        print(f"RANSAC method error: {ransac_errors[i]:.2f}°")
        print(f"SVD method error: {svd_errors[i]:.2f}°")
    print("----------------------------------------")
    print("\nAll test result images have been saved to the current directory")

if __name__ == "__main__":
    # Test cylinder axis fitting methods
    test_cylinder_methods()
