#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

py::array_t<double> compute_normals(
    py::array_t<double> points_array,
    int k_neighbors = 50
) {
    // Convert numpy array to PCL point cloud
    auto points_buffer = points_array.request();
    if (points_buffer.ndim != 2 || points_buffer.shape[1] != 3) {
        throw std::runtime_error("Input points must be an Nx3 array");
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    double* ptr = static_cast<double*>(points_buffer.ptr);
    cloud->points.resize(points_buffer.shape[0]);
    for (size_t i = 0; i < points_buffer.shape[0]; i++) {
        cloud->points[i].x = ptr[i * 3];
        cloud->points[i].y = ptr[i * 3 + 1];
        cloud->points[i].z = ptr[i * 3 + 2];
    }

    // Create the normal estimation object
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(k_neighbors);
    ne.compute(*cloud_normals);

    // Create output numpy array
    py::array_t<double> normals({cloud_normals->size(), 3});
    auto normals_buffer = normals.request();
    double* normals_ptr = static_cast<double*>(normals_buffer.ptr);

    // Copy normals to numpy array
    for (size_t i = 0; i < cloud_normals->size(); i++) {
        normals_ptr[i * 3] = cloud_normals->points[i].normal_x;
        normals_ptr[i * 3 + 1] = cloud_normals->points[i].normal_y;
        normals_ptr[i * 3 + 2] = cloud_normals->points[i].normal_z;
    }

    return normals;
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, double> fit_cylinder_ransac(
    py::array_t<double> points_array,
    double distance_threshold = 0.01,
    int max_iterations = 1000,
    int k_neighbors = 50,
    double normal_distance_weight = 0.1,
    double min_radius = 0.0,
    double max_radius = 1.0
) {
    // Convert numpy array to PCL point cloud
    auto points_buffer = points_array.request();
    if (points_buffer.ndim != 2 || points_buffer.shape[1] != 3) {
        throw std::runtime_error("Input points must be an Nx3 array");
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    double* ptr = static_cast<double*>(points_buffer.ptr);
    cloud->points.resize(points_buffer.shape[0]);
    for (size_t i = 0; i < points_buffer.shape[0]; i++) {
        cloud->points[i].x = ptr[i * 3];
        cloud->points[i].y = ptr[i * 3 + 1];
        cloud->points[i].z = ptr[i * 3 + 2];
    }

    // Create the normal estimation object
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(k_neighbors);  // Use specified number of nearest neighbors
    ne.compute(*cloud_normals);

    // Create the segmentation object
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    
    // Configure the segmentation parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(normal_distance_weight);
    seg.setMaxIterations(max_iterations);
    seg.setDistanceThreshold(distance_threshold);
    seg.setRadiusLimits(min_radius, max_radius);  // Min/Max radius limits
    
    seg.setInputCloud(cloud);
    seg.setInputNormals(cloud_normals);

    // Segment the cylinder
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        throw std::runtime_error("Could not estimate a cylindrical model for the given dataset");
    }

    // Extract cylinder parameters
    // Coefficients: [point_on_axis.x point_on_axis.y point_on_axis.z axis_direction.x axis_direction.y axis_direction.z radius]
    Eigen::Vector3d point_on_axis(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    Eigen::Vector3d axis_direction(coefficients->values[3], coefficients->values[4], coefficients->values[5]);
    double radius = coefficients->values[6];

    // Normalize axis direction
    axis_direction.normalize();

    return std::make_tuple(point_on_axis, axis_direction, radius);
}

void init_cylinder_fitting(py::module& m) {
    m.def("compute_normals", &compute_normals, py::arg("points"),
          py::arg("k_neighbors") = 50,
          R"pbdoc(
            Compute surface normals for a point cloud.
            
            Args:
                points (numpy.ndarray): Nx3 array of 3D points
                k_neighbors (int): Number of nearest neighbors to use for normal estimation
                
            Returns:
                numpy.ndarray: Nx3 array of computed surface normals
                
            Raises:
                RuntimeError: If normal computation fails
          )pbdoc");

    m.def("fit_cylinder_ransac", &fit_cylinder_ransac, py::arg("points"),
          py::arg("distance_threshold") = 0.01, py::arg("max_iterations") = 1000,
          py::arg("k_neighbors") = 50, py::arg("normal_distance_weight") = 0.1,
          py::arg("min_radius") = 0.0, py::arg("max_radius") = 1.0,
          R"pbdoc(
            Fit a cylinder to point cloud data using RANSAC.
            
            Args:
                points (numpy.ndarray): Nx3 array of 3D points
                distance_threshold (float): Maximum distance from points to cylinder surface
                max_iterations (int): Maximum number of RANSAC iterations
                k_neighbors (int): Number of nearest neighbors for normal estimation
                normal_distance_weight (float): Weight for normals in model fitting
                min_radius (float): Minimum cylinder radius to consider
                max_radius (float): Maximum cylinder radius to consider
                
            Returns:
                tuple: (point_on_axis, axis_direction, radius)
                    - point_on_axis (numpy.ndarray): 3D point on cylinder axis
                    - axis_direction (numpy.ndarray): normalized direction vector of cylinder axis
                    - radius (float): radius of the cylinder
                    
            Raises:
                RuntimeError: If cylinder fitting fails
          )pbdoc");
}
