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
#include <Eigen/SVD>

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
    std::vector<ssize_t> shape = {static_cast<ssize_t>(cloud_normals->size()), 3};
    py::array_t<double> normals(shape);
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

Eigen::Vector3d find_cylinder_axis_svd(py::array_t<double> normals_array) {
    // Convert numpy array to Eigen matrix
    auto normals_buffer = normals_array.request();
    if (normals_buffer.ndim != 2 || normals_buffer.shape[1] != 3) {
        throw std::runtime_error("Input normals must be an Nx3 array");
    }

    const size_t num_points = normals_buffer.shape[0];
    double* ptr = static_cast<double*>(normals_buffer.ptr);
    
    // Create matrix A (n x 3) where each row is a normal vector
    Eigen::MatrixXd A(num_points, 3);
    for (size_t i = 0; i < num_points; i++) {
        A(i, 0) = ptr[i * 3];
        A(i, 1) = ptr[i * 3 + 1];
        A(i, 2) = ptr[i * 3 + 2];
    }

    // Compute A^T * A
    Eigen::Matrix3d ATA = A.transpose() * A;

    // Perform SVD on A^T * A
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ATA, Eigen::ComputeFullV);
    
    // Get the eigenvector corresponding to the smallest singular value
    // This is the last column of V and represents the cylinder axis direction
    Eigen::Vector3d axis_direction = svd.matrixV().col(2);
    
    // Normalize the axis direction
    axis_direction.normalize();
    
    return axis_direction;
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, double, int, py::array_t<double>> fit_cylinder_ransac(
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
    ne.setKSearch(k_neighbors);
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
    seg.setRadiusLimits(min_radius, max_radius);
    
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
    Eigen::Vector3d point_on_axis(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    Eigen::Vector3d axis_direction(coefficients->values[3], coefficients->values[4], coefficients->values[5]);
    double radius = coefficients->values[6];

    // Normalize axis direction
    axis_direction.normalize();

    // Get the maximum number of iterations that were allowed
    int max_iters = seg.getMaxIterations();

    // Create filtered points array
    std::vector<ssize_t> shape = {static_cast<ssize_t>(inliers->indices.size()), 3};
    py::array_t<double> filtered_points(shape);
    auto filtered_buffer = filtered_points.request();
    double* filtered_ptr = static_cast<double*>(filtered_buffer.ptr);

    // Copy filtered points to numpy array
    for (size_t i = 0; i < inliers->indices.size(); i++) {
        int idx = inliers->indices[i];
        filtered_ptr[i * 3] = ptr[idx * 3];
        filtered_ptr[i * 3 + 1] = ptr[idx * 3 + 1];
        filtered_ptr[i * 3 + 2] = ptr[idx * 3 + 2];
    }
    
    return std::make_tuple(point_on_axis, axis_direction, radius, max_iters, filtered_points);
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

    m.def("find_cylinder_axis_svd", &find_cylinder_axis_svd, py::arg("normals"),
          R"pbdoc(
            Find cylinder axis direction using SVD method on surface normals.
            
            This method solves the equation AX=0 using SVD, where A is the matrix
            of surface normals. The cylinder axis is the eigenvector corresponding
            to the smallest singular value.
            
            Args:
                normals (numpy.ndarray): Nx3 array of surface normals
                
            Returns:
                numpy.ndarray: 3D normalized vector representing the cylinder axis direction
                
            Raises:
                RuntimeError: If axis computation fails
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
                tuple: (point_on_axis, axis_direction, radius, iterations, filtered_points)
                    - point_on_axis (numpy.ndarray): 3D point on cylinder axis
                    - axis_direction (numpy.ndarray): normalized direction vector of cylinder axis
                    - radius (float): radius of the cylinder
                    - iterations (int): maximum number of RANSAC iterations allowed
                    - filtered_points (numpy.ndarray): Mx3 array of points that fit the cylinder model
                    
            Raises:
                RuntimeError: If cylinder fitting fails
          )pbdoc");
}
