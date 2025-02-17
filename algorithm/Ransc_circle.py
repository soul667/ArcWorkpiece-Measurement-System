import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from circle_arc import CircleArc

class CircleFitter:
    def __init__(self):
        self.arc = CircleArc()
        self.fig, self.ax = plt.subplots()

    def fit_circle_ransac(self, points, max_trials=100, residual_threshold=0.1, inlier_ratio=0.8):
        """使用RANSAC算法拟合圆"""
        def model_circle(x):
            cx, cy, r = x
            distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
            return distances - r

        best_params = None
        best_inliers = 0
        best_points = None
        for _ in range(max_trials):
            sample = points[np.random.choice(points.shape[0], 4, replace=False)]
            try:
                self.arc.fit_circle_arc(np.array(sample))
                center = self.arc.first_param[0:2]
                radius = self.arc.first_param[2]

                residuals = model_circle((center[0], center[1], radius))
                inliers = np.sum(np.abs(residuals) < residual_threshold)
                inlier_points = points[np.abs(residuals) < residual_threshold]
                if inliers > best_inliers and inliers / points.shape[0] >= inlier_ratio:
                    best_inliers = inliers
                    best_points = inlier_points
                    best_params = (center[0], center[1], radius)

            except Exception:
                continue

        return best_params, best_points

    def plot_circle(self, center, radius, points, inner_points):
        """绘制拟合结果"""
        fig, ax = plt.subplots()
        # ax.scatter(inner_points[:, 0], inner_points[:, 1], label="Inliers", color="red",s=1)
        # ax.scatter(points[:, 0], points[:, 1], label="Data Points", color="blue",s=1)
        points_set = set(map(tuple, points))
        inner_set = set(map(tuple, inner_points))
        non_inliers = np.array(list(points_set - inner_set))

        # Step 2: Plot Non-Inliers with Distinct Markers
        fig, ax = plt.subplots()

        # Plot non-inliers with larger size and cross markers
        ax.scatter(non_inliers[:, 0], non_inliers[:, 1], 
                    label="Non-Inliers", color="green", s=20, marker='x')

        # Plot inliers with smaller size and circle markers
        ax.scatter(inner_points[:, 0], inner_points[:, 1], 
                    label="Inliers", color="red", s=5, marker='o')

        circle = plt.Circle(center, radius, color="black", fill=False, label="Fitted Circle")
        ax.add_artist(circle)

        ax.set_aspect("equal", adjustable="datalim")
        plt.legend()
        plt.show()
        plt.cla()

# 示例
if __name__ == "__main__":
    np.random.seed(0)
    fitter = CircleFitter()
    true_center = (2, 3)
    true_radius = 5

    points = fitter.generate_circle_points(true_center, true_radius, num_points=1000, angle_range=25, noise=0.01, outliers=50)

    fitted_params, points_inner = fitter.fit_circle_ransac(points, inlier_ratio=0.7, residual_threshold=0.07)
    if fitted_params:
        print(f"Fitted center: ({fitted_params[0]:.2f}, {fitted_params[1]:.2f})")
        print(f"Fitted radius: {fitted_params[2]:.2f}")
        fitter.plot_circle(fitted_params[:2], fitted_params[2], points, points_inner)

        fitter.arc.fit_circle_arc(points_inner)
        center = fitter.arc.first_param[0:2]
        radius = fitter.arc.first_param[2]

        print(f"拟合的圆心: ({center[0]:.2f}, {center[1]:.2f})")
        print(f"拟合的半径: {radius:.2f}")
    else:
        print("RANSAC fitting failed to find a valid circle.")
