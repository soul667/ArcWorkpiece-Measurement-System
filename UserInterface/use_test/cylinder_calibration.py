import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class CylinderModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.axis = None
        self.radius = None
        self.center = None

    def fit(self, X, y=None):
        # X is an Nx3 matrix representing the point cloud
        # Step 1: Estimate the axis direction using PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)  # We only care about the 2D plane orthogonal to the axis
        pca.fit(X)
        
        self.axis = pca.components_[2]  # The principal axis is the 3rd component
        
        # Step 2: Project points onto a plane orthogonal to the axis
        axis = self.axis / np.linalg.norm(self.axis)
        projected_points = X - (X @ axis)[:, None] * axis

        # Step 3: Fit a circle in the 2D plane using the least-squares method
        center, radius = self._fit_circle(projected_points)
        self.center = center
        self.radius = radius

        return self

    def predict(self, X):
        # Returns the residuals for each point relative to the cylinder
        axis = self.axis / np.linalg.norm(self.axis)
        projected_points = X - (X @ axis)[:, None] * axis
        distances = np.linalg.norm(projected_points - self.center, axis=1)
        return distances - self.radius

    def _fit_circle(self, points):
        # Fit a circle to 2D points using least squares
        def cost(params):
            cx, cy, r = params
            residuals = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r
            return residuals

        from scipy.optimize import least_squares
        center_guess = np.mean(points, axis=0)
        radius_guess = np.std(np.linalg.norm(points - center_guess, axis=1))
        result = least_squares(cost, [center_guess[0], center_guess[1], radius_guess])
        return np.array(result.x[:2]), result.x[2]


class CylinderCalibrator:
    def __init__(self):
        self.model = RANSACRegressor(estimator=CylinderModel(), min_samples=0.5, residual_threshold=0.1)

    def calibrate(self, point_cloud):
        """Calibrate the cylinder given a point cloud."""
        self.model.fit(point_cloud)
        cylinder = self.model.estimator_
        return {
            "center": cylinder.center,
            "axis": cylinder.axis,
            "radius": cylinder.radius,
        }

# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_points = 1000
    axis = np.array([0, 0, 1])
    center = np.array([1, 1, 0])
    radius = 5

    # Generate random points around a cylinder
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(0, 10, n_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    points = np.column_stack((x, y, z))

    # Add noise
    points += np.random.normal(0, 0.1, points.shape)

    # Perform calibration
    calibrator = CylinderCalibrator()
    result = calibrator.calibrate(points)

    print("Calibrated Cylinder Parameters:")
    print(f"Center: {result['center']}")
    print(f"Axis: {result['axis']}")
    print(f"Radius: {result['radius']}")
