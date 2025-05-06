from minimize_test import generate_partial_cylinder_points
import numpy as np

from py_cylinder_fitting import BestFitCylinder
from skspatial.objects import Points
points = [[2, 0, 0], [0, 2, 0], [0, -2, 0], [2, 0, 4], [0, 2, 4], [0, -2, 4]]
best_fit_cylinder = BestFitCylinder(Points(points))
# best_fit_cylinder = BestFitCylinder(Points(points))
print(best_fit_cylinder.vector)
