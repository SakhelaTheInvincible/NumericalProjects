import cv2
import FindCircles
import CalculateTrajectories
import Visualisation

# Load image, take shape and convert it to grayscale
image = cv2.imread("Balls3.png")
height, width, _ = image.shape
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = FindCircles.edge_detection(gray_image)

# Detect connected components
components = FindCircles.detect_components(edges)

# Filter and compute centroids
centroids, radii, colors = FindCircles.filter_and_compute_circles(components, image)

# Clustering using DBSCAN (For comparison purposes, also for cases where circles have outlines)
# centroids, radii, colors = FindCircles.dbscan_clustering(edges, image, eps=20, min_samples=20)

# Generate random point near centroids
random_point = CalculateTrajectories.generate_random_point_near_target(centroids, radii, max_distance=50, width=width,
                                                                       height=height)
# Simulate trajectories to all centroids
trajectories = CalculateTrajectories.findTrajectories(random_point, centroids)

# Background Color
color = image[5, 5]
rgb_color = color[::-1] / 255

# Visualization
Visualisation.visualize(centroids, radii, colors, trajectories, width, height, rgb_color)
