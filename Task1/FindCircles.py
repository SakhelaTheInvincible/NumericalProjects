import numpy as np
from CalculateTrajectories import euclidean_distance

from sklearn.cluster import DBSCAN  # This is not used in main code, just for comparing the methods to each other in documentation


# Edge detection
def edge_detection(gray_image, threshold=30):
    # Convert gray image for calculations
    height, width = gray_image.shape
    edges = np.zeros_like(gray_image, dtype=np.uint8)
    gray_image = gray_image.astype(np.int32)

    # Loop through the image, excluding the border pixels
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Calculate the gradient in the x direction (gx) using Sobel operator
            gx = (gray_image[y - 1, x + 1] + 2 * gray_image[y, x + 1] + gray_image[y + 1, x + 1]) - \
                 (gray_image[y - 1, x - 1] + 2 * gray_image[y, x - 1] + gray_image[y + 1, x - 1])

            # Calculate the gradient in the y direction (gy) using Sobel operator
            gy = (gray_image[y + 1, x - 1] + 2 * gray_image[y + 1, x] + gray_image[y + 1, x + 1]) - \
                 (gray_image[y - 1, x - 1] + 2 * gray_image[y - 1, x] + gray_image[y - 1, x + 1])

            # Calculate the magnitude of the gradient
            magnitude = np.sqrt(gx ** 2 + gy ** 2)

            # Apply threshold to determine if the pixel is an edge
            if magnitude > threshold:
                edges[y, x] = 255  # Mark the pixel as an edge (white)

    return edges  # Return the array containing edge detected image


# Connected components detection
def detect_components(edges):
    # Get the dimensions of the input edge-detected image
    height, width = edges.shape
    # Create a boolean array to keep track of visited pixels
    visited = np.zeros_like(edges, dtype=bool)
    components = []

    # Define a depth-first search (DFS) function to explore connected pixels
    def dfs(X, Y, Component):
        # Start the stack with the initial pixel coordinates
        stack = [(X, Y)]
        while stack:
            # Pop the last pixel from the stack
            cx, cy = stack.pop()
            # Skip the pixel if it has been visited or is not an edge pixel
            if visited[cy, cx] or edges[cy, cx] == 0:
                continue
            # Mark the pixel as visited
            visited[cy, cx] = True
            # Add the pixel to the current component
            Component.append((cx, cy))
            # Check neighboring pixels (4-connectivity)
            for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
                # Ensure the neighboring pixel is within the image bounds and not visited
                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    stack.append((nx, ny))  # Add the neighbor to the stack for exploration

    # Loop through each pixel in the edges image
    for y in range(height):
        for x in range(width):
            # If the pixel is an edge and not visited, start a new component
            if edges[y, x] == 255 and not visited[y, x]:
                component = []  # Initialize a new component
                dfs(x, y, component)  # Run DFS to find all connected edge pixels
                if component:  # If any pixels were found, add the component to the list
                    components.append(component)

    return components  # Return the list of detected connected components


# Filter and compute centers of circles, radius and color
def filter_and_compute_circles(components, image, min_size=20):
    centroids = []
    radii = []
    colors = []

    height, _, _ = image.shape

    for component in components:
        # Filter components with minimum size to avoid noise or small objects
        if len(component) >= min_size:
            # Calculate centroid
            cx = int(np.mean([p[0] for p in component]))
            cy = int(np.mean([p[1] for p in component]))
            # Invert Y axis to match the picture (Image shape coordinates start from above)
            centroids.append((cx, height - cy))

            # Calculate radius as the average distance from centroid to all points
            distances = [euclidean_distance((px, py), (cx, cy)) for px, py in component]
            radius = np.mean(distances)
            radii.append(radius)

            # Extract color of centroid in the component and convert to RGB
            color = image[cy, cx]
            rgb_color = [c / 255.0 for c in color[::-1]]
            colors.append(rgb_color)

    return centroids, radii, colors


# DBSCAN for clustering
def dbscan_clustering(edges, image, eps=2, min_samples=5, min_size=20):
    # Extract edge pixel coordinates
    edge_coords = np.column_stack(np.where(edges == 255))
    height, _, _ = image.shape

    # Run DBSCAN on edge pixels
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(edge_coords)  # Built-in DBSCAN

    # Process clusters
    components = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:
            continue  # Skip noise points
        component = edge_coords[clustering.labels_ == cluster_id]
        if len(component) >= min_size:  # Filter small components
            components.append(component)

    # Compute features for each cluster
    centroids, radii, colors = [], [], []
    for component in components:
        # Calculate centroid
        cx = int(np.mean(component[:, 1]))
        cy = int(np.mean(component[:, 0]))
        centroids.append((cx, height - cy))

        # Calculate radius as the average distance from centroid
        distances = [euclidean_distance((px, py), (cx, cy)) for py, px in component]
        radius = np.mean(distances)
        radii.append(radius)

        # Extract centroid color
        color = image[cy, cx]
        rgb_color = [c / 255.0 for c in color[::-1]]
        colors.append(rgb_color)

    return centroids, radii, colors

    # Non-built-in DBSCAN
    # return filter_and_compute_circles(cluster_points(edge_coords), image)


# Find nearest neighbors of points
def get_neighbors(point, points, eps):  # eps is threshold
    neighbors = []
    for i, other_point in enumerate(points):
        if euclidean_distance(point, other_point) < eps:
            neighbors.append(i)  # add index
    return neighbors


def expand_cluster(point_index, neighbors, labels, cluster_id, points, eps, min_samples):
    labels[point_index] = cluster_id  # Assign the cluster id to the core point
    i = 0
    while i < len(neighbors):
        neighbor_index = neighbors[i]
        if labels[neighbor_index] == -1:  # Previously labeled as noise
            labels[neighbor_index] = cluster_id  # Change noise to a member of the cluster
        elif labels[neighbor_index] == 0:  # Unvisited
            labels[neighbor_index] = cluster_id  # Assign the cluster id
            # Get neighbors of the neighbor point
            neighbor_neighbors = get_neighbors(points[neighbor_index], points, eps)
            if len(neighbor_neighbors) >= min_samples:
                neighbors += neighbor_neighbors  # Add neighbors to the search list
        i += 1


def cluster_points(points, eps=20, min_samples=20):
    labels = [0] * len(points)  # 0 means unvisited, -1 means noise
    cluster_id = 0

    for point_index in range(len(points)):
        if labels[point_index] != 0:
            continue  # Skip if already visited

        # Find neighbors of this point
        neighbors = get_neighbors(points[point_index], points, eps)

        if len(neighbors) < min_samples:
            labels[point_index] = -1  # Mark as noise
        else:
            cluster_id += 1  # New cluster
            expand_cluster(point_index, neighbors, labels, cluster_id, points, eps, min_samples)

    # Form clusters from labels
    clusters = []
    for cluster_id in set(labels):
        if cluster_id > 0:  # Only include valid clusters
            cluster = [points[i] for i in range(len(points)) if labels[i] == cluster_id]
            clusters.append(cluster)

    # Return clusters
    return clusters
