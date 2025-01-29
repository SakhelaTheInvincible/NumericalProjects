import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import numpy as np
from CalculateTrajectories import euclidean_distance

# delete this if code doesn't work
matplotlib.use('TkAgg')


# Function to create circles
def createCircles(centroids, radii, colors, trajectories):
    # Create circle objects with zipping all circle parameters and trajectories to reach these circles
    circles = [
        {
            "circle": Circle(centroid, radius, color=color),
            "centroid": np.array(centroid),
            "radius": radius,
            "trajectory": trajectory
        }
        for centroid, radius, color, trajectory in zip(centroids, radii, colors, trajectories)
    ]

    return circles


# Function to visualize the animation (speed_factor performs the same here as fps)
def visualize(centroids, radii, colors, trajectories, width, height, color, speed_factor=60):
    # Create circle objects
    circles = createCircles(centroids, radii, colors, trajectories)

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_title("Hitting Circles")

    # Set the background color
    ax.set_facecolor(color)

    # Add circles to the plot
    for circle_data in circles:
        ax.add_patch(circle_data["circle"])

    # Plot the moving point
    moving_point, = ax.plot([], [], 'mo')  # Moving point
    trajectory_lines = []  # Trajectory lines to the circles
    current_trajectory = None  # Active trajectory
    frame_index = 0  # Frame index for the current trajectory

    # Initialization function for FuncAnimation
    def init():
        moving_point.set_data([], [])
        for line in trajectory_lines:
            line.remove()
        trajectory_lines.clear()
        return moving_point,

    # Function to detect collisions
    def detect_collision(x, y):
        # Check for all existing circles
        for circle in circles:
            # Extract values
            cx, cy = circle["centroid"]
            radius = circle["radius"]

            # If Euclidean distance is less than a radius, then the moving ball hits the circle
            if euclidean_distance((cx, cy), (x, y)) <= radius:
                return circle
        return None

    # Remove a circle from the plot and the list
    def remove_circle(circle):
        circle["circle"].remove()
        circles.remove(circle)

    # Update function for FuncAnimation
    def update(frame):
        nonlocal trajectory_lines, current_trajectory, frame_index

        if not circles:
            return moving_point,  # No more circles to process

        # If no selected trajectory, initialize the first circle's trajectory
        if current_trajectory is None:
            current_circle = circles[0]
            current_trajectory = current_circle["trajectory"]
            frame_index = 0  # Reset frame index

            # Draw trajectory line
            trajectory_line, = ax.plot(current_trajectory[:, 0], current_trajectory[:, 1], '--', color='orange')
            trajectory_lines.append(trajectory_line)

        # Update moving point position along the trajectory
        if frame_index < len(current_trajectory):
            x, y, _, _ = current_trajectory[frame_index]
            moving_point.set_data([x], [y])

            # Check for collision
            collision_circle = detect_collision(x, y)
            if collision_circle:
                # Remove the collided circle and reset trajectory
                remove_circle(collision_circle)
                current_trajectory = None  # Reset to pick the next circle trajectory

                return moving_point,  # Skip the rest of the frame

            frame_index += speed_factor  # Increment frame index based on speed factor

        # If trajectory is finished, reset for the next circle
        if frame_index >= len(current_trajectory):
            current_trajectory = None

        return moving_point, *trajectory_lines

    # Function to generate frames for FuncAnimation
    # We generate our own frames with frame_index, so this function does nothing
    def frames_generator():
        while circles:
            yield None

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=frames_generator,
        init_func=init, blit=False, interval=1000/60, repeat=False,
        cache_frame_data=False
    )

    plt.show()
