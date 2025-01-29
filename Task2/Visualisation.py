import matplotlib.pyplot as plt
from matplotlib import animation
import random
import matplotlib
import CalculateTrajectory

# delete this if code doesn't work
matplotlib.use('TkAgg')


# This function outputs our created ball animation
# Which is almost the same as original video
def visualize_created_video(coordinates, width, height, bg_color, ball_radius, ball_color, new_trajectory):

    # Create Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title('Ball Animation with Predicted Trajectory')

    # Set the background color
    ax.set_facecolor(bg_color)

    # Create Main Ball
    main_ball, = ax.plot([], [], 'o', markersize=0)
    main_ball.set_markersize(ball_radius * 2)
    main_ball.set_color(ball_color)

    # plot the predicted trajectory
    ax.plot(new_trajectory[:, 0], new_trajectory[:, 1], '--', color='red', label="Predicted Trajectory")

    # Initialization for the FuncAnimation
    def init():
        main_ball.set_data([], [])
        return main_ball,

    # Update animation for new frame (Again for FuncAnimation)
    def update(frame):
        # If ball is moving
        frame *= 2  # For faster and smoother animation
        if frame < len(coordinates):
            # Extract and assign new values for coordinates
            main_x, main_y = coordinates[frame]
            main_ball.set_data([main_x], [main_y])
        return main_ball,

    # Create the animation and display it in 60 fps
    ani = animation.FuncAnimation(fig, update, frames=len(coordinates), init_func=init, blit=True,
                                  interval=1000 / 60, cache_frame_data=False)
    plt.legend()
    plt.show()


# Function to visualize the ball movements
def visualisation(coordinates, width, height, bg_color, ball_radius, ball_color, new_trajectory, start_index, dt=0.1):
    # Show created animation of the moving ball only
    visualize_created_video(coordinates, width, height, bg_color, ball_radius, ball_color, new_trajectory)

    # Decide for which position from the new trajectory of the moving ball we will hit it with our own (Randomly)
    target_index = random.randint(start_index, len(coordinates) - 5)
    # Coordinate length is same as the number of the frames we will have
    # And because we had dt before, we need to adjust this with new time
    total_time = target_index / (1/dt)

    # Take random point from which we will throw a ball (Start point)
    random_point = CalculateTrajectory.generate_random_point(coordinates, width, height, ball_radius)

    # Constants for the motion equations
    g = 9.81  # Gravity
    k = 0.1  # Drag Coefficient (Air Resistance)
    m = 1  # Mass
    params = (g, k, m)

    # Target point (End point of our thrown ball)
    target_position = coordinates[target_index][:2]

    # Calculate trajectory
    thrown_trajectory = CalculateTrajectory.calculate_trajectory(
        target_position, random_point, total_time, dt, params
    )

    # Scale factor helps us to scale down the trajectory
    # for perfect animation for hitting a ball
    scale_factor = len(thrown_trajectory) // target_index

    # Create Plot for animation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title('Hitting The Ball Animation')

    # Set the background color
    ax.set_facecolor(bg_color)

    # Create Main Ball and Our Thrown Ball
    main_ball, = ax.plot([], [], 'o', markersize=0)
    thrown_ball, = ax.plot([], [], 'o', color='green', markersize=8, label="Thrown Ball")
    ax.plot(thrown_trajectory[:, 0], thrown_trajectory[:, 1], '--', color='orange')

    main_ball.set_markersize(ball_radius * 2)
    main_ball.set_color(ball_color)
    thrown_ball.set_markersize(ball_radius * 2)

    # Used for thrown ball animation
    frame2 = 0

    # Initialization for the FuncAnimation
    def init():
        main_ball.set_data([], [])
        thrown_ball.set_data([], [])
        return main_ball, thrown_ball

    # Update animation for new frame (Again for FuncAnimation)
    def update(frame):
        nonlocal frame2, scale_factor
        frame *= 2  # For faster and smoother animation
        frame2 = frame * scale_factor

        # If main ball is moving
        if frame < len(coordinates):
            # Extract values from coordinates
            main_x, main_y = coordinates[frame]
            # Assign new position for the ball, radius and color
            main_ball.set_data([main_x], [main_y])

        # Adjust to last coordinate
        if frame2 + scale_factor > len(thrown_trajectory):
            frame2 = len(thrown_trajectory) - 1

        # If thrown ball is moving
        if frame2 < len(thrown_trajectory):
            # Assign new position for the ball
            thrown_x, thrown_y = thrown_trajectory[frame2][:2]
            thrown_ball.set_data([thrown_x], [thrown_y])

            # If the both of the balls are moving
            if frame < len(coordinates):
                # Calculate distance between Centers of the balls
                distance = CalculateTrajectory.euclidean_distance((main_x, main_y), (thrown_x, thrown_y))

                # If distance is smaller than radius sum of these balls,
                # Then our thrown ball hits the main ball, so we stop the animation
                if distance <= 2 * ball_radius + 2:  # Slight offset for better animation
                    main_ball.set_data([], [])
                    thrown_ball.set_data([], [])
                    ani.event_source.stop()

        return main_ball, thrown_ball

    # Create the animation and display it in 60 fps
    ani = animation.FuncAnimation(fig, update, frames=max(len(coordinates), len(thrown_trajectory)), init_func=init,
                                  blit=True, interval=1000 / 60, cache_frame_data=False)

    plt.show()
