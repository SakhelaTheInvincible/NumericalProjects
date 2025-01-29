import CalculateTrajectory
import Video
import Visualisation

video_path = "Ball1.mp4"

coordinates, width, height, bg_color, radius, ball_color = Video.extract_ball_coordinates(video_path)

predicted_trajectory = CalculateTrajectory.predict_trajectory(coordinates, radius)

# Change video resolution if new coordinates exceed the limits
max_x, max_y = predicted_trajectory[:, 0].max(), predicted_trajectory[:, 1].max()
width, height = max(width, max_x + 20), max(height, max_y + 20)

start_index = len(coordinates)  # From which we start aiming at the moving ball

coordinates = CalculateTrajectory.add_trajectory_coordinates(predicted_trajectory, coordinates)

Visualisation.visualisation(coordinates, width, height, bg_color, radius, ball_color, predicted_trajectory, start_index)

