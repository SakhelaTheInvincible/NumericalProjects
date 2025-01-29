import cv2


# Function to detect ball for each frame
def detect_ball(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Edge detection using cv2
    edges = cv2.Canny(gray, 50, 150)
    # Finding contours of circle
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Fit a circle to the contour
        center, radius = cv2.minEnclosingCircle(contour)
        if radius > 5:  # Filter small noise
            x, y = int(center[0]), int(center[1])
            r = int(radius)
            # Extract the ball's color using the center of the ball
            color = frame[y, x].tolist()  # Get BGR color
            return x, y, r, color
    return None


# Function to process the video and extract ball coordinates and colors
def extract_ball_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)
    coordinates = []

    # Get shape of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Values for finding out background color
    selected, radius, ball_color = 0, 0, [1.0, 1.0, 1.0]
    bg_color = [1.0, 1.0, 1.0]

    # Processing each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:  # Video ended
            break

        # Detect the ball, its coordinates, radius and color to the list (In some videos radius might change dynamically)
        detected = detect_ball(frame)
        if detected:
            x, y, r, color = detected

            # Extract background color, radius of the ball and its color from the first frame
            if selected == 5:
                bg_color = frame[5, 5][::-1] / 255
                radius = r
                ball_color = [c / 255.0 for c in color[::-1]]
                selected = True

            if selected < 5:
                selected += 1

            coordinates.append((x, height - y))

    cap.release()

    # Return coordinates of a ball, and video width, height and background color
    return coordinates, width, height, bg_color, radius, ball_color
