import numpy as np
from scipy.optimize import root


# Function to generate random point (center of our circle)
def generate_random_point_near_target(centroids, radii, max_distance=50, width=800, height=800):
    # We try 1000 times to generate the random ball coordinate
    for _ in range(1000):
        # Choose coordinates randomly with 20 margin (so point isn't on the corners of screen)
        rand_x = np.random.randint(20, width - 20)
        rand_y = np.random.randint(20, height - 20)

        # I don't want the ball to be in the circles itself,
        # So I choose a random point which will be outside of them with some distance
        if all(euclidean_distance((rand_x, rand_y), c) > max_distance + r for c, r in zip(centroids, radii)):
            return rand_x, rand_y

    raise ValueError("Unable to generate a valid random point near centroids.")


# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Equations of motion (From slides)
def equations(t, state, params):
    x, y, vx, vy = state
    g, k, m = params
    speed = np.sqrt(vx ** 2 + vy ** 2)
    drag = (k / m) * speed  # Drag force
    dx_dt = vx
    dy_dt = vy
    dvx_dt = -drag * vx / speed  # Drag in x-direction
    dvy_dt = -g - drag * vy / speed  # Gravity + drag in y-direction
    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])


# Runge-Kutta-4 integration method with given formulas
def rk4_step(f, t, state, dt, params):
    k1 = f(t, state, params)
    k2 = f(t + dt / 2, state + dt * k1 / 2, params)
    k3 = f(t + dt / 2, state + dt * k2 / 2, params)
    k4 = f(t + dt, state + dt * k3, params)

    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# Integrate trajectory
def integrate_trajectory(equations_method, t0, state0, t_end, dt, params):
    # Initialize the time and state and trajectory
    t = t0
    state = state0
    trajectory = [state]
    # Integrate the trajectory until the end time
    while t < t_end:
        # Calculate the next state using the Runge-Kutta-4 method
        state = rk4_step(equations_method, t, state, dt, params)
        # Check if the ball has hit the ground
        if state[1] <= 0:
            break
        # Append the current state to the trajectory
        trajectory.append(state)
        # Increment the time
        t += dt
    # Return the trajectory
    return np.array(trajectory)


# Custom root-finding method (Newton-Raphson with finite differences).
def custom_root_finding(residual_func, initial_guess, tol=1e-6, max_iter=100):
    x = np.array(initial_guess, dtype=float)  # Current guess for the solution

    for iteration in range(max_iter):
        # Compute residual
        residuals = residual_func(x)
        residual_norm = np.linalg.norm(residuals)

        # Check for convergence
        if residual_norm < tol:
            return x

        # Approximate Jacobian using finite differences
        epsilon = tol
        n = len(x)
        jacobian = np.zeros((n, n))  # Jacobian matrix

        for j in range(n):
            x_perturbed = x.copy()
            x_perturbed[j] += epsilon
            jacobian[:, j] = (residual_func(x_perturbed) - residuals) / epsilon

        # Solve for delta_x using Newton-Raphson update
        delta_x = np.linalg.solve(jacobian, -residuals)

        # Update the solution guess
        x += delta_x

    raise ValueError("Custom root-finding did not converge within the maximum iterations.")


# Shooting method
def shooting_method(equations_method, t0, state0, target_point, t_end, dt, params, tol=1e-6):
    x_target, y_target = target_point

    # Define the residual function that calculates the difference between the calculated final position and the target position
    def residual(initial_velocity):
        # Set the initial state with the given initial velocity
        state = np.array([state0[0], state0[1], initial_velocity[0], initial_velocity[1]])
        # Integrate the trajectory using the given equations of motion, initial state, time range, and parameters
        trajectory = integrate_trajectory(equations_method, t0, state, t_end, dt, params)
        # Calculate the final position from the trajectory
        final_position = trajectory[-1][:2]
        return final_position - np.array([x_target, y_target])

    # Define the initial guess for the initial velocity
    initial_guess = np.array([(x_target - state0[0]) / t_end, (y_target - state0[1]) / t_end])

    # Return custom non-built-in root finding:
    # return custom_root_finding(residual, initial_guess, tol)

    # Use scipy root to find the solution
    result = root(residual, initial_guess, tol=tol)

    # return the result
    if result.success:
        return result.x
    else:
        # For debugging purposes
        raise ValueError("Shooting method did not converge")


def findTrajectories(start_point, centroids):
    # Constants
    g = 9.81  # m/s^2
    k = 0.47  # Drag Coefficient
    m = 1  # Mass
    params = (g, k, m)
    t0 = 0  # Time start
    tend = 5  # Time end
    TIME_STEP = 0.001  # Time step

    x0, y0 = start_point

    trajectories = []

    for centroid in centroids:
        # Calculate initial velocity towards each centroid
        initial_velocity = shooting_method(equations, t0, [x0, y0, 0, 0], centroid, tend, TIME_STEP, params)

        # Simulate trajectory using the calculated velocity
        trajectory = integrate_trajectory(equations, t0, [x0, y0, *initial_velocity], tend, TIME_STEP, params)

        trajectories.append(trajectory)  # Store trajectory for each centroid

    return trajectories
