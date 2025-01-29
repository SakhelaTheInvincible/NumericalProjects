import numpy as np
from scipy.optimize import root


# Function to generate random point (center of our circle)
def generate_random_point(coordinates, width, height, radius, max_distance=50):
    # We try 1000 times to generate the random ball coordinate
    for _ in range(1000):
        # Choose coordinates randomly with 20 margin (so point isn't on the corners of screen)
        rand_x = np.random.randint(20, width - 20)
        rand_y = np.random.randint(20, height - 20)

        # I don't want the ball to be on the path where main ball moves,
        # So I choose a random point which will be outside of this path with some distance
        if all(euclidean_distance((rand_x, rand_y), (x, y)) > 2 * radius + max_distance for x, y in coordinates):
            return rand_x, rand_y

    raise ValueError("Impossible to generate a valid random point.")


# Function to calculate trajectory given the parameters
def calculate_trajectory(target_position, start_position, total_time, dt, params):
    x_start, y_start = start_position

    # Use the shooting method to find the initial velocity that hits the target
    initial_velocity = shooting_method(
        equations, 0, [x_start, y_start, 0, 0],  # Start with zero initial velocity
        target_position, total_time, dt, params
    )

    # Integrate trajectory with the calculated initial velocity
    state0 = [x_start, y_start, initial_velocity[0], initial_velocity[1]]
    trajectory = integrate_trajectory(equations, 0, state0, total_time, dt, params)

    # Ensure the trajectory matches the desired time range
    frames = int(total_time / dt)
    if len(trajectory) < frames:
        trajectory = np.vstack([trajectory, np.full((frames - len(trajectory), 4), trajectory[-1])])

    return trajectory


# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Runge-Kutta-4 integration method with given formulas
def rk4_step(f, t, state, dt, params):
    k1 = f(t, state, params)
    k2 = f(t + dt / 2, state + dt * k1 / 2, params)
    k3 = f(t + dt / 2, state + dt * k2 / 2, params)
    k4 = f(t + dt, state + dt * k3, params)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# Euler's Method integration
def euler_step(f, t, state, dt, params):
    return state + dt * f(t, state, params)


# Trapezium Step Method integration
def trapezium_step(f, t, state, dt, params):
    # Define the implicit equation to solve
    def implicit_eq(y_next):
        return y_next - state - 0.5 * dt * (f(t + dt, y_next, params) + f(t, state, params))

    # Solve the equation using a root-finding method
    sol = root(implicit_eq, state)  # Initial guess is the current state
    if not sol.success:
        raise RuntimeError("Failed for Trapezium method")

    return sol.x


# Implicit Euler Step Method integration
def implicit_euler_step(f, t, state, dt, params):
    # Define the implicit equation to solve
    def implicit_eq(y_next):
        return y_next - state - dt * f(t + dt, y_next, params)

    # Solve the equation using a root-finding method
    sol = root(implicit_eq, state)  # Initial guess is the current state
    if not sol.success:
        raise RuntimeError("Newton-Raphson failed for Implicit Euler")

    return sol.x


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


# Integrate trajectory
def integrate_trajectory(equations_method, t0, state0, t_end, dt, params, radius=0):
    # Initialize the time and state and trajectory
    t = t0
    state = state0
    trajectory = [state]
    # Integrate the trajectory until the end time
    while t < t_end:
        # Calculate the next state using the Runge-Kutta-4 method
        state = rk4_step(equations_method, t, state, dt, params)

        # For comparison purposes:
        # state = euler_step(equations_method, t, state, dt, params)
        # state = implicit_euler_step(equations_method, t, state, dt, params)
        # state = trapezium_step(equations_method, t, state, dt, params)
        # Check if the ball has hit the ground
        if state[1] <= radius:
            break
        # Append the current state to the trajectory
        trajectory.append(state)
        # Increment the time
        t += dt
    # Return the trajectory
    return np.array(trajectory)


# Add predicted trajectory coordinates to the new trajectory
def add_trajectory_coordinates(trajectory, coordinates):
    # Add trajectory coordinates to the given coordinates list
    for state in trajectory:
        coordinates.append((state[0], state[1]))
    return coordinates


def predict_trajectory(coordinates, radius):
    # Constants for the motion equations (They must be same in trajectory calculations)
    params = (9.81, 0.1, 1)  # Gravity, Drag Coefficient (Air Resistance), Mass
    TIME_STEP = 0.1

    last_x, last_y = coordinates[len(coordinates) - 1]

    # Backward Euler for velocities
    last_vx = (coordinates[-1][0] - coordinates[-4][0]) / TIME_STEP
    last_vy = (coordinates[-1][1] - coordinates[-4][1]) / TIME_STEP

    return integrate_trajectory(
        equations, 0, [last_x, last_y, last_vx, last_vy],
        20, TIME_STEP, params, radius
    )
