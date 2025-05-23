import math

import numpy as np
from scipy.optimize import minimize

from global_settings import *


# Not used anymore, we just take the major axis

# estimate real radius, using major and minor axes
def estimate_undistorted_radius(major, minor):
    a = major / 2
    b = minor / 2
    r_undistorted = math.sqrt(a * b)  # geometric mean
    return r_undistorted

# calculate error for camera position, based in distances to ball and ball positions for optimization
def calculate_camera_error(pos, distances, ball_positions, center=TRIANGLE_CENTER):
    """
    Calculate error between expected distances and actual distances,
    taking into account that camera is looking at triangle center.
    """
    pos = np.array(pos)
    error = 0

    # Vector from camera to center (viewing direction)
    to_center = center - pos
    to_center = to_center / np.linalg.norm(to_center)

    for ball_pos, target_dist in zip(ball_positions, distances):
        # Vector from camera to ball
        to_ball = ball_pos - pos
        dist = np.linalg.norm(to_ball)

        # Calculate angle between viewing direction and ball direction
        to_ball_norm = to_ball / dist
        cos_angle = np.dot(to_ball_norm, to_center)

        # Penalize both distance error and deviation from expected viewing angle
        angle_error = 1 - cos_angle  # 0 when looking directly at ball
        dist_error = (dist - target_dist) ** 2

        error += dist_error + 5 * angle_error  # Weight angle error more

    return error


# estimate camera position, using calculate_camera_error
def estimate_camera_position(red_dist, green_dist, blue_dist, initial_guess=None):
    """
    Estimate camera position using optimization.
    Takes into account that camera is always looking at triangle center.
    """
    # Ball positions in world space (all at height 18)
    # Equilateral triangle with center at (4.5, 18, 0), side length 9
    # Height of triangle: h = 9 * sqrt(3) / 2 ≈ 7.794
    h = TRIANGLE_SIZE * np.sqrt(3) / 2
    ball_positions = [RED_POSITION, GREEN_POSITION, BLUE_POSITION]

    distances = [red_dist, green_dist, blue_dist]

    if initial_guess is None:
        # Make initial guess based on average distance
        avg_dist = sum(distances) / 3
        initial_guess = [TRIANGLE_CENTER, POLE_SIZE, avg_dist]

    # Use optimization to find best camera position
    result = minimize(
        calculate_camera_error,
        initial_guess,
        args=(distances, ball_positions),
        method='Nelder-Mead'
    )

    return result.x

# calculate location of apex, based on ball positions and distance to balls
def apex_coordinates(a, b, c, side_length=TRIANGLE_SIZE):
    """
    Compute the 3D coordinates of the apex of a pyramid over an equilateral triangle base,
    given distances from the apex to each corner. The triangle vertices are defined by
    RED_POSITION, GREEN_POSITION, and BLUE_POSITION from global_settings.

    Parameters:
    a : float
        Distance from apex to RED_POSITION.
    b : float
        Distance from apex to GREEN_POSITION.
    c : float
        Distance from apex to BLUE_POSITION.
    side_length : float
        Length of the side of the equilateral triangle base. It is assumed that
        RED_POSITION, GREEN_POSITION, BLUE_POSITION from global_settings
        form such a triangle.

    Returns:
    (x, y, z) : tuple of floats
        Global 3D coordinates of the apex point.
    Raises:
    ValueError: If no real solution exists (e.g., distances are too short).
    """
    s = side_length
    h_triangle = (s * np.sqrt(3)) / 2

    # Calculate coordinates (x_local, y_local, z_local) of the apex
    # in a local coordinate system where:
    # - Vertex A (RED_POSITION) is at the origin (0,0,0).
    # - Vertex B (GREEN_POSITION) is at (s,0,0) on the local x-axis.
    # - Vertex C (BLUE_POSITION) is at (s/2, h_triangle, 0) in the local xy-plane.
    # 'a' is distance to A, 'b' to B, 'c' to C.

    # x_local: derived from a^2 = x^2+y^2+z^2 and b^2 = (x-s)^2+y^2+z^2
    # b^2-a^2 = (x-s)^2-x^2 = x^2-2sx+s^2-x^2 = s^2-2sx
    # 2sx = s^2 - (b^2-a^2)
    x_local = (s**2 - (b**2 - a**2)) / (2 * s)

    # y_local: derived from a^2 = x^2+y^2+z^2 and c^2 = (x-s/2)^2+(y-h_triangle)^2+z^2
    # c^2-a^2 = (x-s/2)^2+(y-h_triangle)^2 - x^2-y^2
    # c^2-a^2 = x^2-sx+s^2/4 + y^2-2yh_triangle+h_triangle^2 - x^2-y^2
    # c^2-a^2 = -sx+s^2/4 -2yh_triangle+h_triangle^2
    # 2yh_triangle = -sx+s^2/4+h_triangle^2 - (c^2-a^2)
    y_local = (s**2 / 4 + h_triangle**2 - (c**2 - a**2) - s * x_local) / (2 * h_triangle)
    
    # z_local: from a^2 = x_local^2 + y_local^2 + z_local^2
    z_local_squared = a**2 - x_local**2 - y_local**2
    
    # Use a small tolerance for floating point comparisons to zero
    if z_local_squared < -1e-9: 
        raise ValueError(f"No real solution for z: z_squared ({z_local_squared:.4f}) is negative. Apex is not reachable with given distances.")
    # Ensure z_local_squared is not negative due to precision errors before sqrt
    z_local = np.sqrt(max(0, z_local_squared))

    # Transform (x_local, y_local, z_local) to the global coordinate system.
    # Global vertex positions:
    p_r_global = np.array(RED_POSITION, dtype=float)    # Corresponds to local origin
    p_g_global = np.array(GREEN_POSITION, dtype=float)  # Corresponds to local (s,0,0)
    p_b_global = np.array(BLUE_POSITION, dtype=float)   # Corresponds to local (s/2, h_triangle,0)

    # Define the basis vectors of the local system in terms of global coordinates.
    # ex_prime: local x-axis (vector from RED to GREEN, normalized by s)
    vec_rg_global = p_g_global - p_r_global
    # Assuming np.linalg.norm(vec_rg_global) is very close to s
    ex_prime = vec_rg_global / s 

    # ey_prime: local y-axis. This vector is in the plane of the triangle (P_R, P_G, P_B),
    # orthogonal to ex_prime, and oriented such that P_B is at (s/2, h_triangle)
    # in the local (ex_prime, ey_prime) 2D system relative to P_R.
    # vec_rb_global = p_b_global - p_r_global
    # The component of vec_rb_global along ex_prime is (s/2) * ex_prime.
    # The component of vec_rb_global along ey_prime is h_triangle * ey_prime.
    # So, h_triangle * ey_prime = vec_rb_global - (s/2) * ex_prime
    
    vec_rb_global_projection_on_y_local = (p_b_global - p_r_global) - (s/2) * ex_prime
    # Assuming np.linalg.norm(vec_rb_global_projection_on_y_local) is very close to h_triangle
    ey_prime = vec_rb_global_projection_on_y_local / h_triangle

    # ez_prime: local z-axis (orthogonal to the triangle plane)
    ez_prime = np.cross(ex_prime, ey_prime)
    # Normalize ez_prime to ensure it's a unit vector, in case ex_prime and ey_prime are not perfectly orthogonal
    # or unit due to floating point arithmetic or slight inconsistencies in global positions.
    norm_ez_prime = np.linalg.norm(ez_prime)
    if norm_ez_prime < 1e-9: # Should not happen if P_R, P_G, P_B are not collinear
        raise ValueError("Cannot form a valid z-axis; triangle vertices may be collinear.")
    ez_prime = ez_prime / norm_ez_prime
    
    # Global coordinates of the apex: P_R + x_local*ex' + y_local*ey' + z_local*ez'
    apex_global = p_r_global + x_local * ex_prime + y_local * ey_prime + z_local * ez_prime

    return tuple(apex_global)


# name says it all
def apply_moving_average(data, window_size):
    """
    Applies a moving average filter to a list of numbers.
    Handles np.nan values by ignoring them in the window calculation.
    """
    if not data:
        return []

    # Ensure window_size is odd for a centered window
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    smoothed_data = []

    for i in range(len(data)):
        window_start = max(0, i - half_window)
        window_end = min(len(data), i + half_window + 1)

        current_window = data[window_start:window_end]

        # Filter out NaNs
        valid_values_in_window = [x for x in current_window if not np.isnan(x)]

        if not valid_values_in_window:
            smoothed_data.append(np.nan)
        else:
            smoothed_data.append(np.mean(valid_values_in_window))

    return smoothed_data

# smooths the trajectory
def smooth_trajectory_data(coords_x, coords_y, coords_z, method='moving_average', window_size=5):
    """
    Applies a specified smoothing method to the trajectory data.
    """
    if not coords_x or not coords_y or not coords_z:
        print("Coordinate data is empty, skipping smoothing.")
        return coords_x, coords_y, coords_z

    if method == 'moving_average':
        if len(coords_x) > window_size:  # Apply only if enough data points
            print(f"Applying moving average filter with window size {window_size}...")
            smoothed_x = apply_moving_average(coords_x, window_size)
            smoothed_y = apply_moving_average(coords_y, window_size)
            smoothed_z = apply_moving_average(coords_z, window_size)
            # Optionally, you might want to smooth other related data here too
            # e.g., green_blue_angles if passed in and handled
            return smoothed_x, smoothed_y, smoothed_z
        else:
            print("Not enough data points to apply moving average filter, or filter disabled.")
            return coords_x, coords_y, coords_z
    elif method == 'none' or method is None:
        print("No smoothing method applied.")
        return coords_x, coords_y, coords_z
    else:
        print(f"Unknown smoothing method: {method}. Returning original data.")
        return coords_x, coords_y, coords_z

def slam_like_loop_closure(coords_x, coords_z):
    # SLAM-like loop closure for X-Z coordinates
    if len(coords_x) > 1 and len(coords_z) > 1:
        print("Applying SLAM-like loop closure to X-Z path...")

        x_start, z_start = coords_x[0], coords_z[0]
        x_end, z_end = coords_x[-1], coords_z[-1]

        delta_x = x_start - x_end
        delta_z = z_start - z_end

        num_points = len(coords_x)

        # Create copies to modify, ensuring they are lists of floats
        adjusted_coords_x = [float(val) for val in coords_x]
        adjusted_coords_z = [float(val) for val in coords_z]

        for i in range(num_points):
            adjustment_factor = i / (num_points - 1) if num_points > 1 else 0.0
            adjusted_coords_x[i] += delta_x * adjustment_factor
            adjusted_coords_z[i] += delta_z * adjustment_factor

        # Update coords_x and coords_z with adjusted values for plotting and saving
        coords_x = adjusted_coords_x
        coords_z = adjusted_coords_z
        print(
            f"Loop closure applied. Original end: ({x_end:.2f}, {z_end:.2f}), New end: ({coords_x[-1]:.2f}, {coords_z[-1]:.2f})")
    elif coords_x and coords_z:  # Check if lists are not empty
        print("Not enough points (or only one point) to apply loop closure.")
    else:
        print("Path is empty, skipping loop closure.")

    return coords_x, coords_z


def calculate_camera_positions_from_rgb_distances(ball_sizes):
    # major axes are the sizes (for now)
    original_ball_sizes_red = ball_sizes[0][0][2]
    original_ball_sizes_green = ball_sizes[0][1][2]
    original_ball_sizes_blue = ball_sizes[0][2][2]

    coords_x, coords_y, coords_z = [], [], []
    prev_pos = INITIAL_CAMERA_POSITION  # Use previous position as initial guess for next frame

    for frame_idx, frame_ball_data in enumerate(ball_sizes):
        red_data, green_data, blue_data = frame_ball_data[0], frame_ball_data[1], frame_ball_data[2]

        _, _, red_size, _, _= red_data
        _, _, green_size, _, _ = green_data
        _, _, blue_size, _, _ = blue_data

        try:
            approx_distances = []
            current_sizes = [red_size, green_size, blue_size]
            original_s = [original_ball_sizes_red, original_ball_sizes_green, original_ball_sizes_blue]

            for i in range(3):
                size = current_sizes[i]
                orig_s = original_s[i]
                dist = 18 * (orig_s / size)
                approx_distances.append(dist)

            # the optimization and analytical approach have similar results
            # now choosing optimization to be able to add extra parameters
            camera_pos = estimate_camera_position(
                *approx_distances,
                initial_guess=prev_pos
            )

            # camera_pos = apex_coordinates(*approx_distances)

            prev_pos = camera_pos

            x, y, z = camera_pos

            coords_x.append(x)
            coords_y.append(y)
            coords_z.append(z)
            # Angle is handled at the start of the loop for this frame

        except Exception as e:
            print(f"Error processing frame {frame_idx} for 3D estimation: {e}")
            x, y, z = prev_pos
            coords_x.append(x)
            coords_y.append(y)
            coords_z.append(z)
    return coords_x, coords_y, coords_z


def calculate_triangle(ball_sizes):
    original_green_x = ball_sizes[0][1][0]
    original_blue_x = ball_sizes[0][2][0]
    original_dist_green_blue = abs(original_green_x - original_blue_x)

    green_blue_angles = []  # New list to store angles
    green_blue_distances = []
    triangle_center_x, triangle_center_y = [], []

    for frame_idx, frame_ball_data in enumerate(ball_sizes):
        red_data, green_data, blue_data = frame_ball_data[0], frame_ball_data[1], frame_ball_data[2]

        # Calculate Green-Blue angle
        green_x, green_y, _, _, _ = green_data
        blue_x, blue_y, _, _, _ = blue_data
        red_x, red_y, _, _, _ = red_data

        angle_rad = math.atan2(blue_y - green_y, blue_x - green_x)
        green_blue_angles.append(math.degrees(angle_rad))

        new_distance = abs(green_x - blue_x)
        green_blue_distances.append(new_distance / original_dist_green_blue * TRIANGLE_SIZE)

        # Calculate average x and y from red, green, and blue
        avg_x = (red_x + green_x + blue_x) / 3
        avg_y = (red_y + green_y + blue_y) / 3

        triangle_center_x.append(avg_x)
        triangle_center_y.append(avg_y)

    return green_blue_angles, green_blue_distances, triangle_center_x, triangle_center_y


# bit overcomplicated, but could be useful later on
def calculate_slope_of_horizontal_line_in_image(
    p1_world: np.ndarray,
    p2_world: np.ndarray,
    camera_position_world: np.ndarray,
    camera_yaw_degrees: float,
    camera_pitch_degrees: float,
    camera_roll_degrees: float,
    intrinsics: dict
) -> float:
    """
    Calculates the apparent slope of a 3D horizontal line when projected into a camera image.

    A horizontal line is defined as a line where the Y-coordinate is constant in world space.
    The camera coordinate system is assumed to be X right, Y down, Z forward.
    The world coordinate system is assumed to have Y as the 'up' axis.

    Parameters:
    - p1_world (np.ndarray): (3,) XYZ coordinates of the first point of the line in world space.
    - p2_world (np.ndarray): (3,) XYZ coordinates of the second point of the line in world space.
    - camera_position_world (np.ndarray): (3,) XYZ position of the camera in world space.
    - camera_yaw_degrees (float): Camera yaw. Rotation around the camera's initial Y-axis (degrees).
    - camera_pitch_degrees (float): Camera pitch. Rotation around the camera's new X-axis (degrees).
    - camera_roll_degrees (float): Camera roll. Rotation around the camera's newest Z-axis (degrees).
                                   This uses an intrinsic 'yxz' Euler sequence.
    - intrinsics (dict): Camera intrinsic parameters {'fx': float, 'fy': float, 'cx': float, 'cy': float}.

    Returns:
    - float: The slope (dy/dx) of the line in the 2D image plane.
             Returns np.inf for vertical lines, np.nan if projection fails (e.g. points behind camera).
    """
    from scipy.spatial.transform import Rotation

    # Validate that the line is horizontal (constant Y world coordinate)
    if not np.isclose(p1_world[1], p2_world[1]):
        raise ValueError("The provided 3D line is not horizontal (Y-coordinates of points differ in world space).")

    # Create rotation matrix representing camera orientation in the world (R_cw)
    # Euler sequence 'yxz':
    # 1. Yaw around initial Y axis
    # 2. Pitch around new X axis
    # 3. Roll around newest Z axis
    rot = Rotation.from_euler('yxz', [camera_yaw_degrees, camera_pitch_degrees, camera_roll_degrees], degrees=True)
    R_cw = rot.as_matrix()  # Orientation of camera frame in world frame

    # Rotation matrix from world to camera frame
    R_wc = R_cw.T

    # Transform points from world to camera coordinates
    p1_cam = R_wc @ (p1_world - camera_position_world)
    p2_cam = R_wc @ (p2_world - camera_position_world)

    # Check if points are at or behind the camera's image plane (z_cam <= epsilon)
    # Using a small epsilon to avoid division by zero and issues with points exactly on the plane
    epsilon = 1e-6
    if p1_cam[2] <= epsilon or p2_cam[2] <= epsilon:
        return np.nan  # Points are not in front of the camera

    # Project points to image plane
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    u1 = fx * p1_cam[0] / p1_cam[2] + cx
    v1 = fy * p1_cam[1] / p1_cam[2] + cy

    u2 = fx * p2_cam[0] / p2_cam[2] + cx
    v2 = fy * p2_cam[1] / p2_cam[2] + cy

    # Calculate slope in image plane (delta_v / delta_u)
    delta_u = u2 - u1
    delta_v = v2 - v1

    if np.isclose(delta_u, 0.0):
        if np.isclose(delta_v, 0.0):
            # Points project to the same location.
            # Slope could be considered 0 or undefined. Let's return 0 for simplicity.
            return 0.0
        return np.inf  # Vertical line in image
    
    slope = delta_v / delta_u
    return slope

