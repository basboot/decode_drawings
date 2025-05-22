# https://radufromfinland.com/decodeTheDrawings/
import cv2
import numpy as np
import math
from scipy.optimize import minimize
import json  # Added for saving data
import os    # Added for checking file existence

VIDEO = "1"

RED, GREEN, BLUE = "RED", "GREEN", "BLUE"
TRIANGLE_CENTER = np.array([4.5, 18, 0])  # Point camera is looking at (height = 18)

def estimate_undistorted_radius(major, minor):
    a = major / 2
    b = minor / 2
    r_undistorted = math.sqrt(a * b)  # geometric mean
    return r_undistorted

def calculate_camera_error(pos, distances, ball_positions):
    """
    Calculate error between expected distances and actual distances,
    taking into account that camera is looking at triangle center.
    """
    pos = np.array(pos)
    error = 0
    
    # Vector from camera to center (viewing direction)
    to_center = TRIANGLE_CENTER - pos
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

def estimate_camera_position(red_dist, green_dist, blue_dist, initial_guess=None):
    """
    Estimate camera position using optimization.
    Takes into account that camera is always looking at triangle center.
    """
    # Ball positions in world space (all at height 18)
    # Equilateral triangle with center at (4.5, 18, 0), side length 9
    # Height of triangle: h = 9 * sqrt(3) / 2 ≈ 7.794
    h = 9 * np.sqrt(3) / 2
    ball_positions = [
        np.array([4.5, 18 + (2 / 3) * h, 0]),   # Red (top vertex)
        np.array([9, 18 - h / 3, 0]),     # Green (bottom right)
        np.array([0, 18 - h / 3, 0])      # Blue (bottom left)
    ]
    
    distances = [red_dist, green_dist, blue_dist]
    
    if initial_guess is None:
        # Make initial guess based on average distance
        avg_dist = sum(distances) / 3
        initial_guess = [4.5, 18, avg_dist]
    
    # Use optimization to find best camera position
    result = minimize(
        calculate_camera_error,
        initial_guess,
        args=(distances, ball_positions),
        method='Nelder-Mead'
    )
    
    return result.x

def apex_coordinates(a, b, c, side_length=9):
    """
    Compute the 3D coordinates of the apex of a pyramid over an equilateral triangle base,
    given distances from the apex to each corner.

    Parameters:
    a, b, c : float
        Distances from apex to vertices A, B, and C respectively.
    side_length : float
        Length of the side of the equilateral triangle base.

    Returns:
    (x, y, z) : tuple of floats
        Coordinates of the apex point.
    """
    # Coordinates of triangle vertices
    A = np.array([side_length / 2, (side_length * np.sqrt(3)) / 2, 0.0]) # red
    B = np.array([side_length, 0.0, 0.0]) # green
    C = np.array([0.0, 0.0, 0.0]) # blue

    # From distance equations:
    # |P - A|^2 = a^2 = x^2 + y^2 + z^2
    # |P - B|^2 = b^2 = (x - s)^2 + y^2 + z^2
    # |P - C|^2 = c^2 = (x - s/2)^2 + (y - h)^2 + z^2
    # where h = s * sqrt(3)/2

    s = side_length
    h = (s * np.sqrt(3)) / 2

    # Use differences to eliminate z^2
    # Equation 1: x^2 + y^2 + z^2 = a^2
    # Equation 2: (x - s)^2 + y^2 + z^2 = b^2
    # Equation 3: (x - s/2)^2 + (y - h)^2 + z^2 = c^2

    # Subtract eq1 from eq2:
    # (x - s)^2 - x^2 = b^2 - a^2
    # => x^2 - 2sx + s^2 - x^2 = b^2 - a^2 => -2sx + s^2 = b^2 - a^2
    # => x = (s^2 - (b^2 - a^2)) / (2s)

    x = (s ** 2 - (b ** 2 - a ** 2)) / (2 * s)

    # Subtract eq1 from eq3:
    # (x - s/2)^2 + (y - h)^2 - x^2 - y^2 = c^2 - a^2
    # Expand and simplify:
    # (x^2 - sx + s^2/4) + (y^2 - 2yh + h^2) - x^2 - y^2 = c^2 - a^2
    # => -sx + s^2/4 - 2yh + h^2 = c^2 - a^2
    # => y = (s^2/4 + h^2 - (c^2 - a^2) - sx) / (2h)

    y = (s ** 2 / 4 + h ** 2 - (c ** 2 - a ** 2) - s * x) / (2 * h)

    # Now plug x and y into Equation 1 to get z
    z_squared = a ** 2 - x ** 2 - y ** 2
    if z_squared < 0:
        raise ValueError("No real solution: apex is not reachable with given distances.")
    z = np.sqrt(z_squared)

    return (x, y, z)

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

def smooth_trajectory_data(coords_x, coords_y, coords_z, method='moving_average', window_size=5):
    """
    Applies a specified smoothing method to the trajectory data.
    """
    if not coords_x or not coords_y or not coords_z:
        print("Coordinate data is empty, skipping smoothing.")
        return coords_x, coords_y, coords_z

    if method == 'moving_average':
        if len(coords_x) > window_size: # Apply only if enough data points
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

ball_sizes = []
count = 0 # This count is for video processing; will be 0 if loading from file initially.

if __name__ == '__main__':
    ball_data_filepath = f"ball_sizes_data{VIDEO}.json"
    original_ball_sizes = {} # Initialize

    data_loaded_successfully = False
    if os.path.exists(ball_data_filepath):
        print(f"Found data file: {ball_data_filepath}. Attempting to load...")
        try:
            with open(ball_data_filepath, "r") as f:
                loaded_data = json.load(f)
            original_ball_sizes = loaded_data['original_ball_sizes']
            ball_sizes = loaded_data['ball_sizes']
            print("Ball data loaded successfully from file.")
            data_loaded_successfully = True
        except Exception as e:
            print(f"Error loading data from {ball_data_filepath}: {e}. Will process video instead.")
            original_ball_sizes = {} # Reset in case of partial load or error
            ball_sizes = []

    if not data_loaded_successfully:
        print("Processing video to gather ball data...")
        cap = cv2.VideoCapture(f"videos/{VIDEO}.mp4")

        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit() # Exit if video cannot be opened and no data loaded
        else:
            print("Video file opened successfully!")
        
        count = 0 
        original_ball_sizes = {} # Ensure it's empty before video processing
        ball_sizes = []          # Ensure it's empty

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            lower_green = np.array([40, 70, 70])
            upper_green = np.array([80, 255, 255])
            lower_blue = np.array([100, 150, 0])
            upper_blue = np.array([140, 255, 255])

            mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

            current_frame_data = [] # Stores [R_data, G_data, B_data] for this frame
            output = frame.copy()

            for color_mask, color_name in [(mask_red, RED), (mask_green, GREEN), (mask_blue, BLUE)]:
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                ball_found_for_color = False
                for cnt in contours:
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        cv2.ellipse(output, ellipse, (0, 255, 0), 2)
                        (x_ellipse, y_ellipse), (major, minor), angle_ellipse = ellipse
                        
                        estimated_size = major # Using major axis as size
                        ball_data = [estimated_size, x_ellipse, y_ellipse, angle_ellipse]
                        current_frame_data.append(ball_data)
                        
                        if count == 0: # First frame of video processing
                            original_ball_sizes[color_name] = estimated_size # Store only size for original
                        
                        ball_found_for_color = True
                        break # Take first good contour for this color
                
                if not ball_found_for_color:
                    current_frame_data.append(None) # Append None if ball of this color not found
                    if count == 0 and color_name not in original_ball_sizes:
                         original_ball_sizes[color_name] = None # Mark as not seen in first frame
            
            if len(current_frame_data) == 3:
                ball_sizes.append(current_frame_data)
            else:
                while len(current_frame_data) < 3:
                    current_frame_data.append(None)
                ball_sizes.append(current_frame_data)

            count += 1

        cap.release()

        print("Video processing complete.")
        print("Original ball sizes (from video processing):", original_ball_sizes)
        print("Number of frames processed:", len(ball_sizes))

        data_to_save = {
            'original_ball_sizes': original_ball_sizes,
            'ball_sizes': ball_sizes
        }
        try:
            with open(ball_data_filepath, "w") as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Successfully saved ball data to {ball_data_filepath}")
        except Exception as e:
            print(f"Error saving ball data: {e}")

    if not ball_sizes or not original_ball_sizes:
        print("Error: Ball data is not available after attempting load/processing. Exiting.")
        exit()

    initial_pos = np.array([4.5, 18, 18])  # Same height as triangle center
    
    coords_x, coords_y, coords_z = [], [], []
    green_blue_angles = [] # New list to store angles
    prev_pos = initial_pos  # Use previous position as initial guess for next frame
    
    for frame_idx, frame_ball_data in enumerate(ball_sizes):
        if frame_ball_data is None or len(frame_ball_data) != 3:
            print(f"Skipping frame {frame_idx} due to malformed data: {frame_ball_data}")
            if prev_pos is not None:
                coords_x.append(prev_pos[0])
                coords_y.append(prev_pos[1])
                coords_z.append(prev_pos[2])
            else:
                coords_x.append(np.nan)
                coords_y.append(np.nan)
                coords_z.append(np.nan)
            green_blue_angles.append(np.nan) # Add NaN for angle as well
            continue

        red_data, green_data, blue_data = frame_ball_data[0], frame_ball_data[1], frame_ball_data[2]

        # Calculate Green-Blue angle before checking if all balls are present for 3D estimation
        if green_data and blue_data:
            _, green_x, green_y, _ = green_data
            _, blue_x, blue_y, _ = blue_data
            angle_rad = math.atan2(blue_y - green_y, blue_x - green_x)
            green_blue_angles.append(math.degrees(angle_rad))
        else:
            green_blue_angles.append(np.nan)

        # Check if all balls were found for 3D position estimation
        if not (red_data and green_data and blue_data):
            print(f"Skipping frame {frame_idx} for 3D estimation due to missing ball data.")
            if prev_pos is not None:
                coords_x.append(prev_pos[0])
                coords_y.append(prev_pos[1])
                coords_z.append(prev_pos[2])
            else:
                coords_x.append(np.nan)
                coords_y.append(np.nan)
                coords_z.append(np.nan)
            # Angle was already handled above, so just continue
            continue

        red_size, _, _, _ = red_data
        green_size, _, _, _ = green_data
        blue_size, _, _, _ = blue_data
        
        try:
            approx_distances = []
            current_sizes = [red_size, green_size, blue_size]
            original_s = [original_ball_sizes.get(RED), original_ball_sizes.get(GREEN), original_ball_sizes.get(BLUE)]

            valid_data_for_dist_calc = True
            for i in range(3):
                size = current_sizes[i]
                orig_s = original_s[i]
                if size is None or orig_s is None or size == 0:
                    print(f"Frame {frame_idx}: Missing size or original size for dist calc. Ball {i}")
                    valid_data_for_dist_calc = False
                    break
                dist = 18 * (orig_s / size)
                approx_distances.append(dist)
            
            if not valid_data_for_dist_calc or len(approx_distances) != 3:
                print(f"Frame {frame_idx}: Not enough valid distances. Using previous position.")
                if prev_pos is not None:
                    coords_x.append(prev_pos[0]); coords_y.append(prev_pos[1]); coords_z.append(prev_pos[2])
                else:
                    coords_x.append(np.nan); coords_y.append(np.nan); coords_z.append(np.nan)
                # Angle is already handled if a ball required for it was missing, or if dist calc failed.
                # If dist calc failed but G-B was present, angle is already in the list.
                # No need to add another NaN for angle here specifically for dist calc failure.
                continue

            camera_pos = estimate_camera_position(
                *approx_distances,
                initial_guess=prev_pos
            )
            
            prev_pos = camera_pos
            
            x, y, z = camera_pos
            
            coords_x.append(x)
            coords_y.append(y)
            coords_z.append(z)
            # Angle is handled at the start of the loop for this frame
            
        except Exception as e:
            print(f"Error processing frame {frame_idx} for 3D estimation: {e}")
            if prev_pos is not None:
                x, y, z = prev_pos
                coords_x.append(x)
                coords_y.append(y)
                coords_z.append(z)
            else:
                coords_x.append(np.nan)
                coords_y.append(np.nan)
                coords_z.append(np.nan)
            # Angle was already added (or NaN) at the start of the loop iteration

    coords_x, coords_y, coords_z = smooth_trajectory_data(coords_x, coords_y, coords_z, method='moving_average', window_size=5)

    # SLAM-like loop closure for X-Z coordinates
    if len(coords_x) > 1 and len(coords_z) > 1:
        # Check if start and end points are valid numbers before attempting closure
        first_x_is_valid = not np.isnan(coords_x[0]) if coords_x else False
        last_x_is_valid = not np.isnan(coords_x[-1]) if coords_x else False
        first_z_is_valid = not np.isnan(coords_z[0]) if coords_z else False
        last_z_is_valid = not np.isnan(coords_z[-1]) if coords_z else False

        if first_x_is_valid and last_x_is_valid and first_z_is_valid and last_z_is_valid:
            print("Applying SLAM-like loop closure to X-Z path...")
            
            x_start, z_start = coords_x[0], coords_z[0]
            x_end, z_end = coords_x[-1], coords_z[-1]

            delta_x = x_start - x_end
            delta_z = z_start - z_end

            num_points = len(coords_x)
            
            # Create copies to modify, ensuring they are lists of floats
            # This also handles cases where original lists might contain mixed types if NaNs were introduced
            adjusted_coords_x = [float(val) for val in coords_x]
            adjusted_coords_z = [float(val) for val in coords_z]

            for i in range(num_points):
                # Only adjust if the current point is not NaN
                if not np.isnan(adjusted_coords_x[i]) and not np.isnan(adjusted_coords_z[i]):
                    adjustment_factor = i / (num_points - 1) if num_points > 1 else 0.0
                    adjusted_coords_x[i] += delta_x * adjustment_factor
                    adjusted_coords_z[i] += delta_z * adjustment_factor
            
            # Update coords_x and coords_z with adjusted values for plotting and saving
            coords_x = adjusted_coords_x
            coords_z = adjusted_coords_z
            print(f"Loop closure applied. Original end: ({x_end:.2f}, {z_end:.2f}), New end: ({coords_x[-1]:.2f}, {coords_z[-1]:.2f})")
        else:
            print("Skipping loop closure due to NaN or missing start/end points in the path.")
    elif coords_x and coords_z : # Check if lists are not empty
        print("Not enough points (or only one point) to apply loop closure.")
    else:
        print("Path is empty, skipping loop closure.")

    # Convert lists to numpy arrays for easier manipulation in plotting
    coords_x = np.array(coords_x)
    coords_y = np.array(coords_y)
    coords_z = np.array(coords_z)
    green_blue_angles = np.array(green_blue_angles)

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm # Added for colormap

    if len(coords_x) == 0:
        print("No data to plot.")
    else:
        plt.figure(figsize=(20, 6)) # Adjusted figure size for clarity
        
        num_frames = len(coords_x)
        color_indices = np.arange(num_frames)
        cmap = plt.cm.get_cmap('inferno') # You can choose other colormaps like 'plasma', 'inferno', 'magma', 'cividis'
        
        # Normalization factor for cmap, handles num_frames=1 case
        norm_factor = (num_frames - 1.0) if num_frames > 1 else 1.0

        # Plot top view (X-Z plane)
        ax1 = plt.subplot(1, 3, 1)
        if num_frames > 1:
            for i in range(num_frames - 1):
                if not (np.isnan(coords_x[i]) or np.isnan(coords_z[i]) or np.isnan(coords_x[i+1]) or np.isnan(coords_z[i+1])):
                    ax1.plot([coords_x[i], coords_x[i+1]], [coords_z[i], coords_z[i+1]], color=cmap(color_indices[i] / norm_factor), linestyle='-')
        valid_indices_xz = ~(np.isnan(coords_x) | np.isnan(coords_z))
        if np.any(valid_indices_xz):
             ax1.scatter(coords_x[valid_indices_xz], coords_z[valid_indices_xz], c=color_indices[valid_indices_xz], cmap=cmap, marker='o', s=25, zorder=2, linewidth=0)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_title('Top View (X-Z)')
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot camera height over time
        ax2 = plt.subplot(1, 3, 2)
        frames_indices = np.arange(num_frames)
        if num_frames > 1:
            for i in range(num_frames - 1):
                if not (np.isnan(coords_y[i]) or np.isnan(coords_y[i+1])):
                    ax2.plot([frames_indices[i], frames_indices[i+1]], [coords_y[i], coords_y[i+1]], color=cmap(color_indices[i] / norm_factor), linestyle='-')
        valid_indices_y = ~np.isnan(coords_y)
        if np.any(valid_indices_y):
            ax2.scatter(frames_indices[valid_indices_y], coords_y[valid_indices_y], c=color_indices[valid_indices_y], cmap=cmap, marker='o', s=25, zorder=2, linewidth=0)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Y (height)')
        ax2.set_title('Camera Height')
        ax2.grid(True)

        # New subplot for Green-Blue angle
        ax3 = plt.subplot(1, 3, 3)
        # Normalize green_blue_angles to be in [0, 360) range, preserving NaNs
        angles_to_plot = np.full_like(green_blue_angles, np.nan, dtype=float)
        valid_angle_mask = ~np.isnan(green_blue_angles)
        angles_to_plot[valid_angle_mask] = np.where(green_blue_angles[valid_angle_mask] < 0, green_blue_angles[valid_angle_mask] + 360, green_blue_angles[valid_angle_mask])

        if num_frames > 1:
            for i in range(num_frames - 1):
                if not (np.isnan(angles_to_plot[i]) or np.isnan(angles_to_plot[i+1])):
                    ax3.plot([frames_indices[i], frames_indices[i+1]], [angles_to_plot[i], angles_to_plot[i+1]], color=cmap(color_indices[i] / norm_factor), linestyle='-')
        valid_indices_angle = ~np.isnan(angles_to_plot)
        if np.any(valid_indices_angle):
            ax3.scatter(frames_indices[valid_indices_angle], angles_to_plot[valid_indices_angle], c=color_indices[valid_indices_angle], cmap=cmap, marker='.', s=30, zorder=2, linewidth=0)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Green-Blue Angle (degrees)')
        ax3.set_title('Angle of Green-Blue Line')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"decode{VIDEO}.pdf", format="pdf", bbox_inches="tight")

        plt.show()

    with open("apex_xz.txt", "w") as f:
        for x, z in zip(coords_x, coords_z):
            f.write(f"{x} {z}\n")

