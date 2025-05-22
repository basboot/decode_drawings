import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# --- Constants ---
BALL_RADIUS = 3
PROJECTION_Z = 18
PROJECTION_POS = [0, 0, PROJECTION_Z] # Projection plane position (assuming centered at xy=0)
PROJECTION_WIDTH = 7.2
PROJECTION_HEIGHT = 7.2 / 16 * 9
FOCUS_POS = [0, 0, 20] # Focus point

# --- Ball Definitions ---
balls_data = [
    {'position': [0, math.sqrt(3)*(9/4), 0], 'color': 'red', 'label': 'Red Ball'},
    {'position': [4.5, -math.sqrt(3)*(9/4), 0], 'color': 'green', 'label': 'Green Ball'},
    {'position': [-4.5, -math.sqrt(3)*(9/4), 0], 'color': 'blue', 'label': 'Blue Ball'}
]

# Extract initial positions into a NumPy array
initial_positions = np.array([ball['position'] for ball in balls_data])

# --- Transformation Parameters ---
# Define the sequence of transformations to be applied
Y_ROTATION_ANGLE_DEG = 0  # Degrees for rotation around Y-axis
Z_ROTATION_ANGLE_DEG = 0  # Degrees for rotation around Z-axis
Z_TRANSLATION_UNITS = 0   # Units for translation along Z-axis

# --- Helper Functions ---
def calculate_projection_parameters(
    initial_ball_pos,  # Expects a (3,) numpy array for a single ball's initial position
    y_rotation_deg,
    z_rotation_deg,
    z_translation_val,
    focus_pos,
    projection_z,
    ball_radius
):
    """Calculates all parameters needed for the projected ellipse after applying transformations."""

    # 1. Apply Y-axis rotation to initial_ball_pos
    y_angle_rad = np.radians(y_rotation_deg)
    rotation_matrix_y = np.array([
        [np.cos(y_angle_rad), 0, np.sin(y_angle_rad)],
        [0, 1, 0],
        [-np.sin(y_angle_rad), 0, np.cos(y_angle_rad)]
    ])
    current_pos = np.dot(initial_ball_pos, rotation_matrix_y.T)

    # 2. Apply Z-axis rotation to the result of Y-axis rotation
    z_angle_rad = np.radians(z_rotation_deg)
    rotation_matrix_z = np.array([
        [np.cos(z_angle_rad), -np.sin(z_angle_rad), 0],
        [np.sin(z_angle_rad), np.cos(z_angle_rad), 0],
        [0, 0, 1]
    ])
    current_pos = np.dot(current_pos, rotation_matrix_z.T)

    # 3. Apply Z-axis translation to the result of Z-axis rotation
    translation_vector_z = np.array([0, 0, z_translation_val])
    final_ball_pos = current_pos + translation_vector_z
    
    # --- Original projection logic starts here, using final_ball_pos ---
    delta_z_focus_projection = focus_pos[2] - projection_z
    delta_z_focus_ball = focus_pos[2] - final_ball_pos[2]

    if delta_z_focus_ball == 0:
        print(f"Warning: Ball at z={final_ball_pos[2]} (after transform) is at the same z-level as focus z={focus_pos[2]}. Skipping.")
        return None

    projected_center_x = final_ball_pos[0] * delta_z_focus_projection / delta_z_focus_ball
    projected_center_y = final_ball_pos[1] * delta_z_focus_projection / delta_z_focus_ball

    # 3D distance from focus to ball center (using final_ball_pos)
    dist_focus_ball = np.sqrt(
        (final_ball_pos[0] - focus_pos[0])**2 +
        (final_ball_pos[1] - focus_pos[1])**2 +
        (final_ball_pos[2] - focus_pos[2])**2
    )

    # Apparent radius if viewed head-on
    semi_axis_perp_los = ball_radius * abs(delta_z_focus_projection) / abs(delta_z_focus_ball)

    # Semi-axis along LOS (foreshortened)
    if dist_focus_ball == 0: # Ball is exactly at the focus point
        print(f"Warning: Ball {final_ball_pos} (after transform) is at the focus point. Projection is undefined.")
        semi_axis_along_los = float('inf') # Or handle as an error
    else:
        semi_axis_along_los = semi_axis_perp_los * abs(delta_z_focus_ball) / dist_focus_ball
        
    # Ellipse angle on the projection plane
    angle_rad_ellipse = np.arctan2(projected_center_y, projected_center_x)
    angle_degrees_ellipse = np.degrees(angle_rad_ellipse)
    
    return {
        'center_x': projected_center_x,
        'center_y': projected_center_y,
        'semi_axis_along_los': semi_axis_along_los,
        'semi_axis_perp_los': semi_axis_perp_los,
        'angle_degrees': angle_degrees_ellipse, # This is the 2D ellipse orientation
        'final_ball_pos_3d': final_ball_pos  # The 3D position after all transformations
    }

DEBUG = False

if DEBUG:
# --- Main Plotting Logic ---
    fig, ax = plt.subplots(figsize=(10, 10)) # Adjust figure size if needed

    for i, ball_info in enumerate(balls_data):
        # Use the initial position of the ball
        original_ball_pos = initial_positions[i]
        color = ball_info['color']
        label = ball_info['label']
        
        params = calculate_projection_parameters(
            original_ball_pos,
            Y_ROTATION_ANGLE_DEG,
            Z_ROTATION_ANGLE_DEG,
            Z_TRANSLATION_UNITS,
            FOCUS_POS,
            PROJECTION_POS[2], # This is projection_z
            BALL_RADIUS
        )
        
        if params:
            # Use the final 3D position returned by the function for printing
            final_pos_3d_for_print = params['final_ball_pos_3d']
            # Updated print statement to show initial and final positions
            print(f"Processing {label}: Initial Pos: {original_ball_pos.round(2)}, Final Transformed Pos: {final_pos_3d_for_print.round(2)}")
            print(f"  Projected Center: ({params['center_x']:.2f}, {params['center_y']:.2f})")
            print(f"  Semi-axes (along/perp LOS): ({params['semi_axis_along_los']:.2f}, {params['semi_axis_perp_los']:.2f})")
            print(f"  Angle: {(params['angle_degrees'] - 90) % 180:.2f}°")

            ellipse = Ellipse(
                (params['center_x'], params['center_y']),
                width=2 * params['semi_axis_along_los'],
                height=2 * params['semi_axis_perp_los'],
                angle=params['angle_degrees'],
                edgecolor=color,
                facecolor='none',
                label=f'{label}\nAngle: {params["angle_degrees"]:.1f}°'
            )
            ax.add_artist(ellipse)

            # Draw Major and Minor Axes
            cx = params['center_x']
            cy = params['center_y']
            angle_deg = params['angle_degrees']
            semi_along_los = params['semi_axis_along_los']
            semi_perp_los = params['semi_axis_perp_los']

            # Determine major and minor axes lengths and their angles
            if semi_along_los >= semi_perp_los:
                major_semi_axis_len = semi_along_los
                minor_semi_axis_len = semi_perp_los
                major_axis_angle_deg = angle_deg
                minor_axis_angle_deg = angle_deg + 90
            else:
                major_semi_axis_len = semi_perp_los
                minor_semi_axis_len = semi_along_los
                major_axis_angle_deg = angle_deg + 90
                minor_axis_angle_deg = angle_deg

            # Convert angles to radians for trigonometric functions
            major_axis_angle_rad = np.radians(major_axis_angle_deg)
            minor_axis_angle_rad = np.radians(minor_axis_angle_deg)

            # Major Axis (blue)
            major_dx = major_semi_axis_len * np.cos(major_axis_angle_rad)
            major_dy = major_semi_axis_len * np.sin(major_axis_angle_rad)
            ax.plot([cx - major_dx, cx + major_dx], [cy - major_dy, cy + major_dy], color='blue', linestyle='--', linewidth=0.75)

            # Minor Axis (red)
            minor_dx = minor_semi_axis_len * np.cos(minor_axis_angle_rad)
            minor_dy = minor_semi_axis_len * np.sin(minor_axis_angle_rad)
            ax.plot([cx - minor_dx, cx + minor_dx], [cy - minor_dy, cy + minor_dy], color='red', linestyle='--', linewidth=0.75)

    # --- Plot Setup ---
    ax.set_xlim(-PROJECTION_WIDTH / 2, PROJECTION_WIDTH / 2)
    ax.set_ylim(-PROJECTION_HEIGHT / 2, PROJECTION_HEIGHT / 2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X on Projection Plane")
    ax.set_ylabel("Y on Projection Plane")
    ax.set_title("Perspective Projection of Spheres")
    ax.legend(fontsize='small', loc='upper right')
    ax.grid(True)

    plt.show()

else:
    data = []
    for rotation_y in np.arange(-80, 80, 0.5):
        for translation_z in np.arange(-50, 10, 0.1):
            angles = []
            for i, ball_info in enumerate(balls_data):
                # Use the initial position of the ball
                original_ball_pos = initial_positions[i]
                color = ball_info['color']
                label = ball_info['label']
                
                params = calculate_projection_parameters(
                    original_ball_pos,
                    rotation_y,
                    0,
                    translation_z,
                    FOCUS_POS,
                    PROJECTION_POS[2], # This is projection_z
                    BALL_RADIUS
                )
                angles.append((params['angle_degrees'] - 90) % 180)

            # TODO: caclculate rotation for PROJECTION_Z
            data.append(angles)

