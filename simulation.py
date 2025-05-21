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

# --- Helper Functions ---
def calculate_projection_parameters(ball_pos, focus_pos, projection_z, ball_radius):
    """Calculates all parameters needed for the projected ellipse."""
    
    # Projected center
    # Simplified formula assuming focus_pos[0] and focus_pos[1] are 0
    # P_c = S_c * (F_z - P_z) / (F_z - S_z)
    # More general: Pc_x = F_x + (S_x - F_x) * (P_z - F_z) / (S_z - F_z)
    # Using the simpler form as focus is at [0,0,20]
    
    delta_z_focus_projection = focus_pos[2] - projection_z
    delta_z_focus_ball = focus_pos[2] - ball_pos[2]

    if delta_z_focus_ball == 0: # Avoid division by zero if ball is at focus's z-plane
        # This case would mean infinite projection or ball behind focus, handle as error or specific logic
        print(f"Warning: Ball at z={ball_pos[2]} is at the same z-level as focus z={focus_pos[2]}. Skipping.")
        return None

    projected_center_x = ball_pos[0] * delta_z_focus_projection / delta_z_focus_ball
    projected_center_y = ball_pos[1] * delta_z_focus_projection / delta_z_focus_ball

    # 3D distance from focus to ball center
    dist_focus_ball = np.sqrt(
        (ball_pos[0] - focus_pos[0])**2 +
        (ball_pos[1] - focus_pos[1])**2 +
        (ball_pos[2] - focus_pos[2])**2
    )

    # Apparent radius if viewed head-on (this is semi_axis_perp_los)
    semi_axis_perp_los = ball_radius * abs(delta_z_focus_projection) / abs(delta_z_focus_ball)

    # Semi-axis along LOS (foreshortened)
    if dist_focus_ball == 0: # Ball is exactly at the focus point
        print(f"Warning: Ball {ball_pos} is at the focus point. Projection is undefined.")
        semi_axis_along_los = float('inf') # Or handle as an error
    else:
        semi_axis_along_los = semi_axis_perp_los * abs(delta_z_focus_ball) / dist_focus_ball
        
    # Ellipse angle
    # Angle of the vector from projected focus (0,0) to projected ball center
    angle_rad = np.arctan2(projected_center_y, projected_center_x)
    angle_degrees = np.degrees(angle_rad)
    
    return {
        'center_x': projected_center_x,
        'center_y': projected_center_y,
        'semi_axis_along_los': semi_axis_along_los,
        'semi_axis_perp_los': semi_axis_perp_los,
        'angle_degrees': angle_degrees
    }

# --- Main Plotting Logic ---
fig, ax = plt.subplots(figsize=(PROJECTION_WIDTH / 2, PROJECTION_HEIGHT / 2)) # Adjust figure size if needed

for ball_info in balls_data:
    ball_pos = ball_info['position']
    color = ball_info['color']
    label = ball_info['label']
    
    params = calculate_projection_parameters(ball_pos, FOCUS_POS, PROJECTION_POS[2], BALL_RADIUS)
    
    if params:
        print(f"Processing {label}: Ball Pos: {ball_pos}")
        print(f"  Projected Center: ({params['center_x']:.2f}, {params['center_y']:.2f})")
        print(f"  Semi-axes (along/perp LOS): ({params['semi_axis_along_los']:.2f}, {params['semi_axis_perp_los']:.2f})")
        print(f"  Angle: {params['angle_degrees']:.2f}°")

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