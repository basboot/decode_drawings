import math

import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter,filtfilt
from moviepy import *


from global_settings import *


# calculate error for camera position, based in distances to ball and ball positions for optimization
def calculate_camera_error(camera_pos, distances, ball_positions, offset_x=None):
    """
    Calculate error between expected distances and actual distances,
    taking into account that camera is looking at triangle center.
    """
    camera_pos = np.array(camera_pos) # estimated using distances
    camera_centered_pos = np.array(camera_pos) # estimated using angles
    if offset_x is not None:
        # TODO:
        camera_centered_pos += offset_x # fix center if camera is not pointing at the center (ignore horiz offset for now)

    error = 0

    # Vector from centered camera to triangle center (viewing direction)
    to_center = TRIANGLE_CENTER - camera_centered_pos
    # normalized viewing direction
    to_center = to_center / np.linalg.norm(to_center)

    for ball_pos, target_dist in zip(ball_positions, distances):
        # Vector from uncentered camera to ball
        to_ball = ball_pos - camera_pos
        dist = np.linalg.norm(to_ball)

        # Vector from centered camera to ball
        to_ball_from_center = ball_pos - camera_centered_pos
        to_ball_from_center_dist = np.linalg.norm(to_ball_from_center)

        # Calculate angle between viewing direction and ball direction
        to_ball_from_center_norm = to_ball_from_center / to_ball_from_center_dist
        cos_angle = np.dot(to_ball_from_center_norm, to_center)

        # Penalize both distance error and deviation from expected viewing angle
        angle_error = 1 - cos_angle  # 0 when looking directly at ball
        dist_error = (dist - target_dist) ** 2

        error += dist_error + 5 * angle_error  # Weight angle error more

    return error


# estimate camera position, using calculate_camera_error
def estimate_camera_position(red_dist, green_dist, blue_dist, initial_guess=None, offset_x=None):
    """
    Estimate camera position using optimization.
    Takes into account that camera is always looking at triangle center.
    """

    ball_positions = [RED_POSITION, GREEN_POSITION, BLUE_POSITION]

    distances = [red_dist, green_dist, blue_dist]

    if initial_guess is None:
        # Make initial guess based on average distance
        avg_dist = sum(distances) / 3
        initial_guess = TRIANGLE_CENTER

    # Use optimization to find best camera position
    result = minimize(
        calculate_camera_error,
        initial_guess,
        args=(distances, ball_positions, offset_x),
        method='Nelder-Mead'
    )

    return result.x

def create_perpendicular_vector_xz(camera_pos):
    to_center = TRIANGLE_CENTER - camera_pos
    to_center[1] = 0  # project on x, z

    # Create a vector perpendicular to to_center in the x-z plane
    perpendicular_vector = np.array([-to_center[2], 0, to_center[0]])
    perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)

    return perpendicular_vector

def calculate_camera_positions_from_rgb_major_axis(ball_information, offset_x=None, fix_offset_afterwards=True):
    # major axes are the sizes (for now)
    original_major_axis_red = ball_information[0][0][2]
    original_major_axis_green_green = ball_information[0][1][2]
    original_major_axis_blue = ball_information[0][2][2]

    # use average size from 1st frame as reference, to account for camera not exactly in the middle
    original_ball_size = (original_major_axis_red + original_major_axis_green_green + original_major_axis_blue) / 3

    original_major_axis_red = original_ball_size
    original_major_axis_green_green = original_ball_size
    original_major_axis_blue = original_ball_size
        

    coords_x, coords_y, coords_z = [], [], []
    prev_pos = INITIAL_CAMERA_POSITION  # Use previous position as initial guess for next frame

    for frame_idx, frame_ball_data in enumerate(ball_information):
        red_data, green_data, blue_data = frame_ball_data[0], frame_ball_data[1], frame_ball_data[2]

        _, _, red_size, _, _= red_data
        _, _, green_size, _, _ = green_data
        _, _, blue_size, _, _ = blue_data


        try:
            approx_distances = []
            current_sizes = [red_size, green_size, blue_size]
            original_s = [original_major_axis_red, original_major_axis_green_green, original_major_axis_blue]

            for i in range(3):
                size = current_sizes[i]
                orig_s = original_s[i]
                dist = INITIAL_BALL_DISTANCE[i] * (orig_s / size)
                
                approx_distances.append(dist)


            # the optimization and analytical approach have similar results
            # now choosing optimization to be able to add extra parameters
            camera_pos = estimate_camera_position(
                *approx_distances,
                initial_guess=prev_pos,
                # use previous pos for offset estimation, because we don't know where we are now
                offset_x=None if fix_offset_afterwards or offset_x is None else create_perpendicular_vector_xz(prev_pos) * offset_x[frame_idx]
            )

            # camera_pos = apex_coordinates(*approx_distances)

            # try to fix position the naive way (afterwards)
            if offset_x is not None and fix_offset_afterwards:
                camera_pos += create_perpendicular_vector_xz(camera_pos) * offset_x[frame_idx]


            

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


def analyze_green_blue_ball(ball_information):
    original_green_x = ball_information[0][1][0]
    original_blue_x = ball_information[0][2][0]
    original_dist_green_blue = abs(original_green_x - original_blue_x)

    green_blue_angles = []
    green_blue_distances = []

    for frame_idx, frame_ball_data in enumerate(ball_information):
        green_data, blue_data = frame_ball_data[1], frame_ball_data[2]

        green_x, green_y, _, _, _ = green_data
        blue_x, blue_y, _, _, _ = blue_data

        # Calculate Green-Blue angle
        angle_rad = math.atan2(blue_y - green_y, blue_x - green_x)
        green_blue_angles.append(math.degrees(angle_rad))

        new_distance = abs(green_x - blue_x)
        green_blue_distances.append(new_distance / original_dist_green_blue * TRIANGLE_SIZE)

    return green_blue_angles, green_blue_distances


def calculate_horizontal_offsets(ball_information, angle_error=None):
    horizontal_offsets = []

    for frame_idx, frame_ball_data in enumerate(ball_information):
        red_data, green_data, blue_data = frame_ball_data[0], frame_ball_data[1], frame_ball_data[2]

        red_x, red_y, _, _, _ = red_data
        green_x, green_y, _, _, _ = green_data
        blue_x, blue_y, _, _, _ = blue_data

        if angle_error is not None:
            # Create rotation matrix for the given angle
            angle_rad = np.radians(-angle_error[frame_idx])
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)

            rotation_matrix = np.array([
                [cos_angle, -sin_angle],
                [sin_angle, cos_angle]
            ])

            # Center point for rotation
            center_x = VIDEO_WIDTH / 2
            center_y = VIDEO_HEIGHT / 2

            # Translate points to origin, apply rotation, and translate back
            red_coords = np.array([red_x - center_x, red_y - center_y])
            green_coords = np.array([green_x - center_x, green_y - center_y])
            blue_coords = np.array([blue_x - center_x, blue_y - center_y])

            red_rotated = rotation_matrix @ red_coords + np.array([center_x, center_y])
            green_rotated = rotation_matrix @ green_coords + np.array([center_x, center_y])
            blue_rotated = rotation_matrix @ blue_coords + np.array([center_x, center_y])

            red_x, red_y = red_rotated
            green_x, green_y = green_rotated
            blue_x, blue_y = blue_rotated


        # Calculate camera offset in x direction
        horizontal_offset = calculate_horizontal_distance_from_center(blue_x, green_x, red_x, VIDEO_WIDTH / 2)
        horizontal_offsets.append(horizontal_offset)

    return horizontal_offsets


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

def get_audio_levels_per_frame(video_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    n_frames = clip.n_frames

    # create fake volume, if clip has no audio
    if audio is None:
        return [-40] * (n_frames + 1) # TODO: why does videoclip has frame less than opencv for test mp4

    levels = []

    audio_clip = audio.to_soundarray()
    frame_duration = len(audio_clip) // n_frames

    # TODO: CHECK frame duration!

    # Loop over each video frame time
    for i in range(n_frames):
        t_start = i * frame_duration
        t_end = t_start + frame_duration

        samples = audio_clip[t_start:t_end]

        # Convert stereo to mono if needed
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        # Calculate RMS volume (can use other metrics)
        rms = np.sqrt(np.mean(samples ** 2))
        db = 20 * np.log10(rms + 1e-10)  # dB, avoid log(0)

        levels.append(db)

    # print(levels)
    # print(len(levels))

    return levels

def calculate_horizontal_distance_from_center(left_x, right_x, mid_x, desired_x):
    # https://en.wikipedia.org/wiki/Cross-ratio

  
    a = left_x
    b = right_x
    c = mid_x
    d = desired_x

    num = (c - a) * (d - b)
    den = (c - b) * (d - a)

    assert den != 0, "division by zero, c == b or d == a"

    cross_ratio = num / den

    a_real = 0
    b_real = TRIANGLE_SIZE
    c_real = CENTER_X

    d_real = (a_real - b_real) / ((cross_ratio * (c_real - b_real)) / (c_real - a_real) - 1) + a_real 

    return d_real - CENTER_X


    