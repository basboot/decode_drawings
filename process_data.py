# https://radufromfinland.com/decodeTheDrawings/
from global_settings import *
from helper_functions import *
from plot_data import plot_data_grid
from process_video import get_video_data



if __name__ == '__main__':

    VIDEO = "8"
    ball_information, video_information = get_video_data(VIDEO)
    ball_information = np.array(ball_information)
    # 1-2 Hz cutoff is needed to fully to cancel measurement noise
    ball_information = butter_lowpass_filter(ball_information, 1.5, 60, 2)

    print(len(video_information["volume"]))

    green_blue_angles, green_blue_distances, center, center_offset_px = analyze_ball_information(ball_information)
    center_offset_px = np.array(center_offset_px)

    # correct center
    # centers of the balls aren't used, so correcting them won't do anything
    ball_information[:, 0, 0:2] -= center_offset_px
    ball_information[:, 1, 0:2] -= center_offset_px
    ball_information[:, 2, 0:2] -= center_offset_px


    coords_x, coords_y, coords_z = calculate_camera_positions_from_rgb_major_axis(ball_information)


    # coords_x, coords_y, coords_z = smooth_trajectory_data(coords_x, coords_y, coords_z, method='moving_average', window_size=5)

    # coords_x, coords_z = slam_like_loop_closure(coords_x, coords_z)

    # Convert lists to numpy arrays for easier manipulation in plotting
    coords_x = np.array(coords_x)
    coords_y = np.array(coords_y)
    coords_z = np.array(coords_z)

    center = np.array(center)

    green_blue_angles = np.array(green_blue_angles)


    # Common plot setup variables
    num_frames = len(coords_x)
    frames_indices = np.arange(num_frames)

    # Prepare data for specific plots
    # Angles are around 180 degrees. Normalize green_blue_angles to be in [0, 360) range, and subtract 180 to get them around 0
    angles_to_plot = np.where(green_blue_angles < 0,
                                                green_blue_angles + 360,
                                                green_blue_angles)
    angles_to_plot -= 180
    pole_length = 18  # cm
    pole_displacement = pole_length * np.deg2rad(angles_to_plot)

    angles_to_plot_ft = butter_lowpass_filter(angles_to_plot, 0.5, 60, 2)


    # Define configurations for each plot
    plot_configs = [
        {
            "x_data": coords_x, "y_data": coords_z,
            "xlabel": 'X', "ylabel": 'Z', "title": 'Top View (X-Z)',
            "marker": 'o', "marker_size": 25, "equal_axis": True
        },
        {
            "x_data": frames_indices, "y_data": coords_y,
            "xlabel": 'Frame', "ylabel": 'Y (height)', "title": 'Camera Height',
            "marker": 'o', "marker_size": 25
        },
        {
            "x_data": frames_indices, "y_data": center[:, 0],
            "xlabel": 'Frame', "ylabel": 'center x', "title": 'estimated center x',
            "marker": '.', "marker_size": 30
        },
        {
            "x_data": frames_indices, "y_data": video_information["volume"],
            "xlabel": 'Frame', "ylabel": 'volume (db)', "title": 'video volume',
            "marker": '.', "marker_size": 30
        },
    ]

    # plot data
    plot_data_grid(plot_configs, num_frames, VIDEO)

    with open(f"drawing{VIDEO}.txt", "w") as f:
        for x, z in zip(coords_x, coords_z):
            f.write(f"{x} {z}\n")

