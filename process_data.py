# https://radufromfinland.com/decodeTheDrawings/
from global_settings import *
from helper_functions import *
from plot_data import plot_data_grid
from process_video import get_video_data



if __name__ == '__main__':

    VIDEO = "1"
    ball_information, video_information = get_video_data(VIDEO)
    ball_information = np.array(ball_information)

    print(f"Number of video frames: {len(ball_information)}")
    print(f"Number of audio frames: {len(video_information['volume'])}")

    # work around, for missing one frame
    while len(video_information['volume']) < len(ball_information):
              video_information['volume'].append(video_information['volume'][-1])
              
    # 1-2 Hz cutoff at least needed to cancel measurement noise
    # ball_information = butter_lowpass_filter(ball_information, 5, 60, 2)

    # filter audio
    video_information['volume'] = butter_lowpass_filter(video_information['volume'], 1, 60, 2)

    green_blue_angles, green_blue_distances, horizontal_offsets = analyze_ball_information(ball_information)

    coords_x, coords_y, coords_z = calculate_camera_positions_from_rgb_major_axis(ball_information, horizontal_offsets)


    # coords_x, coords_y, coords_z = smooth_trajectory_data(coords_x, coords_y, coords_z, method='moving_average', window_size=5)

    # coords_x, coords_z = slam_like_loop_closure(coords_x, coords_z)

    # Convert lists to numpy arrays for easier manipulation in plotting
    coords_x = np.array(coords_x)
    coords_y = np.array(coords_y)
    coords_z = np.array(coords_z)

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

    drawing_x, drawing_y = [], []
    for i in range(len(coords_x)):
        if video_information["volume"][i] > DRAWING_VOLUME:
            drawing_x.append(coords_x[i])
            drawing_y.append(coords_z[i])


    # Define configurations for each plot
    plot_configs = [
        {
            "x_data": coords_x, "y_data": coords_z,
            "xlabel": 'X', "ylabel": 'Z', "title": 'Top View (X-Z)',
            "marker": 'o', "marker_size": 25, "color": None, "equal_axis": True, "mirror_y": True
        },
        {
            "x_data": drawing_x, "y_data": drawing_y,
            "xlabel": 'X', "ylabel": 'Z', "title": 'Drawing',
            "marker": 'o', "marker_size": 10, "color": "k",  "equal_axis": True, "scatter_only": True, "mirror_y": True
        },
        {
            "x_data": frames_indices, "y_data": horizontal_offsets,
            "xlabel": 'Frame', "ylabel": 'offset', "title": 'estimated cameraoffset from center',
            "marker": '.', "marker_size": 30, "color": None, 
        },
        {
            "x_data": frames_indices, "y_data": video_information["volume"],
            "xlabel": 'Frame', "ylabel": 'volume (db)', "title": 'video volume',
            "marker": '.', "marker_size": 30, "color": None, 
        },
    ]

    # plot data
    plot_data_grid(plot_configs, num_frames, VIDEO)

    with open(f"drawings/drawing{VIDEO}.txt", "w") as f:
        for x, z in zip(drawing_x, drawing_y):
            f.write(f"{x} {z}\n")

