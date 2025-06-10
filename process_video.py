# https://radufromfinland.com/decodeTheDrawings/
import cv2
import numpy as np
import math
import json  # Added for saving data
import os  # Added for checking file existence


# https://amroamroamro.github.io/mexopencv/matlab/cv.fitEllipse.html
def ellipse_to_conic_matrix(xc, yc, width, height, angle_deg):
    a = width / 2
    b = height / 2
    theta = -np.radians(angle_deg)  # clockwise to counterclockwise

    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # Diagonal matrix with inverse squares
    D = np.diag([1 / a**2, 1 / b**2])

    # Conic part in 2D
    C_2x2 = R.T @ D @ R

    # Center
    c = np.array([xc, yc])
    C_2 = -C_2x2 @ c
    C_3 = c.T @ C_2x2 @ c - 1

    # Final 3x3 conic matrix using numpy
    C = np.block([
        [C_2x2, C_2.reshape(2,1)],
        [C_2.reshape(1,2), np.array([[C_3]])]
    ])

    return C


from global_settings import *
from helper_functions import *
from tqdm import tqdm


# processes the given video (only use the number of the video 1-6) and returns original_ball_sizes, ball_sizes
# also saves the data to disc, and tries to reuse unless use_cache = False
def get_video_data(video, use_cache = True, createTrainingData = False, showVideo = False, saveFrames = False):
    ball_sizes = []

    extension = "mp4"

    ball_data_filepath = f"data/ball_sizes_data{video}.json"

    data_loaded_successfully = False
    if use_cache and os.path.exists(ball_data_filepath):
        print(f"Found data file: {ball_data_filepath}. Attempting to load...")
        try:
            with open(ball_data_filepath, "r") as f:
                loaded_data = json.load(f)
            ball_sizes = loaded_data['ball_data']
            video_info_property = loaded_data['video']
            print("Ball data loaded successfully from file.")
            data_loaded_successfully = True
        except Exception as e:
            print(f"Error loading data from {ball_data_filepath}: {e}. Will process video instead.")
            ball_sizes = []

    if not data_loaded_successfully:
        print("Processing video to gather sound data...")


        if not os.path.exists(f"videos/{video}.{extension}"):
            print("mp4 not found, fallback to webm")
            extension = "webm"

        audio_levels = get_audio_levels_per_frame(f"videos/{video}.{extension}")

        print("Processing video to gather ball data...")
        cap = cv2.VideoCapture(f"videos/{video}.{extension}")



        frame_width = None
        frame_height = None
        total_frames_in_video = None

        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()  # Exit if video cannot be opened and no data loaded
        else:
            print("Video file opened successfully!")
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        count = 0
        ball_sizes = []  # Ensure it's empty

        # TODO: add progress bar

        progress_bar = tqdm(total=total_frames_in_video, desc="Processing Frames", unit="frame")
        slowmo = True
        while True:
            progress_bar.update(1)
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            # Save first and last frame as a PNG image for visual comparison
            if saveFrames and (count == 0 or count == total_frames_in_video - 1):
                frame_filename = f"frames/video_{video}_frame_{count:04d}.png"
                os.makedirs("frames", exist_ok=True)  # Ensure the directory exists
                cv2.imwrite(frame_filename, frame)
     

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

            # TODO: create functions to easiliy choose between masks
            # Convert to RGB for better color segmentation
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create better masks based on RGB dominance and account for shadows
            red_mask = (rgb[:, :, 0] > 100) & (rgb[:, :, 0] > rgb[:, :, 1] + 30) & (rgb[:, :, 0] > rgb[:, :, 2] + 30)
            green_mask = (rgb[:, :, 1] > 100) & (rgb[:, :, 1] > rgb[:, :, 0] + 30) & (rgb[:, :, 1] > rgb[:, :, 2] + 30) & (rgb[:, :, 2] < 100)
            blue_mask = (rgb[:, :, 2] > 100) & (rgb[:, :, 2] > rgb[:, :, 0] + 30) & (rgb[:, :, 2] > rgb[:, :, 1] + 30)

            # Convert boolean masks to uint8 (binary masks)
            mask_red = red_mask.astype(np.uint8) * 255
            mask_green = green_mask.astype(np.uint8) * 255
            mask_blue = blue_mask.astype(np.uint8) * 255

            current_frame_data = []  # Stores [R_data, G_data, B_data] for this frame
            output = frame.copy()

            for color_mask, color_name in [(mask_red, RED), (mask_green, GREEN), (mask_blue, BLUE)]:
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                ball_found_for_color = False
                largest_contour = None
                max_points = 0

                for cnt in contours:
                    num_points = len(cnt)
                    if num_points > max_points:
                        largest_contour = cnt
                        max_points = num_points

                lengths = [len(cnt) for cnt in contours]
                lengths.sort(reverse=True)
                if len(lengths) > 1 and lengths[1] > 4:
                    print("WARNING: multiple large countours")

                if largest_contour is not None:
                    ellipse = cv2.fitEllipse(largest_contour)
                    # print(f"Frame: {count} #points {len(largest_contour)}", lengths)
                    cv2.ellipse(output, ellipse, (0, 255, 0), 2)
                    (x_ellipse, y_ellipse), (minor, major), angle_ellipse = ellipse

                    if minor > major:
                        print(f"WARNING: minor and major axes reversed in frame {count}")

                    ball_data = [x_ellipse, y_ellipse, minor, major, angle_ellipse]

                    current_frame_data.append(ball_data)

                    ball_found_for_color = True

                if not ball_found_for_color:
                    assert count > 0, f"Problem reading first frame, cannot find color {color_name}."
                    print(f"WARNING: problem reading frame {count}, cannot find color {color_name}.")

            if len(current_frame_data) == 3:
                ball_sizes.append(current_frame_data)
            else:
                assert count > 0, f"Problem reading first frame, found {len(current_frame_data)} contours. Cannot continue."
                print(f"WARNING: problem reading frame {count}, found {len(current_frame_data)} contours. Using previous frame as work around.")
                prev_ball_sizes = ball_sizes[-1]
                ball_sizes.append(prev_ball_sizes)


            if showVideo:
                if slowmo:
                    print("Data for red, green, blue:")
            # Draw ellipses and axes on the frame
                for ball_data in current_frame_data:
                    x_ellipse, y_ellipse, minor, major, angle_ellipse = ball_data

                    if slowmo:
                        print(ball_data)

                    center = (int(x_ellipse), int(y_ellipse))
                    axes = (int(minor / 2), int(major / 2))
                    cv2.ellipse(output, center, axes, angle_ellipse, 0, 360, (255, 255, 255), 2)
                            
                    # Draw the minor axis
                    cv2.line(output, center, 
                             (center[0] + int(axes[0] * math.cos(math.radians(angle_ellipse))),
                              center[1] + int(axes[0] * math.sin(math.radians(angle_ellipse)))), 
                             (128, 128, 128), 2)
                    cv2.line(output, center, 
                             (center[0] - int(axes[0] * math.cos(math.radians(angle_ellipse))),
                              center[1] - int(axes[0] * math.sin(math.radians(angle_ellipse)))), 
                             (128, 128, 128), 2)

                    # Draw the major axis
                    cv2.line(output, center, 
                             (center[0] - int(axes[1] * math.sin(math.radians(angle_ellipse))),
                              center[1] + int(axes[1] * math.cos(math.radians(angle_ellipse)))), 
                             (0, 0, 0), 2)
                    cv2.line(output, center, 
                             (center[0] + int(axes[1] * math.sin(math.radians(angle_ellipse))),
                              center[1] - int(axes[1] * math.cos(math.radians(angle_ellipse)))), 
                             (0, 0, 0), 2)

                # Display the current frame count
                cv2.putText(output, f"Frame: {count}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                # Display the frame with overlays
                cv2.imshow("Processed Frame", output)
                
                if cv2.waitKey(10000 if slowmo else 1) != -1:
                    # key was pressed, toggle slowmo
                    slowmo = not slowmo

            count += 1

        cap.release()

        print("Video processing complete.")
        print("Number of frames processed:", len(ball_sizes))

        video_info_property = {
            "name": str(video),
            "frames": total_frames_in_video,
            "width": frame_width,
            "height": frame_height,
            "volume": audio_levels
        }
        data_to_save = {
            'video': video_info_property,
            'ball_data': ball_sizes
        }
        try:
            with open(ball_data_filepath, "w") as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Successfully saved ball data to {ball_data_filepath}")
        except Exception as e:
            print(f"Error saving ball data: {e}")

    if not ball_sizes:
        print("Error: Ball data is not available after attempting load/processing. Exiting.")
        exit()

    if createTrainingData:
        try:
            training_data_filepath = f"video_locations/{video}.json"
            if os.path.exists(training_data_filepath):
                print(f"Found training data file: {training_data_filepath}. Attempting to load...")
                with open(training_data_filepath, "r") as f:
                    training_data = json.load(f)
                print("Training data loaded successfully.")
            else:
                print(f"Training data file {training_data_filepath} does not exist.")
                training_data = []
        except Exception as e:
            print(f"Error loading training data from {training_data_filepath}: {e}")
            training_data = []

        assert len(training_data) == len(ball_sizes), f"Mismatch in data sizes: training_data ({len(training_data)}) and ball_sizes ({len(ball_sizes)})"



    if createTrainingData:
        return ball_sizes, video_info_property, training_data
    
    return ball_sizes, video_info_property