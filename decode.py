import cv2
import numpy as np
import math

RED, GREEN, BLUE = "RED", "GREEN", "BLUE"
TRIANGLE_CENTER = np.array([4.5, 18, 0])
DEBUG = True

COLOR_RANGES = {
    RED:   [(np.array([0, 100, 100]), np.array([10, 255, 255])), (np.array([160, 100, 100]), np.array([179, 255, 255]))],
    GREEN: [(np.array([40, 70, 70]), np.array([80, 255, 255]))],
    BLUE:  [(np.array([100, 150, 0]), np.array([140, 255, 255]))]
}

def estimate_undistorted_radius(major, minor):
    return math.sqrt((major / 2) * (minor / 2))

def get_color_mask(hsv, color):
    masks = [cv2.inRange(hsv, low, high) for low, high in COLOR_RANGES[color]]
    return np.bitwise_or.reduce(masks)

def process_contour(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [cnt for cnt in contours if len(cnt) > 4]
    assert len(valid) == 1, f"Found {len(valid)} valid contours (expected 1) in color mask"
    
    ellipse_fit_data = cv2.fitEllipse(valid[0])
    (x, y), (raw_axis_1, raw_axis_2), angle = ellipse_fit_data
    
    major_axis = max(raw_axis_1, raw_axis_2)
    minor_axis = min(raw_axis_1, raw_axis_2)
    
    estimated_radius = estimate_undistorted_radius(major_axis, minor_axis)

    if DEBUG:
        output = frame.copy()
        cv2.ellipse(output, ellipse_fit_data, (0, 255, 0), 2)
        # Use raw_axis_1 and raw_axis_2 for drawing, as they correspond to 'angle'
        a, b = raw_axis_1 / 2, raw_axis_2 / 2
        angle_rad = math.radians(angle)
        # Define axes for drawing
        axes_lines = [
            ((int(x + a * math.cos(angle_rad)), int(y + a * math.sin(angle_rad))),
             (int(x - a * math.cos(angle_rad)), int(y - a * math.sin(angle_rad))), (255, 0, 0)), # Axis 1
            ((int(x + b * math.cos(angle_rad + math.pi / 2)), int(y + b * math.sin(angle_rad + math.pi / 2))),
             (int(x - b * math.cos(angle_rad + math.pi / 2)), int(y - b * math.sin(angle_rad + math.pi / 2))), (0, 0, 255))  # Axis 2
        ]
        for pt1, pt2, line_color in axes_lines: # Renamed 'color' to 'line_color'
            cv2.line(output, pt1, pt2, line_color, 2)
        cv2.imshow("contours", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return {
        'center_x': float(x),
        'center_y': float(y),
        'major_axis': float(major_axis),
        'minor_axis': float(minor_axis),
        'angle': float(angle),
        'estimated_radius': float(estimated_radius)
    }

    

if __name__ == '__main__':
    cap = cv2.VideoCapture("videos/1.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    print("Video file opened successfully!")
    
    frame_info = [] # This will be a list of dictionaries
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        
        if DEBUG:
            cv2.imshow("First Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        current_frame_details = {} # Dictionary for the current frame's ball data
        for color_key in [RED, GREEN, BLUE]:
            mask = get_color_mask(hsv, color_key)
            ball_details = process_contour(frame, mask)
            current_frame_details[color_key] = ball_details
            
        frame_info.append(current_frame_details)
        
        if DEBUG: # If DEBUG is true, process only the first frame
            break
            
    cap.release()
    print("Frame info:", frame_info)
    print("Number of frames processed:", len(frame_info))

