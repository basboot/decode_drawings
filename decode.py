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
    assert len(valid) == 1, "Found multiple contours in same color mask"
    ellipse = cv2.fitEllipse(valid[0])
    (x, y), (major, minor), angle = ellipse
    if DEBUG:
        output = frame.copy()
        cv2.ellipse(output, ellipse, (0, 255, 0), 2)
        a, b = major / 2, minor / 2
        angle_rad = math.radians(angle)
        axes = [
            ((int(x + a * math.cos(angle_rad)), int(y + a * math.sin(angle_rad))),
             (int(x - a * math.cos(angle_rad)), int(y - a * math.sin(angle_rad))), (255, 0, 0)),
            ((int(x + b * math.cos(angle_rad + math.pi / 2)), int(y + b * math.sin(angle_rad + math.pi / 2))),
             (int(x - b * math.cos(angle_rad + math.pi / 2)), int(y - b * math.sin(angle_rad + math.pi / 2))), (0, 0, 255))
        ]
        for pt1, pt2, color in axes:
            cv2.line(output, pt1, pt2, color, 2)
        cv2.imshow("contours", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return estimate_undistorted_radius(max(major, minor), min(major, minor))

    

if __name__ == '__main__':
    cap = cv2.VideoCapture("videos/1.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    print("Video file opened successfully!")
    ball_sizes = []
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
        sizes = []
        for color in [RED, GREEN, BLUE]:
            mask = get_color_mask(hsv, color)
            sizes.append(process_contour(frame, mask))
        ball_sizes.append(sizes)
        if DEBUG:
            break
    cap.release()
    print("Ball sizes:", ball_sizes)
    print("Number of frames:", len(ball_sizes))
