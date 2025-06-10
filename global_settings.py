import numpy as np

# rotates clockwise for positive angles (same as open cv rectangle/ellipse rotation)
def rotate_around_z_negative_angle(vector, angle):
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle), 0],
        [np.sin(-angle), np.cos(-angle), 0],
        [0, 0, 1]
    ])

    return np.dot(rotation_matrix, vector)

RED, GREEN, BLUE = "RED", "GREEN", "BLUE"

CENTER_X = 0
POLE_SIZE = 18
TRIANGLE_SIZE = 9
BALL_RADIUS = 3

DRAWING_VOLUME = -55 # minimal volume at which we assume drawing

TRIANGLE_CENTER = np.array([CENTER_X, POLE_SIZE, 0])  # Point camera is looking at (height = 18)

INITIAL_CAMERA_POSITION = np.array([CENTER_X, POLE_SIZE, POLE_SIZE])  # Same height as triangle center

# Height of the equilateral triangle
TRIANGLE_HEIGHT = TRIANGLE_SIZE * np.sqrt(3) / 2

RED_POSITION = np.array([CENTER_X, POLE_SIZE + (2 / 3) * TRIANGLE_HEIGHT, 0])
GREEN_POSITION = np.array([CENTER_X + TRIANGLE_SIZE / 2, POLE_SIZE - (1 / 3) * TRIANGLE_HEIGHT, 0])
BLUE_POSITION = np.array([CENTER_X - TRIANGLE_SIZE / 2, POLE_SIZE - (1 / 3) * TRIANGLE_HEIGHT, 0])

INITIAL_BALL_DISTANCE = [
    np.linalg.norm(INITIAL_CAMERA_POSITION - RED_POSITION),
    np.linalg.norm(INITIAL_CAMERA_POSITION - GREEN_POSITION),
    np.linalg.norm(INITIAL_CAMERA_POSITION - BLUE_POSITION),
]

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

INITIAL_MAJOR_ELLIPSE_BOUNDARY = [
    np.array([0, 6.7422, 0.77979]),
    rotate_around_z_negative_angle(np.array([0, 6.7422, 0.77979]), 2 * np.pi / 3),
    rotate_around_z_negative_angle(np.array([0, 6.7422, 0.77979]), 4 * np.pi / 3)
]

# camera intrinsics
# assume fx = fy, principle point in middle of screen, no distortions
# f estimated from pov
K = np.array([[623.5, 0., 639.5], 
             [  0, 623.5, 359.5],
             [  0, 0, 1 ]])

if __name__ == '__main__':
    print(INITIAL_MAJOR_ELLIPSE_BOUNDARY)


