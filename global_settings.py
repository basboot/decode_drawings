import numpy as np

RED, GREEN, BLUE = "RED", "GREEN", "BLUE"

CENTER_X = 4.5
POLE_SIZE = 18
TRIANGLE_SIZE = 9

TRIANGLE_CENTER = np.array([CENTER_X, POLE_SIZE, 0])  # Point camera is looking at (height = 18)

INITIAL_CAMERA_POSITION = np.array([CENTER_X, POLE_SIZE, POLE_SIZE])  # Same height as triangle center

RED_POSITION = np.array([CENTER_X, POLE_SIZE + (2 / 3) * TRIANGLE_SIZE, 0])
GREEN_POSITION = np.array([TRIANGLE_SIZE, POLE_SIZE - TRIANGLE_SIZE / 3, 0])
BLUE_POSITION = np.array([0, POLE_SIZE - TRIANGLE_SIZE / 3, 0])
