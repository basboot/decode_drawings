# Fun with Radu

Very cool puzzle created by Radu Mariescu-Istodor (Lecturer in Computer Science at Karelia University of Applied
Sciences) to decode drawing by analyzing the video of a triangle of balls, taken from the top of a pencil.

https://radufromfinland.com/decodeTheDrawings/

## Languages

- Python (for analyzing)
- Javascript (for simulation)

## Libraries

- Three js (3D simulation)
- Scipy (optimization and filtering)
- OpenCV (image analyzing)
- MoviePy (audio analyzing)
- Matplotlib (for visualizing and analyzing the results)

## Simulation

A Javascript simulation of the first drawing (a circle) has been used to validate the solution on a known problem,
without disturbances: camera always horizontal, pointing at the centroid.

The simualtion was also used to validate the assumptions that roll and pitch are (within limits) not very important.
Update: roll cannot be ignored if combined with yaw, pitch and yaw does not seem to be an extra complication.

Also the horizontal offset estimation was validated with the simulation.

## Image processing

For image processing _OpenCV_ was used. Because of the camera projection of the 3D balls onto 2D, they appear as
ellipses. OpenCV can find contours, fit the ellipses and get the center, and axes of the ellipses.

## Reconstructing camera position

Under the assumption that the camera starts 18cm in front of the cetroid of the equilateral triangle, the initial ball
sizes in pixels are used to calculate the distances to the balls to the camera, using that the size is reversed
proportional to the distance. The 3 distances, in combination with the known viewing angle and positions of the balls
are used to reconstruct the apex of the 'pyramid'.

## Correcting errors

### Rotation (roll)

To correct rotation of the camera, the angle between blue and green is used. This angle changes slowly due to
persective while moving. Sudden moves therefore are probably errors. We find these errors by subtracting a low pass
filter from the measured data. This error angle is then reversed, by rotating the coordinates of red, green and blue
around the center of the screen, before estimating the horizontal offset.

### Horizontal offset (yaw)

The assumption that the camera is always pointing at the centroid is not true, because of small aiming errors.
Experiments in simulation show that rotation (roll) which is very small and the y-axis (pitch) are not so much of a
problem. The x-axis (yaw) is the most important error to fix. For this we take advantage that vertical lines stay
vertical under perspective (under the assumption that roll is small). Therefore the we can use the x positions of the
blue, red and green ball and the x position of the middle of the screen, together with the known size of the triangle
to calculate the real center of the camera to find the horizontal offset. We do this by using the cross-ratio of the 4
points which are projective invariant. This offset is then used to shift the pencil perpendicular to the viewing
direction (this is not fully accurate, but helps to reduce the error).

## Lifting the pen

To detect lifting the pen the audio track was used. Calculating the dB-level of the video and using a low pass filter
to avoid reacting to short sounds, makes it clear when the pen is on the paper or when it is in the air.

## Open ideas aka TODO

- Use x, z position to estimate correct angle for blue-green, instead of filtering the measured angles
- Explore video frames for bad parts of the drawing.
- Check different color models. RGB works good in simulation, but maybe HSV is better for the real videos
- Filtering makes the image more smooth, but not necessary more accurate. Needs more experiments. Maybe Kalmann/sensor
  fusion can do better, than 'dumb' filtering.
- Improve the simulation so it can be used to train a neural network.
