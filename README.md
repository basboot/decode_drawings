# Fun with Radu

Very cool puzzle created by Radu Mariescu-Istodor (Lecturer in Computer Science at Karelia University of Applied
Sciences) to decode drawings by analyzing the video of a triangle of balls, taken from the top of a pencil.

https://radufromfinland.com/decodeTheDrawings/

**Excluded (data) directories**

- drawings (txt-files with drawing data)
- drawings-ai (txt-files with drawing data by neural network)
- frames (screenshots for visual comparison)
- models (the neural network models)
- output (plots)
- output-ai (plots by neural network)
- video_locations (metadata for simulated videos to train neural network)
- videos (videos from Radu, and simulation)

## Languages

- Python (for analyzing)
- Javascript (for simulation)

## Libraries

- Three.js (3D simulation)
- Scipy (optimization and filtering)
- OpenCV (image analysis)
- MoviePy (audio analysis)
- Matplotlib (for visualizing and analyzing the results)
- PyTorch (for the Neural Network)

## Simulation

A Javascript simulation of the first drawing (a circle) has been used to validate the solution on a known problem,
without disturbances: camera always horizontal, pointing at the centroid.

The simulation was also used to validate the assumptions that roll and pitch are less important than yaw. Update: roll
cannot be ignored if combined with yaw.

Also the horizontal offset estimation was validated with the simulation.

## Camera intrinsics

Both OpenCV and mathematical approaches failed on estimating the camera intrinsics, shame on me :-(

I therefore used a ruler and eyeballed the horizontal FOV at 60 degrees, and assumed square pixels and an intrinsic
point exactly at the middle of the screen.

## Image processing

For image processing _OpenCV_ was used. Because of the camera projection of the 3D balls onto 2D, they appear as
ellipses. OpenCV can find contours, fit the ellipses and get the center, and axes of the ellipses.

## Reconstructing camera position

Under the assumption that the camera starts 18cm in front of the centroid of the equilateral triangle, the initial ball
sizes in pixels are used to calculate the distances from the balls to the camera, using that the size is inversely
proportional to the distance. The 3 distances, in combination with the known viewing angle and positions of the balls
are used to reconstruct the apex of the 'pyramid'.

## Correcting errors

### Rotation (roll)

To correct rotation of the camera, the angle between blue and green is used. This angle changes slowly due to
perspective while moving. Sudden changes therefore are probably errors. We find these errors by subtracting a low pass
filter from the measured data. This error angle is then reversed, by rotating the coordinates of red, green and blue
around the center of the screen, before estimating the horizontal offset.

### Horizontal offset (yaw)

The assumption that the camera is always pointing at the centroid is not true, because of small aiming errors.
Experiments in simulation show that rotation (roll) which is very small and the y-axis (pitch) are not so much of a
problem. The x-axis (yaw) is the most important error to fix. For this we take advantage that vertical lines stay
vertical under perspective (under the assumption that roll is small). Therefore we can use the x positions of the blue,
red and green ball and the x position of the middle of the screen, together with the known size of the triangle to
calculate the real center of the camera to find the horizontal offset. We do this by using the cross-ratio of the 4
points which is projectively invariant. This offset is then used to shift the pencil perpendicular to the viewing
direction (this is not fully accurate, but helps to reduce the error).

### Vertical offset (pitch)

As stated before, the camera is not always pointed at the center of the triangle. With the camera intrinsics, and the
estimated distance the angle towards (or from) the triangle can be calculated and using simple triangle equations also
the offset towards or away from the triangle.

## Lifting the pen

To detect lifting the pen the audio track was used. Calculating the dB-level of the video and using a low pass filter
to avoid reacting to short sounds, makes it clear when the pen is on the paper or when it is in the air.

## AI

The simulation data was used to train a neural network in simulation, to predict the position on the real videos. The
results were not spectacular, but still much better than expected. With more training data results got better, but the
biggest improvement was made by normalizing the data from -1, to 1 around the center of the screen. Reducing the
network from 4 to 3 hidden layers also improved the results, which indicates there was (and still might be) overfitting
to the simulated training data. Fun fact: I have also tried to use the real video 1 (a perfect circle) as training
data, which resulted in a prediction of a morphed star, but still much better than expected... maybe also training on a
five-pointed star can improve this approach. Note: the AI has not been learned to lift the pen, instead I have used the
sound like in the non-AI decodings.

## Open ideas aka TODO

- Compare different color models. RGB works good in simulation, but maybe HSV is better for the real videos
- Filtering makes the image smoother, but not necessarily more accurate. Needs more experiments. Maybe Kalman/sensor
  fusion can do better than 'dumb' filtering.
- Decoding is now based on assumption that camera is held correctly, and corrections are done afterward. Maybe
  estimating this position without assumptions is better.
- Distance estimation is based on the size of the minor axis of the circles. There are much more sophisticated ways in
  these papers which I (unfortunately) did not get to work:
  [Single View 3D Reconstruction under an Uncalibrated Camera and an Unknown Mirror Sphere](https://www.researchgate.net/publication/311756431_Single_View_3D_Reconstruction_under_an_Uncalibrated_Camera_and_an_Unknown_Mirror_Sphere)
  and
  [A Minimal Solution for Image-Based Sphere Estimation](https://link.springer.com/article/10.1007/s11263-023-01766-1).
