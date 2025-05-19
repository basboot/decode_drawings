# https://radufromfinland.com/decodeTheDrawings/
import cv2
import numpy as np
import math

RED, GREEN, BLUE = "RED", "GREEN", "BLUE"

def estimate_undistorted_radius(major, minor):
    a = major / 2
    b = minor / 2
    r_undistorted = math.sqrt(a * b)  # geometric mean
    return r_undistorted

ball_sizes = []
count = 0

if __name__ == '__main__':
    cap = cv2.VideoCapture("videos/1.mp4")

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        print("Video file opened successfully!")

    # Read the first frame to confirm reading
    while True:
        ret, frame = cap.read()

        original_ball_sizes = {}

        if ret:
            # Display the frame using imshow
            # cv2.imshow("First Frame", frame)
            # cv2.waitKey(0)  # Wait for a key press to close the window
            # cv2.destroyAllWindows()  # Close the window

            # output = frame.copy()
            # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #
            # # Preprocessing: blur and enhance contrast
            # img = cv2.GaussianBlur(img, (9, 9), 2)
            # img = cv2.equalizeHist(img)
            #
            # # Find circles
            # # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, 100)
            #
            # # Hough Circle Detection with tuned parameters
            # circles = cv2.HoughCircles(
            #     img,
            #     cv2.HOUGH_GRADIENT,
            #     dp=1.2,  # Resolution factor
            #     minDist=40,  # Minimum distance between circle centers
            #     param1=100,  # Higher threshold for Canny edge detector
            #     param2=30,  # Accumulator threshold — tweak this!
            #     minRadius=100,
            #     maxRadius=150
            # )
            #
            # # If some circle is found
            # if circles is not None:
            #     # Get the (x, y, r) as integers
            #     circles = np.round(circles[0, :]).astype("int")
            #     print(circles)
            #     # loop over the circles
            #     for (x, y, r) in circles:
            #         cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            # # show the output image
            # cv2.imshow("circle", output)
            # cv2.waitKey(0)

            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define color ranges (tune these for your images)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])

            lower_green = np.array([40, 70, 70])
            upper_green = np.array([80, 255, 255])

            lower_blue = np.array([100, 150, 0])
            upper_blue = np.array([140, 255, 255])

            # Create masks
            # TODO: do not combine, but perform 3 times
            mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

            # TODO: change name, since we do not combine anymore
            sizes = []
            for combined_mask, color in [(mask_red, RED), (mask_green, GREEN), (mask_blue, BLUE)]:

                # Combine all masks
                # combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_green), mask_blue)

                # Optional: Clean up mask
                # combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
                # erosion not needed after fremoving the blur
                # combined_mask = cv2.erode(combined_mask, np.ones((3, 3), np.uint8), iterations=1)

                # TODO: check if needed
                # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

                # Find contours on the clean mask
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw contours
                # output = frame.copy()
                # for cnt in contours:
                #     area = cv2.contourArea(cnt)
                #     if area > 100:
                #         cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
                #
                # cv2.imshow("contours", output)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                output = frame.copy()

                for cnt in contours:
                    if len(cnt) >= 5:  # Need at least 5 points to fit an ellipse
                        ellipse = cv2.fitEllipse(cnt)
                        cv2.ellipse(output, ellipse, (0, 255, 0), 2)

                        (x, y), (major, minor), angle = ellipse
                        # Print the parameters
                        # print(f"Ellipse center: ({x:.2f}, {y:.2f})")
                        # print(f"Major axis length: {max(major, minor):.2f}")
                        # print(f"Minor axis length: {min(major, minor):.2f}")
                        # print(f"Rotation angle: {angle:.2f} degrees")
                        # print("---")

                        angle_rad = math.radians(angle)

                        # Half-lengths
                        a = major / 2
                        b = minor / 2

                        # Major axis vector
                        x1 = int(x + a * math.cos(angle_rad))
                        y1 = int(y + a * math.sin(angle_rad))
                        x2 = int(x - a * math.cos(angle_rad))
                        y2 = int(y - a * math.sin(angle_rad))

                        # Minor axis vector (perpendicular)
                        x3 = int(x + b * math.cos(angle_rad + math.pi / 2))
                        y3 = int(y + b * math.sin(angle_rad + math.pi / 2))
                        x4 = int(x - b * math.cos(angle_rad + math.pi / 2))
                        y4 = int(y - b * math.sin(angle_rad + math.pi / 2))

                        # Draw major axis (blue)
                        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # Draw minor axis (red)
                        cv2.line(output, (x3, y3), (x4, y4), (0, 0, 255), 2)
                        # Green: fitted ellipse
                        # Blue: major axis
                        # Red: minor axis

                        # Assume (major, minor) from fitEllipse
                        a = max(major, minor)
                        b = min(major, minor)

                        # Clamp b/a to avoid math domain errors
                        ratio = max(min(b / a, 1.0), 0.0)

                        theta_rad = math.acos(ratio)
                        theta_deg = math.degrees(theta_rad)

                        # print(f"Approximate tilt angle relative to camera: {theta_deg:.2f}°")

                        # print(f"Approximate radius of ball in pixels: {estimate_undistorted_radius(a, b):.2f}px")

                        estimated_size = estimate_undistorted_radius(a, b)
                        if count == 0:
                            original_ball_sizes[color] = estimated_size

                        sizes.append(estimated_size)


            count += 1
            ball_sizes.append(sizes)

            # DEBUG
            if count > 10:
                print("end after 10 frames")
                break






                # cv2.imshow("ellipses", output)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()



        else:
            print("End of video.")
            break


    # Release the video capture object
    cap.release()

    print(original_ball_sizes)

    # calculate original distance
    bisec = math.sqrt(9*9 + 4.5*4.5)
    line_to_center = 2/3 * bisec
    original_distance = math.sqrt(line_to_center*line_to_center + 18*18)

    print(f"Distance: {original_distance}")

    print(ball_sizes)

