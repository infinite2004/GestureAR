import cv2
import numpy as np
import math
import RPi.GPIO as GPIO
import time

# Initialize GPIO
LED_PIN = 18  # Change this to your GPIO pin number
GPIO.setwarnings(False)  # Disable GPIO warnings
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Function to turn on the LED
def turn_on_led():
    GPIO.output(LED_PIN, GPIO.HIGH)

# Function to turn off the LED
def turn_off_led():
    GPIO.output(LED_PIN, GPIO.LOW)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function for finger tracking and gesture recognition
def finger_tracking(frame, faces):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image
    _, threshold = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological closing to remove noise
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find contour with maximum area
        max_contour = max(contours, key=cv2.contourArea)

        # Convex hull
        hull = cv2.convexHull(max_contour)

        # Draw hull
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

        # Calculate defects
        defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))
        
        # Initialize finger count
        finger_count = 0

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Calculate angles
                angle = math.degrees(math.atan2(far[1] - start[1], far[0] - start[0]))
                if angle < 0:
                    angle += 180
                
                # Filter defects based on distance and angle
                if d > 10000 and angle < 90:
                    cv2.circle(frame, far, 5, [0, 0, 255], -1)
                    finger_count += 1

        # Check if finger is close to left side of face
        led_on = False
        for (x, y, w, h) in faces:
            face_left = x
            for point in max_contour.squeeze():
                if point[0] < face_left:  # Check if finger is on the left side of the face
                    led_on = True
                    break
            if led_on:
                break

        if led_on:
            turn_on_led()
        else:
            turn_off_led()

    return frame

if __name__ == "__main__":
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the virtual glasses image
    glasses = cv2.imread('mainglasses.png', cv2.IMREAD_UNCHANGED)

    # Check if the glasses image is valid
    if glasses is None:
        print("Error: Failed to load the glasses image.")
        exit()

    # Create a VideoCapture object to capture frames from the camera
    cap = cv2.VideoCapture(0)

    # Increase brightness and contrast parameters
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break

        # Increase brightness and contrast of the frame
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # Convert the adjusted frame to grayscale for face detection
        gray = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.9, minNeighbors=2, minSize=(30, 30))

        # Convert the tuple of faces to a list
        faces = list(faces)

        # Display finger tracking
        frame_with_tracking = finger_tracking(adjusted_frame.copy(), faces.copy())

        # Overlay glasses on the frame
        for (x, y, w, h) in faces:
            # Check if the face region is valid
            if w > 0 and h > 0:
                # Resize the glasses to fit the face
                glasses_resized = cv2.resize(glasses, (w, h))

                # Calculate the position for overlaying the glasses
                x_offset = x
                y_offset = y - int(h / 3.5)

                # Get the region of interest (ROI) for overlaying the glasses
                roi = frame_with_tracking[y_offset:y_offset+glasses_resized.shape[0], x_offset:x_offset+glasses_resized.shape[1]]

                # Create a mask for the glasses to handle transparency
                mask = glasses_resized[:, :, 3] / 255.0
                mask_inv = 1.0 - mask

                # Check if ROI has non-zero dimensions before overlaying
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    # Overlay the glasses onto the frame
                    for c in range(3):
                        roi[:, :, c] = (mask * glasses_resized[:, :, c] +
                                        mask_inv * roi[:, :, c])

        # Display the resulting frame
        cv2.imshow('Finger Tracking', frame_with_tracking)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
