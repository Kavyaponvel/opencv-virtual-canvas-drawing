import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas (same size as webcam feed)
canvas = None

# Define color range for detection (example: blue object)
lower_color = np.array([20, 100, 100])
upper_color = np.array([30, 255, 255])


# Previous point (for drawing lines)
prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame (mirror view)
    frame = cv2.flip(frame, 1)

    # Initialize canvas once we know frame size
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for the chosen color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Reduce noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours (to detect the object)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:  # Ignore small objects/noise
            # Get the center of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            center_x, center_y = x + w // 2, y + h // 2

            # Draw a circle on the frame to show tracking
            cv2.circle(frame, (center_x, center_y), 10, (0, 255, 255), -1)


            # Draw line on canvas
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (center_x, center_y), (0, 255, 255), 5)


            prev_x, prev_y = center_x, center_y
        else:
            prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    # ✅ Combine frame + canvas with opacity
    alpha = 0.7  # opacity of original frame
    beta = 0.3   # opacity of drawings
    combined = cv2.addWeighted(frame, alpha, canvas, beta, 0)

    # Show windows
    cv2.imshow("Mask", mask)
    cv2.imshow("Virtual Drawing Board", combined)

    # Key bindings
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Press 'c' to clear canvas
        canvas = np.zeros_like(frame)
    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
