import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np
import os

# Suppress TensorFlow Lite logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Try multiple camera indices to avoid 'Camera index out of range' error
for i in range(3):  # Try up to 3 possible camera indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} selected.")
        break
else:
    raise ValueError("No camera found or all camera indices are out of range.")

cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

# Initial rectangle parameters
cx, cy, w, h = 100, 100, 200, 200


class DragRect:
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter  # Center of the rectangle
        self.size = size  # Width and height of the rectangle

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # Check if the index finger tip is inside the rectangle
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor  # Update the center to follow the cursor


# Create a list of 5 rectangles at different horizontal positions
rectList = [DragRect([x * 250 + 150, 150]) for x in range(5)]

while True:
    # Read frame from the webcam
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Find hands in the image and landmarks
    hands, img = detector.findHands(img)  # Updated method returns both hands data and image
    if hands:
        lmList = hands[0]['lmList']  # Landmark list for the first hand

        # Get the coordinates of index finger (landmark 8) and middle finger (landmark 12)
        x1, y1, _ = lmList[8]  # Index finger tip (ignore z value)
        x2, y2, _ = lmList[12]  # Middle finger tip (ignore z value)

        # Find the distance between index and middle fingers
        l, _, _ = detector.findDistance((x1, y1), (x2, y2), img)  # Pass coordinates, not indices

        # If the distance is less than 30, update the rectangle position
        if l < 30:
            cursor = lmList[8][:2]  # Index finger tip position (x, y)
            for rect in rectList:
                rect.update(cursor)  # Update rectangle based on cursor

    # Transparent layer for rectangles
    imgNew = np.zeros_like(img, np.uint8)  # Create a black image with the same size as the frame

    # Draw the rectangles on the transparent layer
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        # Draw filled rectangle
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        # Draw the corner rectangle with rounded corners using cvzone
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # Blend the transparent rectangles with the original frame
    out = img.copy()  # Copy of the original frame
    alpha = 0.5  # Transparency factor
    mask = imgNew.astype(bool)  # Mask where the new image has content
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # Display the final output image
    cv2.imshow("Image", out)

    # Wait for the key press and exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

