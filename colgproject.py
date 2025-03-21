import cv2
import numpy as np
from audioplayer import AudioPlayer

# Function to detect fire in the video frame
def detect_fire(frame, background_subtractor):
    # Resize the frame for faster processing
    frame = cv2.resize(frame, (960, 540))

    # Apply Gaussian blur to reduce noise and improve detection
    blur = cv2.GaussianBlur(frame, (21, 21), 0)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Define HSV range for detecting fire-like colors
    lower = np.array([0, 50, 50], dtype="uint8")
    upper = np.array([35, 255, 255], dtype="uint8")

    # Create a mask for fire colors
    mask = cv2.inRange(hsv, lower, upper)

    # Apply background subtraction to detect motion
    fg_mask = background_subtractor.apply(frame)

    # Combine color mask and motion mask
    combined_mask = cv2.bitwise_and(mask, mask, mask=fg_mask)

    # Perform morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours match the criteria for fire
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Adjust area threshold as needed
            return True

    return False

# Function to play the alert sound
def show_alert():
    print("Fire Detected!")
    #AudioPlayer(r"C:\Users\steve\OneDrive\Desktop\Learn\python\alarm-sound.mp3").play(block=True,loop=False)

# Main function
def main():
    # Capture video from the default camera
    cap = cv2.VideoCapture(0)

    # Create background subtractor for motion detection
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Check for fire in the frame
        if detect_fire(frame, background_subtractor):
            show_alert()

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
