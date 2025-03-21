from ultralytics import YOLO
import cv2
import pygame  # Library to play sound

# Initialize pygame mixer for sound playback
pygame.mixer.init()
fire_alert_sound = r'C:\Users\steve\OneDrive\Desktop\Learn\python\colgproject\steve\AUD-20241203-WA0030.mp3'  # Path to your alert sound file
pygame.mixer.music.load(fire_alert_sound)

# Load the YOLO model
model = YOLO(r'C:\Users\steve\OneDrive\Desktop\Learn\python\colgproject\steve\best.pt')

# Function to play the fire alert sound
def play_fire_alert_sound():
    if not pygame.mixer.music.get_busy():  # Check if sound is already playing
        pygame.mixer.music.play(-1)  # Play in a loop

# Function to stop the fire alert sound
def stop_fire_alert_sound():
    if pygame.mixer.music.get_busy():  # Check if sound is playing
        pygame.mixer.music.stop()

# Start prediction from webcam
cap = cv2.VideoCapture(0)  # Open the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Make predictions
    results = model.predict(source=frame, imgsz=640, conf=0.6)

    # Display the predictions
    annotated_frame = results[0].plot()  # Annotated frame with bounding boxes and labels
    cv2.imshow("Fire Detection", annotated_frame)

    # Check for 'fire' in detected classes
    fire_detected = False
    for detection in results[0].boxes.data:
        label = results[0].names[int(detection[5])]  # Get the class label
        if label.lower() == 'fire':
            fire_detected = True
            break

    if fire_detected:
        play_fire_alert_sound()
    else:
        stop_fire_alert_sound()

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
