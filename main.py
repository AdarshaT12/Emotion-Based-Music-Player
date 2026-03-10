import cv2
from fer import FER
import webbrowser
import time

# -------------------------------
# YouTube Songs / Playlists
# -------------------------------

emotion_songs = {
    "happy": "https://www.youtube.com/watch?v=UPkMkIOzej8",
    "sad": "https://www.youtube.com/watch?v=s7FTAxw37hk",
    "angry": "https://www.youtube.com/watch?v=hTWKbfoikeg",
    "surprise": "https://www.youtube.com/watch?v=CpNBODqTA34"
}

# -------------------------------
# Initialize Emotion Detector
# -------------------------------

detector = FER(mtcnn=True)

# -------------------------------
# Initialize Webcam
# -------------------------------

cap = cv2.VideoCapture(0)

current_emotion = None
last_open_time = 0
cooldown = 10   # seconds before opening another song

print("Emotion Based Music Player Started")
print("Press Q to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect emotions in frame
    results = detector.detect_emotions(frame)

    for face in results:

        x, y, w, h = face["box"]

        # Draw rectangle around face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # Get dominant emotion
        emotion = max(face["emotions"], key=face["emotions"].get)

        # Display emotion on screen
        cv2.putText(
            frame,
            emotion.upper(),
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

        # Open YouTube song
        if emotion in emotion_songs:

            if emotion != current_emotion and time.time() - last_open_time > cooldown:

                url = emotion_songs[emotion]

                print("Detected Emotion:", emotion)
                print("Opening Song:", url)

                webbrowser.open(url)

                current_emotion = emotion
                last_open_time = time.time()

    # Show webcam window
    cv2.imshow("Emotion Based Music Player", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------
# Release resources
# -------------------------------

cap.release()
cv2.destroyAllWindows()
