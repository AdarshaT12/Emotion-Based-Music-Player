import cv2
import numpy as np
from fer import FER
import pygame
import time

# Initialize music player
pygame.mixer.init()

songs = {
    "happy": "songs/happy.mp3",
    "sad": "songs/sad.mp3",
    "angry": "songs/angry.mp3",
    "surprise": "songs/surprise.mp3"
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

emotion_detector = FER()

cap = cv2.VideoCapture(0)

current_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        face = frame[y:y+h, x:x+w]

        result = emotion_detector.detect_emotions(face)

        if result:

            emotion = max(
                result[0]["emotions"],
                key=result[0]["emotions"].get
            )

            cv2.putText(
                frame,
                emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,0,255),
                2
            )

            if emotion in songs and current_emotion != emotion:

                pygame.mixer.music.load(songs[emotion])
                pygame.mixer.music.play()

                current_emotion = emotion

    cv2.imshow("Emotion Based Music Player", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
