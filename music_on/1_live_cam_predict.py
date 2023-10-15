import cv2
import numpy as np

import pygame
from keras.models import model_from_json
from keras.preprocessing import image
import os
import random

def music(music_folder):
    music_files = [f for f in os.listdir(music_folder) if f.endswith(".mp3")]

    # Check if there are music files in the folder
    if not music_files:
        print("No music files found in the folder.")
        exit()

    # Initialize the mixer (for playing audio)
    pygame.mixer.init()

    # Shuffle the list of music files to play them randomly
    random.shuffle(music_files)

    # Loop to play random songs
    for music_file in music_files:
        print(f"Song: {current_emotion }")
        print(f"Playing: {music_file}")

        # Load the music file
        pygame.mixer.music.load(os.path.join(music_folder, music_file))

        # Play the music
        pygame.mixer.music.play()

        # Wait while the music is playing
        pygame.time.delay(10000)  # A djust the delay as needed (milliseconds)

        # Stop the music
        pygame.mixer.music.stop()

# Load model from JSON file
json_file = open('top_models\\fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
pygame.mixer.init()
emotions_music = {
    "happy": "happy",
    "sad": "sad",

    "neutral": "neutral",
    "surprise":"surprise"

}
#'anger', 'disgust', 'fear'
# Load weights and them to model

model.load_weights('top_models\\fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))

        emotions = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]

        current_emotion = predicted_emotion

        if current_emotion in emotions_music:
            folder = emotions_music[current_emotion]
            music(folder)






        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        resized_img = cv2.resize(img, (1000, 700))
        cv2.imshow('Facial Emotion Recognition', resized_img)





    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
