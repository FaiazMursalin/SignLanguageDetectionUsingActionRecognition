"""
Real time sign language detection sequence taking the whole frame from kth time to nth time(number of frames)

1. Using mediapipe holistic to extract keypoints
2. tensorflow keras LSTM DL model
3. put everything together to predict sign

"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import utility

# need to download model and detect
mp_holistic = mp.solutions.holistic
# and later draw them in video
mp_drawing = mp.solutions.drawing_utils

# Reading from webcam in real-time
cap = cv2.VideoCapture(0)

# access mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistics:
    while cap.isOpened():
        ret, frame = cap.read()

        # make detection
        image, results = utility.mediapipe_detection(frame, holistics)
        print(results)
        # draw landmarks
        utility.draw_styled_landmarks(image, results)

        cv2.imshow('Live webcam Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# print(len(results.left_hand_landmarks.landmark))  # turns it into list
#
# print(frame)  # last frame position
#
# utility.draw_landmarks(frame, results)
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # last frame photo
# plt.show()  # showing it into sciview

# append all land mark
# pose = []
# for res in results.pose_landmarks.landmark:
#     test = np.array([res.x, res.y, res.z, res.visibility])
#     pose.append(test)
# print(len(pose))
# lets do the same thing in list comprehension
# pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
#                  results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
# print(pose)
# face = np.array([[res.x, res.y, res.z] for res in
#                  results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
# print(face)
# lh = np.array([[res.x, res.y, res.z] for res in
#                results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
# print(lh)
# rh = np.array([[res.x, res.y, res.z] for res in
#                results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
# print(rh)
