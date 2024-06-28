import cv2
import mediapipe as mp
import numpy as np


def image_process(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def draw_landmarks(image, results):
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

def keypoint_extraction(results):
    # Extract keypoints from the results
    keypoints = []
    if results.left_hand_landmarks:
        keypoints.extend([(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark])
    else:
        keypoints.extend([(0, 0, 0)] * 21)  # 21 landmarks for hand

    if results.right_hand_landmarks:
        keypoints.extend([(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark])
    else:
        keypoints.extend([(0, 0, 0)] * 21)  # 21 landmarks for hand

    return np.array(keypoints).flatten()
