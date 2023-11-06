import math
import serial
import pyttsx3
import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf

ser = serial.Serial('COM13', 9600)
cap = cv.VideoCapture(0)
hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
model = tf.keras.models.load_model('model.h5')
while True:
    success, frame = cap.read()
    results = hands.process(frame)
    datapoint = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            idx_to_coordinates = {}
            for idx, landmark in enumerate(hand_landmark.landmark):
                datapoint.append(landmark.x)
                datapoint.append(landmark.y)
                x = min(math.floor(landmark.x * frame.shape[1]), frame.shape[1] - 1)
                y = min(math.floor(landmark.y * frame.shape[0]), frame.shape[0] - 1)
                idx_to_coordinates[idx] = x, y
                cv.circle(frame, (x,y), 4, (0,0,250), -1)
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                    cv.line(frame, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], (0,250,0), 2)
    frame = cv.flip(frame, 1)
    datapoint=datapoint[:21]
    try:
        pred = model.predict([datapoint])
        [pred] = np.argmax(pred, axis=-1)
        if(pred==0): ser.write(bytearray('F', 'ascii'))
        if(pred==1): ser.write(bytearray('B', 'ascii'))
        if(pred==2): ser.write(bytearray('L', 'ascii'))
        if(pred==3): ser.write(bytearray('R', 'ascii'))
        if(pred==4): ser.write(bytearray('S', 'ascii'))
    except:
        pass
    cv.imshow('window', frame)
    c = cv.waitKey(1)
    if c==ord('q'):
        break