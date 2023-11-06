import cv2 as cv
import mediapipe as mp
import math
import csv 

f = open(r"dataset.csv", 'w', newline='')
writer = csv.writer(f)

i=0
cap = cv.VideoCapture(0)
hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
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
    cv.imshow('Hands', cv.flip(frame, 1))
    c = cv.waitKey(1)
    if c==ord('a'):
        i+=1
        print(i)
        datapoint=datapoint[:21]
        datapoint.append(0)
        writer.writerow(datapoint)
    elif c==ord('b'):
        i+=1
        print(i)
        datapoint=datapoint[:21]
        datapoint.append(1)
        writer.writerow(datapoint)
    elif c==ord('c'):
        i+=1
        print(i)
        datapoint=datapoint[:21]
        datapoint.append(2)
        writer.writerow(datapoint)
    elif c==ord('d'):
        i+=1
        print(i)
        datapoint=datapoint[:21]
        datapoint.append(3)
        writer.writerow(datapoint)
    elif c==ord('e'):
        i+=1
        print(i)
        datapoint=datapoint[:21]
        datapoint.append(4)
        writer.writerow(datapoint)
    elif c==ord('f'):
        i+=1
        print(i)
        datapoint=datapoint[:21]
        datapoint.append(5)
        writer.writerow(datapoint)
    elif c==ord('g'):
        i+=1
        print(i)
        datapoint=datapoint[:21]
        datapoint.append(6)
        writer.writerow(datapoint)
    elif c==ord('h'):
        i+=1
        print(i)
        datapoint=datapoint[:21]
        datapoint.append(7)
        writer.writerow(datapoint)
    elif c==32: #space
        break

f.close()
cap.release()
cv.destroyAllWindows()