import torch
import numpy as np
import cv2

# model
model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=True)

# vedio capure
cap = cv2.VideoCapture("parking.mp4")

targetClasses = ["car"]

TheArea = [[4, 359], [2, 431], [287, 415], [600, 353], [539, 300]]

# def area(event, x, y, flags, _):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         TheArea.append([x, y])
        
cv2.namedWindow("frame")
        

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    else:
        frame = cv2.resize(frame, (800, 500))
        result = model(frame)
        count = []
        for index, row in result.pandas().xyxy[0].iterrows():
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            label = row['name']
            cx = int((xmax+xmin)/2)
            cy = int((ymax+ymin)/2)
            if label in targetClasses:
                check = cv2.pointPolygonTest(np.array(TheArea), (cx, cy), False)
                if check >= 0:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 1)
                    cv2.putText(frame, f"{label}", (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255], 1)
                    count.append([cx])
        counts = len(count)
        cv2.polylines(frame, [np.array(TheArea)], True, [255, 0, 0], 2)
        cv2.putText(frame, f"Number of cars in the parking area: {counts}", (8, 35), cv2.FONT_HERSHEY_COMPLEX,1 , [255,20,20], 2)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()