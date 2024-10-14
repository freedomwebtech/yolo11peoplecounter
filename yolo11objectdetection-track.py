import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load the YOLO11 model
model = YOLO("yolo11s.pt")
names=model.model.names
# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('people1.avi')
count=0
cy1=261
cy2=286
offset=8
inp={}
enter=[]
exp={}
exitp=[]
while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True,classes=0)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
       
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
                  

#    cv2.line(frame,(440,286),(1018,286),(0,0,255),2)
#    cv2.line(frame,(438,261),(1018,261),(255,0,255),2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

