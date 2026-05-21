from ultralytics import YOLO
import cv2
from IPython.display import Image, Video

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO11 - Webcam', annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


#results =  model.predict(source="teste.jpg", save=True)
#results_video =  model.predict(source="video_teste.mp4", save=True)

##img = cv2.imread('runs/detect/predict/teste.jpg')
#Image(filename='runs/detect/predict/teste.jpg')
#Video(filename='runs/detect/predict/video_teste.mp4')

"""cv2.imshow("Deteccao", img)
cv2.waitKey(0)
cv2.destroyAllWindows() """
