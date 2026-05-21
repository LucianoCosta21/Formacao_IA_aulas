from ultralytics import YOLO
import cv2
from IPython.display import Image

model = YOLO("yolo11n.pt")

results =  model("teste.jpg")

results[0].show()

boxes = results[0].boxes

print(boxes.xyxy)

print(boxes.conf)

print(boxes.cls)

classes = {0: "person", 2:"car", 16:"dog"}

for c in boxes.cls:
    print(classes.get(int(c), 'desconhecido'))

for i, conf in enumerate(boxes.conf):
    if conf> 0.8:
         print(f"Detecção confiável: {classes.get(int(boxes.cls[i]), 'desconhecido')} ({conf:.2f})")
