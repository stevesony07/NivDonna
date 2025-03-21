
from ultralytics import YOLO
model= YOLO(r'C:\Users\steve\OneDrive\Desktop\Learn\python\colgproject\steve\best.pt')

model.predict(source=0, imgsz=640,conf=0.6,show=True)