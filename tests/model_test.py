# test bib detector with trained model
from ultralytics import YOLO
import os

model = YOLO('../models/model.pt')

print("Folder path:")
path = input()

for file in os.listdir(path):
    full_path = os.path.join(path, file)
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        results = model(full_path)
        results[0].show()
        break