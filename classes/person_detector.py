import torch

class PersonDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # download and load yolo model 

    def detect_people(self, image_pil):
        results = self.model(image_pil) 
        df = results.pandas().xyxy[0]
        people = df[df['name'] == 'person'] # search only people in results
        return people, results