from ultralytics import YOLO

class Model:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def infer(self, image_path):
        return self.model(image_path)