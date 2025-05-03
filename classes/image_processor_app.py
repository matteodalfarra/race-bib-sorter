import os
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
from collections import defaultdict

from classes.model import Model
from classes.text_recognizer import TextRecognizer
from classes.visualizer import Visualizer
from classes.image_processor import ImageProcessor

class ImageProcessorApp:
    def __init__(self, model_path, folder_path, output_path):
        self.model = Model(model_path)
        self.folder_path = folder_path
        self.output_path = Path(output_path) / "images"
        self.text_recognizer = TextRecognizer()
        self.visualizer = Visualizer()
        self.category_counts = defaultdict(int)

    def process_images(self):
        self.output_path.mkdir(parents=True, exist_ok=True)

        # loop all file in the folder
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for file in tqdm(files, desc="Processing images"):
            full_path = os.path.join(self.folder_path, file)
            # filter file
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                results = self.model.infer(full_path)
                image = ImageProcessor.read_image(full_path)

                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    cropped_image, bbox = ImageProcessor.crop_image(image, x1, y1, x2, y2)

                    # save image
                    # output_path = os.path.join(self.folder_path, f"cropped_expanded_{file}")
                    # cv2.imwrite(output_path, cropped_image)

                    # self.visualizer.show_image(cropped_image, title="img")

                    result = self.text_recognizer.extract_text(cropped_image)

                    # print(f"\nfile: {file}")
                    # print(f"bounding box: {bbox}")

                    if len(result) == 0:
                        img_folder_path = self.output_path / "others"
                        if not img_folder_path.exists():
                            img_folder_path.mkdir(parents=True, exist_ok=True)
                        shutil.copy(full_path, img_folder_path / file)
                    else:
                        for detection in result:
                            text = detection[1]
                            if text.isdigit():
                                img_folder_path = self.output_path / text
                                # print(f"numero: {text}")
                                if not img_folder_path.exists():
                                    img_folder_path.mkdir(parents=True, exist_ok=True)
                                shutil.copy(full_path, img_folder_path / file)
                                self.category_counts[text] += 1



            # elimina il file spostato
            # os.remove(full_path)

            # break
