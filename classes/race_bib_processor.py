from .bib_reader import BibReader
from .person_detector import PersonDetector
import os
from PIL import Image
import cv2

class RaceBibProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.detector = PersonDetector()
        self.ocr = BibReader()

    def process_images(self):
        # loop each file
        for filename in os.listdir(self.folder_path):
            img_path = os.path.join(self.folder_path, filename) # full image path

            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            # print(filename)
            image_pil = Image.open(img_path)
            image_cv2 = cv2.imread(img_path) 

            people, _ = self.detector.detect_people(image_pil)
            print(f"people found: {len(people)}")

            for i, row in people.iterrows(): # loop each people with his cord 
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                crop = image_cv2[y1:y2, x1:x2] # cut image

                bib_texts = self.ocr.read_bib(crop)
                print(f"  → numbers {i+1}: {bib_texts}")

                # view photo
                # cv2.imshow(f"persona {i+1}", crop)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            break 
