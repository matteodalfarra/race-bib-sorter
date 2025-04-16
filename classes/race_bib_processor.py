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

    def preprocess_for_ocr(self, image):
        # Converti in scala di grigi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Inverti i colori (se lo sfondo è rosso e il testo è bianco, il bianco sarà sul nero)
        inverted = cv2.bitwise_not(gray)

        # Aumenta il contrasto con equalizzazione istogramma
        contrast = cv2.equalizeHist(inverted)


        # Aumenta il contrasto con equalizzazione istogramma
        #gray = cv2.equalizeHist(gray)

        # Rimuove rumore con filtro bilaterale (preserva bordi)
        denoised = cv2.bilateralFilter(contrast, 11, 17, 17)

        # Threshold binario (puoi provare anche adaptiveThreshold)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh = cv2.bitwise_not(thresh)
        return thresh
    
    def process_images(self):
        # loop each file
        output_folder = os.path.join(self.folder_path, "debug_crops")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
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
                # Preprocessing prima dell'OCR
                processed_crop = self.preprocess_for_ocr(crop)
                
                # Salva il crop preprocessato
                debug_filename = f"{os.path.splitext(filename)[0]}_person{i+1}.png"
                debug_path = os.path.join(output_folder, debug_filename)
                cv2.imwrite(debug_path, processed_crop)
                print(f"  → Saved debug image: {debug_path}")
                
                bib_texts = self.ocr.read_bib(processed_crop)
                print(f"  → numbers {i+1}: {bib_texts}")


                # view photo
                #cv2.imshow(f"persona {i+1}", crop)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

            break 
    