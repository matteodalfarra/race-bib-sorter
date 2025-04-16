from .bib_reader import BibReader
from .person_detector import PersonDetector
import os
from PIL import Image
import cv2
import numpy as np

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
    
    def preprocess_for_ocr_bis(self, image):
        # Converti in HSV per filtrare il rosso
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Maschera per escludere lo sfondo rosso
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Inverti la maschera per mantenere SOLO i numeri bianchi
        non_red_mask = cv2.bitwise_not(red_mask)

        # Applica la maschera all'immagine originale
        filtered = cv2.bitwise_and(image, image, mask=non_red_mask)

        # Scala di grigi
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        # Equalizzazione + soglia
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
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
                processed_crop = self.preprocess_for_ocr_bis(crop)
                
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
    