import os
import cv2
from ultralytics import YOLO

def detect_bibs(image_folder):
    # Carica il modello YOLOv8 addestrato
    model = YOLO('./model.pt')  # Assicurati che il modello sia nel percorso corretto

    # Elenco delle immagini nel percorso fornito
    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_name in images:
        # Carica l'immagine
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        # Esegui la previsione
        results = model(image_path)

        # Estrai le previsioni (boxes, confidences e labels)
        boxes = results[0].boxes.xywh  # Coordinate (x_center, y_center, width, height)
        confidences = results[0].boxes.conf  # Confidenza della predizione
        labels = results[0].boxes.cls  # Classi rilevate (es. bibs)

        # Ottieni il nome delle classi (bib)
        class_names = results[0].names

        # Filtra solo i pettorali (bib), la classe 0 è solitamente 'bib' nel tuo modello
        bib_detections = [(box, conf, class_names[int(label)]) for box, conf, label in zip(boxes, confidences, labels) if class_names[int(label)] == 'bib']


        if bib_detections:
            print(f"Detections in {image_name}:")
            for idx, (box, conf, label) in enumerate(bib_detections):
                x_center, y_center, width, height = box
                print(f" - Bib {idx + 1} at [x: {x_center}, y: {y_center}, width: {width}, height: {height}] with confidence {conf:.2f}")
        else:
            print(f"No bibs detected in {image_name}")

if __name__ == "__main__":
    image_folder = input("Enter the path to the folder containing the images: ")
    detect_bibs(image_folder)
