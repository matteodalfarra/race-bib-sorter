from ultralytics import YOLO
import os
import cv2
import easyocr
import matplotlib.pyplot as plt

model = YOLO('./models/model.pt')
reader = easyocr.Reader(['it']) 

if __name__ == '__main__':
    print('Folder path:')
    path = input()

    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            results = model(full_path)

            image = cv2.imread(full_path)

            margine_x = 20 
            margine_y = 10 

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                x1_exp = max(x1 - margine_x, 0)
                y1_exp = max(y1 - margine_y, 0)
                x2_exp = min(x2 + margine_x, image.shape[1])
                y2_exp = min(y2 + margine_y, image.shape[0])

                cropped_exp = image[y1_exp:y2_exp, x1_exp:x2_exp]

                output_path = os.path.join(path, f"cropped_expanded_{file}")
                cv2.imwrite(output_path, cropped_exp)

                cropped_rgb = cv2.cvtColor(cropped_exp, cv2.COLOR_BGR2RGB)
                plt.imshow(cropped_rgb)
                plt.title("Cropped Pettorale")
                plt.axis("off")
                plt.show()

                result = reader.readtext(cropped_exp)

                print(f"\nfile: {file}")
                print(f"bounding box: {x1_exp}, {y1_exp}, {x2_exp}, {y2_exp}")
                for detection in result:
                    text = detection[1]
                    print(f"🔢 numero: {text}")

            break  