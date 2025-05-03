import matplotlib.pyplot as plt
import cv2

class Visualizer:
    @staticmethod
    def show_image(image, title="Image"):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis("off")
        plt.show()
