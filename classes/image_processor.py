import cv2

class ImageProcessor:
    @staticmethod
    def read_image(image_path):
        return cv2.imread(image_path)

    @staticmethod
    def crop_image(image, x1, y1, x2, y2, margin_x=20, margin_y=10):
        x1_exp = max(x1 - margin_x, 0)
        y1_exp = max(y1 - margin_y, 0)
        x2_exp = min(x2 + margin_x, image.shape[1])
        y2_exp = min(y2 + margin_y, image.shape[0])
        return image[y1_exp:y2_exp, x1_exp:x2_exp], (x1_exp, y1_exp, x2_exp, y2_exp)
