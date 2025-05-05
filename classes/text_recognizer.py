import easyocr

class TextRecognizer:
    def __init__(self, lang='it'):
        self.reader = easyocr.Reader([lang])

    def extract_text(self, image):
        return self.reader.readtext(image)
