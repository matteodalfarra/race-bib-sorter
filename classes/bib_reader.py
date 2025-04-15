import easyocr

class BibReader:
    def __init__(self):
        self.reader = easyocr.Reader(['it'])

    def read_bib(self, image_crop):
        texts = self.reader.readtext(image_crop, detail=0, paragraph=False) # get text in the cut image
        return texts