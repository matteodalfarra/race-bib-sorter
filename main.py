from classes.race_bib_processor import RaceBibProcessor

if __name__ == "__main__":
    # get folder path
    print("Folder path:")
    folder = input()

    processor = RaceBibProcessor(folder)
    processor.process_images()