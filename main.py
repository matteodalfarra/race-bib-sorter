from classes.image_processor_app import ImageProcessorApp

if __name__ == '__main__':
    print('Folder path:')
    path = input()

    print('Folder output:')
    path_output = input()
    model_path = './models/model.pt'

    app = ImageProcessorApp(model_path, path, path_output)
    app.process_images()