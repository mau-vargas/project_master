import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
from PIL import ImageFont, ImageDraw
from IPython.display import display
import person_recognition.file_admin as FileAdmin
from tensorflow import keras
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

# Mostrar las predicciones
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class Recognition:
    label2name = {1: 'persona', 2: 'bicicleta', 3: 'auto', 4: 'moto',
                  8: 'camioneta', 18: 'perro'}

    def __init__(self):
        # Constructor sin argumentos, pero puede inicializar atributos aquí
        self.data = None

    def recognition_init(self):
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        transform = transforms.ToTensor()
        img = Image.open(FileAdmin.get_name_image())  # leer imagen descargada
        img_tensor = transform(img)

        result = model(img_tensor.unsqueeze(0))[0]
        return model, result, img

    def draw_rectangles(self, img, bbox, lbls):
        draw = ImageDraw.Draw(img)
        for k in range(len(bbox)):
            if lbls[k] in self.label2name.keys():
                draw.rectangle(bbox[k], fill=None, outline='white', width=2)
                draw.text([int(d) for d in bbox[k][:2]],
                          self.label2name[lbls[k]], fill='white')

    def filter_results(self, result, threshold=0.9):
        mask = result['scores'] > threshold
        bbox = result['boxes'][mask].detach().cpu().numpy()
        lbls = result['labels'][mask].detach().cpu().numpy()
        return bbox, lbls

    def set_rectangle_to_image(self, result):
        bbox, lbls = self.filter_results(result)
        img = Image.open(FileAdmin.get_name_image())
        self.draw_rectangles(img, bbox, lbls)
        display(img)

######################################################################

    def set_image_array(self, img_path):
        img = image.load_img(img_path, target_size=(
            28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_preprocessed = np.expand_dims(img_array, axis=0)

        return img, img_array, img_preprocessed

    def load_image_cnn(self, img_path):
        # Cargar y preprocesar la imagen
        _, _, img_preprocessed = self.set_image_array(img_path)

        model = keras.models.load_model('models/modelo_cnn.h5')
        predictions = model.predict(img_preprocessed)
        predicted_class = class_names[np.argmax(predictions)]

        print(f"Predicted class: {predicted_class}")
        print(f"Prediction probabilities: {predictions[0]}")

        return predicted_class, predictions[0]

    def load_image_svm(self, _):
        # Cargar y preprocesar la imagen
        _, img_array, _ = self.set_image_array()
        # Esto convierte la imagen en un vector 1D.
        img_flat = img_array.reshape(1, -1)
        # predictions = svm_model.predict(img_flat)
        svm_model = joblib.load('models/svm_model.pkl')

        predictions = svm_model.predict(img_flat)
        predicted_class = class_names[np.argmax(predictions)]

        print(f"Predicted class: {predicted_class}")
        print(f"Prediction probabilities: {predictions}")

        return predicted_class, predictions


######################################################################
