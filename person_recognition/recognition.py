import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
from PIL import ImageFont, ImageDraw
from IPython.display import display
import person_recognition.file_admin as FileAdmin
# import file_admin as FileAdmin
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import joblib
import matplotlib.pyplot as plt


from skimage import io, img_as_float
from skimage.filters import gaussian
import numpy as np
from tensorflow.keras.preprocessing import image

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

    # def set_image_array(self, img_path):
    #    img = image.load_img(img_path, target_size=(
    #        28, 28), color_mode='grayscale')
    #    img_array = image.img_to_array(img)
    #    img_preprocessed = np.expand_dims(img_array, axis=0)
#
    #    return img, img_array, img_preprocessed

    def set_image_array(self, img_path):
        # Cargar la imagen y cambiar el tamaño

        img = image.load_img(img_path, target_size=(
            28, 28), color_mode='grayscale')

        # img = img.convert('L')

        # Convertir la imagen a un arreglo y asegurarse de que es float
        img_array = img_as_float(image.img_to_array(img))

        # Aplicar el filtro gaussiano para suavizar la imagen y reducir el ruido
        # El valor de 'sigma' controla el grado de suavizado
        img_array = gaussian(img_array, sigma=1)

        # Preparar la imagen para el modelo expandiendo las dimensiones
        img_preprocessed = np.expand_dims(img_array, axis=0)

        return img, img_array, img_preprocessed

    def load_image_cnn(self, img_path):
        # Cargar y preprocesar la imagen
        _, _, img_preprocessed = self.set_image_array(img_path)

        model = keras.models.load_model('models/modelo_cnn.h5')
        predictions = model.predict(img_preprocessed)
        predicted_class = class_names[np.argmax(predictions)]

        return predicted_class, predictions[0]

    def show_image(img_path, img, img_array, img_preprocessed):
        # Configurar un subplot para mostrar las imágenes
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Mostrar imagen original
        axs[0].imshow(img, cmap='gray')
        axs[0].title.set_text('Original Image')
        axs[0].axis('off')

        # Mostrar imagen como arreglo NumPy
        axs[1].imshow(np.squeeze(img_array), cmap='gray')
        axs[1].title.set_text('Image as NumPy Array')
        axs[1].axis('off')

        # Mostrar imagen preprocesada (quitando la dimensión del batch)
        axs[2].imshow(np.squeeze(img_preprocessed), cmap='gray')
        axs[2].title.set_text('Preprocessed Image for Model')
        axs[2].axis('off')

        plt.show()

    def load_image_svm(self, img_path):
        # Cargar y preprocesar la imagen
        _, img_array, _ = self.set_image_array(img_path)
        # Esto convierte la imagen en un vector 1D.
        img_flat = img_array.reshape(1, -1)
        # predictions = svm_model.predict(img_flat)
        svm_model = joblib.load('models/svm_model.pkl')

        predictions = svm_model.predict(img_flat)
        predicted_class = class_names[np.argmax(predictions)]

        print(f"Predicted class: {predicted_class}")
        print(f"Prediction probabilities: {predictions}")

        return predicted_class, predictions

    def load_image_random_forest(self, img_path):
        # Cargar y preprocesar la imagen
        _, img_array, _ = self.set_image_array(img_path)
        # Esto convierte la imagen en un vector 1D.
        img_flat = img_array.reshape(1, -1)
        # predictions = svm_model.predict(img_flat)
        random_forest_model = joblib.load('models/random_forest_model.joblib')

        predictions = random_forest_model.predict(img_flat)
        predicted_class = class_names[np.argmax(predictions)]

        print(f"Predicted class: {predicted_class}")
        print(f"Prediction probabilities: {predictions[0]}")

        return predicted_class, predictions


######################################################################
# recognition = Recognition()
# img_path = "person_recognition/image/persona_superior.png"
# img, img_array, img_preprocessed = recognition.set_image_array(img_path)
# recognition.show_image(img, img_array, img_preprocessed)
