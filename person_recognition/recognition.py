import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
from PIL import ImageFont, ImageDraw
from IPython.display import display


class Recognition:
    label2name = {1: 'persona', 2: 'bicicleta', 3: 'auto', 4: 'moto',
                  8: 'camioneta', 18: 'perro'}

    def __init__(self):
        # Constructor sin argumentos, pero puede inicializar atributos aquí
        self.data = None

    def test_init(self):
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        transform = transforms.ToTensor()
        img = Image.open("person_recognition/persona1.png")
        # img = Image.open("personas-produktmarketing.jpg") # No require normalización de color
        img_tensor = transform(img)

        result = model(img_tensor.unsqueeze(0))[0]
        self.set_rectangle_to_image()

    def draw_rectangles(self, img, bbox, lbls):
        draw = ImageDraw.Draw(img)
        for k in range(len(bbox)):
            if lbls[k] in self.label2name.keys():
                draw.rectangle(bbox[k], fill=None, outline='white', width=2)
                draw.text([int(d) for d in bbox[k][:2]],
                          self.label2name[lbls[k]], fill='white')

    def filter_results(result, threshold=0.9):
        mask = result['scores'] > threshold
        bbox = result['boxes'][mask].detach().cpu().numpy()
        lbls = result['labels'][mask].detach().cpu().numpy()
        return bbox, lbls

    def set_rectangle_to_image(self):
        bbox, lbls = self.filter_results(self.result)
        img = Image.open("person_recognition/persona1.png")
        # img = Image.open("personas-produktmarketing.jpg")
        self.draw_rectangles(img, bbox, lbls)
        display(img)


# Crear un objeto de la clase Persona
# recognition = Recognition()

# Mostrar la información de persona1
# recognition.test_init()


# fnt = ImageFont.truetype("arial.ttf", 20)
