from PIL import Image, ImageFont, ImageDraw
from IPython.display import display
from torchvision import models
from torchvision import transforms

label2name = {1: 'persona', 2: 'bicicleta', 3: 'auto', 4: 'moto',
              8: 'camioneta', 18: 'perro'}


class ReadImage:

    def filter_results(result, threshold=0.9):  # borrar
        mask = result['scores'] > threshold
        bbox = result['boxes'][mask].detach().cpu().numpy()
        lbls = result['labels'][mask].detach().cpu().numpy()
        return bbox, lbls

    def draw_rectangles(img, bbox, lbls):
        draw = ImageDraw.Draw(img)
        for k in range(len(bbox)):
            if lbls[k] in label2name.keys():
                draw.rectangle(bbox[k], fill=None, outline='white', width=2)
                draw.text([int(d) for d in bbox[k][:2]],
                          label2name[lbls[k]], fill='white')

    def draw_rectangles_red(img, bbox, lbls):
        draw = ImageDraw.Draw(img)
        for k in range(len(bbox)):
            if lbls[k] in label2name.keys() and label2name[lbls[k]] == 'persona':
                x1, y1, x2, y2 = [int(d) for d in bbox[k]]
                # Divide la región en tres cajas
                height = y2 - y1
                y2_top = y1 + height // 2  # Se divide en 2 para identificar el torzo y las piernas
                y1_bottom = y2 - height // 7  # Se divide en 7 para localizar los pies

                draw.rectangle([x1, y1, x2, y2_top], fill=None,
                               outline='red', width=2)
                # Agregar texto a la parte superior
                draw.text([x1, y1], 'Superior', fill='red')

                draw.rectangle([x1, y2_top, x2, y1_bottom],
                               fill=None, outline='blue', width=2)
                # Agregar texto al medio
                draw.text([x1, y2_top], 'Medio', fill='red')

                draw.rectangle([x1, y1_bottom, x2, y2+10],
                               fill=None, outline='green', width=2)
                # Agregar texto a la parte inferior
                draw.text([x1, y1_bottom], 'Inferior', fill='red')

                # Cortar y guardar cada sección
                img_top = img.crop((x1, y1, x2, y2_top))
                img_top.save("person_recognition/image/persona_top.png")

                img_middle = img.crop((x1, y2_top, x2, y1_bottom))
                img_middle.save("person_recognition/image/persona_middle.png")

                img_bottom = img.crop((x1, y1_bottom, x2, y2))
                img_bottom.save("person_recognition/image/persona_bottom.png")


model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.ToTensor()
img = Image.open("person_recognition/persona1.png")
# img = Image.open("personas-produktmarketing.jpg") # No require normalización de color
img_tensor = transform(img)

result = model(img_tensor.unsqueeze(0))[0]


bbox, lbls = ReadImage.filter_results(result)
img = Image.open("person_recognition/persona1.png")
# img = Image.open("personas-produktmarketing.jpg")
ReadImage.draw_rectangles(img, bbox, lbls)
display(img)


# Luego puedes llamar a la función
img = Image.open("person_recognition/persona1.png")
# img = Image.open("personas-produktmarketing.jpg")

ReadImage.draw_rectangles_red(img, bbox, lbls)

# Muestra la imagen con los rectángulos dibujados
display(img)
