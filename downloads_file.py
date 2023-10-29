from telegram import Bot, Update
from telegram.ext import MessageHandler, Filters, Updater
from telegram.ext import Filters
import subprocess
from person_recognition.recognition import Recognition
from person_recognition.read_image import ReadImage
import person_recognition.file_admin as FileAdmin
import requests
import threading
import time


import os

TOKEN = '6957289365:AAES1UgAefuj2GTgFhIyDJvCiLaV1fxLhfc'
DOWNLOAD_FOLDER = "downloads/"
URL = "https://api.telegram.org/bot" + TOKEN + "/"


recognition = Recognition()
IMAGE_FOLDER = "person_recognition/image/"


def adminFile():
    FileAdmin.deletFolder(DOWNLOAD_FOLDER)
    FileAdmin.newFolder(DOWNLOAD_FOLDER)


def handle_document(update: Update, context):
    adminFile()
    # Obteniendo el archivo/documento enviado al bot
    file = update.message.document.get_file()
    # Descargando el archivo
    file.download(custom_path=os.path.join(
        DOWNLOAD_FOLDER, update.message.document.file_name))
    # update.message.reply_text("Archivo descargado con éxito!")
    evaluate_image(update)


def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    # Manejador para archivos/documentos
    dp.add_handler(MessageHandler(Filters.document, handle_document))

    # Iniciando el bot
    updater.start_polling()
    updater.idle()


def evaluate_image(update):
    model, result, img = recognition.recognition_init()
    # recognition.set_rectangle_to_image(result)
    bbox, lbls = ReadImage.filter_results(result)
    FileAdmin.deletFolder(IMAGE_FOLDER)
    ReadImage.draw_rectangles_red(img, bbox, lbls)
    recognize_clothing(update)


def recognize_clothing(update):
    for element in FileAdmin.list_files(IMAGE_FOLDER):
        predicted_class, predictions = recognition.load_image(
            IMAGE_FOLDER+element)
        time.sleep(1)
        threading.Thread(target=send_message, args=(
            update, element+": "+predicted_class, IMAGE_FOLDER+element)).start()


def send_message(update, predicted_class, path):
    with open(path, 'rb') as photo:
        update.message.reply_photo(photo=photo, caption=predicted_class)
        # update.message.reply_text(predicted_class)


main()
