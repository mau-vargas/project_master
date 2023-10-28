import json
import requests
from loading import Loader
from person_recognition.recognition import Recognition


TOKEN = '6957289365:AAES1UgAefuj2GTgFhIyDJvCiLaV1fxLhfc'
URL = "https://api.telegram.org/bot" + TOKEN + "/"


def get_updates(offset):
    respuesta = requests.get(URL + "getUpdates" +
                             "?offset=" + str(offset) + "&timeout=" + str(100))
    return json.loads(respuesta.content.decode("utf8"))


def handle_message(message_value):
    tipo, idchat, nombre, id_update = info_mensaje(message_value)

    if tipo == "texto":
        texto = leer_mensaje(message_value)
        texto_respuesta = "Has escrito: \"" + texto + "\""
    elif tipo == "sticker":
        texto_respuesta = "Bonito sticker!"
    elif tipo == "animacion":
        texto_respuesta = "Me gusta este GIF!"
    elif tipo == "foto":
        texto_respuesta = "Bonita foto!"
        evaluate_image()

    else:
        texto_respuesta = "Es otro tipo de mensaje"

    enviar_mensaje(idchat, texto_respuesta)


def info_mensaje(message_value):
    if "text" in message_value["message"]:
        tipo = "texto"
    elif "sticker" in message_value["message"]:
        tipo = "sticker"
    elif "animation" in message_value["message"]:
        tipo = "animacion"
    elif "photo" in message_value["message"]:
        tipo = "foto"
    else:
        tipo = "otro"

    persona = message_value["message"]["from"]["first_name"]
    id_chat = message_value["message"]["chat"]["id"]
    id_update = message_value["update_id"]

    return tipo, id_chat, persona, id_update


def leer_mensaje(message_value):
    texto = message_value["message"]["text"]
    return texto


def enviar_mensaje(idchat, texto):
    requests.get(URL + "sendMessage?text=" + texto + "&chat_id=" + str(idchat))


def evaluate_image():
    recognition = Recognition()
    recognition.test_init()


ultima_id = 0

while True:
    mensajes_diccionario = get_updates(ultima_id)

    if mensajes_diccionario["ok"] != "False":
        try:
            for message in mensajes_diccionario["result"]:
                handle_message(message)
                ultima_id = message["update_id"] + 1
        except Exception as e:
            print(f"Se produjo una excepción: {e}")
