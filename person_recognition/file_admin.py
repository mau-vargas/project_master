import os
import shutil


directory_path = 'downloads'
DOWNLOAD_FOLDER = "downloads/"


def list_files(directory):
    with os.scandir(directory) as entries:
        return [entry.name for entry in entries if entry.is_file()]


def get_name_image():
    name_file = str(list_files(directory_path)[0])
    return directory_path+"/"+name_file


def deletFolder(path):
    # Borrar archivos y subcarpetas
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    # Borrar la carpeta principal
    os.rmdir(path)


def newFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
