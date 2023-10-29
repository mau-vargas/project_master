import os
directory_path = 'person_recognition/image'


def list_files(directory):
    with os.scandir(directory) as entries:
        return [entry.name for entry in entries if entry.is_file()]


def get_name_image():
    for element in list_files(directory_path):
        print(element)


get_name_image()
