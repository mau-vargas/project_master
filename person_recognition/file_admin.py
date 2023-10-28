import os


def list_files(directory):
    with os.scandir(directory) as entries:
        return [entry.name for entry in entries if entry.is_file()]


# Ejemplo de uso:
directory_path = 'downloads'
name_file = str(list_files(directory_path)[0])
print(name_file)
