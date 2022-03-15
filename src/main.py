import os

from src.modules import image_process, image_calibration


def main(project_root):
    if input("¿Limpiar resultados anteriores? [Y, y] ") in ("y", "y"):
        d_list = ["/res/1_distor/", "/res/2_joined/", "/res/3_sliced/", "/res/4_concat/"]

        for directory in d_list:
            for root, dirs, files in os.walk(project_root + directory):
                for f in files:
                    os.remove((os.path.join(root, f)).replace("\\", "/")) if not f.startswith(".") else 0

    _ = image_calibration.main(project_root)
    calibration = None

    image_process.main(project_root, calibration)
