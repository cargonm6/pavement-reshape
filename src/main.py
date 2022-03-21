import os

from src.modules import image_process, image_calibration


def txt_f(p_txt, p_type):
    """
    Apply text effect format
    :param p_txt: input text
    :param p_type: type of effect
    :return: output text
    """
    switcher = {
        1: "\U0000256D\U00002500\U00002190 ",
        2: "\U0000251C\U00002500\U00002190 ",
        3: "\U00002570\U00002500\U00002192 ",
    }
    return switcher.get(p_type, "") + p_txt


def main(project_root):
    """
    Main function
    :param project_root: root path of the project
    :return:
    """
    print("\U00002581" * 21)
    print("|| OpenCV-Pavement ||")
    print("\U00002594" * 21)

    if input(txt_f("Clear last results? [Y, y] \U000000BB ", 1)) in ("Y", "y"):
        d_list = ["/res/1_distor/", "/res/2_joined/", "/res/3_sliced/", "/res/4_concat/"]

        for directory in d_list:
            for root, dirs, files in os.walk(project_root + directory):
                for f in files:
                    os.remove((os.path.join(root, f)).replace("\\", "/")) if not f.startswith(".") else 0
        print(txt_f("Last results deleted.\n", 3))
    else:
        print(txt_f("Last results preserved.\n", 3))

    if input(txt_f("Apply image calibration? [Y, y] \U000000BB ", 1)) in ("Y", "y"):
        if input(txt_f("Use existing calibration pattern? [Y, y] \U000000BB ", 2)) in ("y", "y"):
            calibration_params = image_calibration.main(project_root)
            if calibration_params is not None:
                print(txt_f("Automatic calibration will be used.", 3))
            else:
                print(txt_f("Calibration pattern not found. Manual calibration will be used.", 3))
            image_process.main(project_root, True, calibration_params)
        else:
            print(txt_f("Manual calibration will be used.", 3))
            image_process.main(project_root, True)

    else:
        print(txt_f("Image calibration discarded.", 3))
        image_process.main(project_root)
