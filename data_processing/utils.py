import os
import json


def list_annotation_paths_recursively(
        directory: str, ignore_background_only_ones: bool = True
) -> list:
    """
    Accepts a folder directory containing image files.
    Returns a list of image file paths present in given directory.
    If ignore_background_only_ones is True, json's coresponding  to background
    only images are discarded.
    """

    # form image regions
    image_xmin = 0
    image_xmax = 1080
    image_ymin = 0
    image_ymax = 1920
    image_bbox = [image_xmin, image_ymin, image_xmax, image_ymax]

    # walk directories recursively and find json files
    relative_filepath_list = []
    # r=root, d=directories, f=files
    for r, _, f in os.walk(directory):
        for file in f:
            if file.split(".")[-1] in ["json"]:
                # get abs file path
                abs_filepath = os.path.join(r, file)

                # pass if sample id card json (such as 43_tur_id.json)
                if "id" in abs_filepath.split(os.sep)[-1]:
                    continue
                else:
                    try:
                        # load poly
                        with open(abs_filepath, "r") as json_file:
                            quad = json.load(json_file)
                            coords = quad["quad"]
                    except:
                        # fix for 29_irn_drvlic.json
                        continue

                # reformat corners
                label_xmin = min([pos[0] for pos in coords])
                label_xmax = max([pos[0] for pos in coords])
                label_ymin = min([pos[1] for pos in coords])
                label_ymax = max([pos[1] for pos in coords])

                # ignore label if label bbox doesnt intersect with image boox
                label_bbox = [label_xmin, label_ymin, label_xmax, label_ymax]
                if ignore_background_only_ones:
                    intersect_area = calculate_intersect_area(label_bbox, image_bbox)
                    if intersect_area < 1:
                        continue

                abs_filepath = abs_filepath.replace("\\", "/")  # for windows
                relative_filepath = abs_filepath.split(directory)[
                    -1
                ]  # get relative path from abs path
                relative_filepath = [
                    relative_filepath[1:]
                    if relative_filepath[0] == "/"
                    else relative_filepath
                ][0]
                relative_filepath_list.append(relative_filepath)

    number_of_files = len(relative_filepath_list)
    folder_name = directory.split(os.sep)[-1]
    print("There are {} image files in folder {}.".format(number_of_files, folder_name))

    return relative_filepath_list


def create_dir(_dir: str):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def calculate_intersect_area(bbox1: list, bbox2: list) -> list:
    """
    Calculates the returns the intersected area btw given bboxes.
    Bounding boxes in the form of: [xmin, ymin, xmax, ymax]
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    # compute the area of intersection rectangle
    intersect_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    return intersect_area


def get_bbox_inside_image(label_bbox: list, image_bbox: list) -> list:
    """
    Corrects label_bbox so that all points are inside image bbox.
    Returns the corrected bbox.
    """
    xA = max(label_bbox[0], image_bbox[0])
    yA = max(label_bbox[1], image_bbox[1])
    xB = min(label_bbox[2], image_bbox[2])
    yB = min(label_bbox[3], image_bbox[3])
    corrected_label_bbox = [xA, yA, xB, yB]

    return corrected_label_bbox
