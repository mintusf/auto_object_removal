import os
import base64
import json
from time import time
from typing import List, Tuple
import numpy as np
import cv2

import dash_html_components as html
from urllib.parse import quote as urlquote

def save_file(name, content, folder):
    """Save image in byte format"""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(folder, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files(folder):
    """List the files from the upload directory."""
    files = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            files.append(filename)
    files = sorted(files, key = lambda x: os.path.getmtime(os.path.join(folder, x)))
    return files

def file_download_link(filename, route):
    """Creates a download link"""
    location = "/{}/{}".format(route, urlquote(filename))
    return html.A(filename, href=location)


def update_dict_json(path, key, value):
    if not os.path.isfile(path):
        with open(path, 'w') as fp:
            json.dump({key: value}, fp)
    else:
        with open(path, 'r') as fp:
            loaded_dict = json.load(fp)
            loaded_dict[key] = value
            with open(path, 'w') as fp:
                json.dump({key: value}, fp)


def get_classes_from_json(image_filename, json_path):
    if not os.path.isfile(json_path):
        return []
    
    with open(json_path, 'r') as fp:
        loaded_dict = json.load(fp)
        try:
            classes = loaded_dict[image_filename]
        except KeyError:
            return []

    return classes




def get_bounds(img: np.array) -> List[List[float]]:
    """ Generates bound for image visualization

    Args:
        img (np.array): Input image

    Returns:
        List[List[float]]: Boundary definition
    """
    x, y, _ = img.shape
    x_vis = 100
    y_vis = 0.85 * x_vis * y / x
    bounds = [[x_vis / 2, y_vis / 2], [-x_vis / 2, -y_vis / 2]]
    return bounds

def get_img_coordinates(marker_position: Tuple, image_bounds: List[List[float]], img: np.array) -> Tuple[int]:
    """ Calculates img pixel coordinates given visualization marker's position

    Args:
        marker_position (Tuple): Position of clicked marker
        image_bounds (List[List[float]]): Image bounds 
        img (np.array): INput image

    Returns:
        Tuple[int]: X,Y pixel coordinates of the marker
    """
    marker_x_map, marker_y_map = marker_position

    x_dim_map = image_bounds[0][0] - image_bounds[1][0]
    y_dim_map = image_bounds[0][1] - image_bounds[1][1]
    x_dim_img = img.shape[0]
    y_dim_img = img.shape[1]

    marker_y_img = int(
        x_dim_img * ((image_bounds[0][0] - marker_x_map) / x_dim_map)
    )
    marker_x_img = int(
        y_dim_img * ((marker_y_map - image_bounds[1][1]) / y_dim_map)
    )

    return marker_x_img, marker_y_img

def update_mask(orig_image_name, new_mask, action, upload_directory, masks_directory):
    img_name, ext = os.path.splitext(orig_image_name)
    orig_img = cv2.imread(os.path.join(upload_directory, orig_image_name))
    orig_img_shape = orig_img.shape
    mask_path = os.path.join(masks_directory, f"{img_name}_mask.png")
    if action != "None" and os.path.isfile(mask_path):
        old_mask = np.expand_dims(cv2.imread(mask_path, 0), 2)
    else:
        old_mask = np.zeros((orig_img_shape[0], orig_img_shape[1], 1))
        action = "None"

    # Combine mask
    if action in ["None", "add"]:
        new_mask = np.logical_or(old_mask, new_mask).astype(np.uint8) * 255
    elif action == "remove":
        new_mask = (
            np.logical_and(
                old_mask, np.logical_not(new_mask.astype(np.bool))
            ).astype(np.uint8)
            * 255
        )

    cv2.imwrite(mask_path, new_mask)

    # Update masked_image
    masked_img_path = os.path.join(
        masks_directory, f"{img_name}_masked_{int(time())}.png"
    )
    masked_img = np.where(new_mask, 0, orig_img)
    cv2.imwrite(masked_img_path, masked_img)

    return new_mask, os.path.split(masked_img_path)[-1]