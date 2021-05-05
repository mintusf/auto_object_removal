import os
import base64
import json

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