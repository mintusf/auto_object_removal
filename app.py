import os
import shutil
import cv2
import numpy as np
from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, State, Output
from urllib.parse import quote as urlquote

from utils.dash_utils import (
    save_file,
    uploaded_files,
    update_dict_json,
    get_classes_from_json,
)
from models.segmentation.segmentor import Segmentor
from models.inpainting import CRFillModel

# Remove all files and create new folder
UPLOAD_DIRECTORY = "app_files/upload"
SEM_MASKS_DIRECTORY = "app_files/sem_masks"
RESULTS_DIRECTORY = "app_files/results"
MISC_DIRECTORY = "app_files/misc"
for path in [UPLOAD_DIRECTORY, RESULTS_DIRECTORY, SEM_MASKS_DIRECTORY, MISC_DIRECTORY]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

config_path = os.path.join("config", "default.yaml")
segmentor = Segmentor(config_path)
inpainter = CRFillModel(config_path)

server = Flask(__name__)
app = dash.Dash(server=server)


@server.route("/results/<path:path>")
def download_results(path):
    """Serve a file from the upload directory."""
    img = send_from_directory(
        RESULTS_DIRECTORY, path, as_attachment=True, cache_timeout=0
    )
    return img


@server.route("/uploaded_images/<path:path>")
def get_upload_url(path):
    """Serve a file from the upload directory."""
    img = send_from_directory(
        UPLOAD_DIRECTORY, path, as_attachment=True, cache_timeout=0
    )
    return img


app.layout = html.Div(
    [
        html.H1("Auto Object removal", style={"text-align": "center"}),
        html.H3(f"Upload image", style={"text-align": "center"}),
        dcc.Upload(
            id="upload-image",
            children=html.Div(["Drag and drop or click to select an image to upload"]),
            style={
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        dcc.Dropdown(
            id="image-dropdown",
            placeholder="Please choose the input image",
            value="None",
            style={"text-align": "center"},
        ),
        dcc.Dropdown(
            id="class-dropdown",
            placeholder="Please choose the class to remove",
            value="None",
            multi=True,
            style={"text-align": "center"},
        ),
        html.Div(
            id="dynamic-button-container",
            children=[
                html.Button(
                    id="start-inpainting",
                    children="Start inpainting",
                    style={
                        "display": "inline-block",
                        "textalign": "center",
                        "margin-left": "196px",
                        "margin-top": "10px",
                    },
                )
            ],
            style={"display": "inline-block", "text-align": "center"},
        ),
        html.H3(f"Original image", style={"text-align": "center"}),
        html.Img(id="original-image", style={"display": "inline-block", "width": 500}),
        html.H3(f"Inpainted image", style={"text-align": "center"}),
        html.Ul(id="results-download", style={"text-align": "center"}),
        html.Img(id="inpainted-image", style={"display": "inline-block", "width": 500}),
    ],
    style={"max-width": "500px"},
)


@app.callback(
    [
        Output("image-dropdown", "options"),
        Output("image-dropdown", "value"),
        Output("original-image", "src"),
    ],
    [Input("upload-image", "filename"), Input("upload-image", "contents")],
)
def upload_image(uploaded_filenames, uploaded_file_contents):
    """
    Uploads an image and run segmentation to detect available masks
    """
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data, UPLOAD_DIRECTORY)
    else:
        return [], "None", ""

    files = uploaded_files(UPLOAD_DIRECTORY)

    drop_list = [{"label": file, "value": file} for file in files]

    selected_option = files[-1]

    return drop_list, selected_option, os.path.join("/uploaded_images", selected_option)


@app.callback([Output("class-dropdown", "options")], [Input("image-dropdown", "value")])
def select_image(selected_image):
    """Generates mask options for a selected image

    Args:
        selected_image ([type]): [description]

    Returns:
        [type]: [description]
    """

    if selected_image is None or selected_image == "None":
        return [[]]

    # Check if masks were already generated and analyzed
    classes = get_classes_from_json(
        selected_image, os.path.join(MISC_DIRECTORY, "available_classes.json")
    )

    if not classes:
        # Generate semantic masks
        input_image = cv2.imread(os.path.join(UPLOAD_DIRECTORY, selected_image))
        sem_mask_results = segmentor.predict_mask(input_image)
        image_name = os.path.splitext(selected_image)[0]
        np.save(
            os.path.join(SEM_MASKS_DIRECTORY, image_name + "_masks"), sem_mask_results
        )

        # Save available classes to remove
        available_classes = segmentor.get_available_masks(sem_mask_results)
        update_dict_json(
            os.path.join(MISC_DIRECTORY, "available_classes.json"),
            selected_image,
            available_classes,
        )

        classes = get_classes_from_json(
            selected_image, os.path.join(MISC_DIRECTORY, "available_classes.json")
        )

    drop_list = [{"label": class_name, "value": class_name} for class_name in classes]

    return [drop_list]


@app.callback(
    [Output("inpainted-image", "src"), Output("results-download", "children")],
    [
        Input("start-inpainting", "n_clicks"),
        State("image-dropdown", "value"),
        State("class-dropdown", "value"),
    ],
)
def run_inpainting(nclicks, image_name, class_names):
    input_image = cv2.imread(
        os.path.join(UPLOAD_DIRECTORY, image_name), cv2.IMREAD_COLOR
    )

    image_none_condition = image_name is None or image_name == "None"
    class_none_condition = class_names is None or class_names == "None"

    if image_none_condition or class_none_condition:
        return "", ""

    filename, ext = os.path.splitext(image_name)

    model_output = np.load(os.path.join(SEM_MASKS_DIRECTORY, filename + "_masks.npy"))

    single_mask = segmentor.get_mask_sem(model_output, class_names)

    cv2.imwrite(
        f"{SEM_MASKS_DIRECTORY}/{image_name}_{('_').join(class_names)}.png", single_mask
    )

    inpainted_result = inpainter.inpaint(input_image, single_mask)
    results_name = f"{filename}_{('_').join(class_names)}_removed{ext}"
    cv2.imwrite(os.path.join(RESULTS_DIRECTORY, results_name), inpainted_result)

    results_location = f"/results/{urlquote(results_name)}"
    return os.path.join("results", results_name), html.A(
        "Download results", href=results_location
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)