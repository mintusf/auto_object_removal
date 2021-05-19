import os
import cv2
from shutil import rmtree
from time import time
import numpy as np
from flask import send_from_directory
from dash import Dash, callback_context
import dash_core_components as dcc
import dash_html_components as html
import dash_leaflet as dl
from dash.dependencies import Input, State, Output
from urllib.parse import quote as urlquote

from utils.dash_utils import (
    save_file,
    uploaded_files,
    update_mask,
    get_img_coordinates,
    get_bounds,
)
from models.segmentation.instseg_maskrcnn import InstSegMaskRcnn
from models.inpainting import CRFillModel

# Remove all files and create new folder
UPLOAD_DIRECTORY = "app_files/upload"
SEM_MASKS_DIRECTORY = "app_files/sem_masks"
RESULTS_DIRECTORY = "app_files/results"
MISC_DIRECTORY = "app_files/misc"
for path in [UPLOAD_DIRECTORY, RESULTS_DIRECTORY, SEM_MASKS_DIRECTORY, MISC_DIRECTORY]:
    if os.path.exists(path):
        rmtree(path)
    os.makedirs(path)

config_path = os.path.join("config", "default.yaml")
segmentor = InstSegMaskRcnn(config_path)
inpainter = CRFillModel(config_path)


def attach_instseg_app(server):

    app = Dash(server=server)

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

    @server.route("/masks/<path:path>")
    def get_mask_url(path):
        """Serve a file from the upload directory."""
        img = send_from_directory(
            SEM_MASKS_DIRECTORY, path, as_attachment=True, cache_timeout=0
        )
        return img

    image_bounds = [[40.712216, -74.22655], [40.773941, -74.12544]]
    image_url = "/uploaded_images/human_multi.jpg"

    app.layout = html.Div(
        [
            dcc.Upload(
                id="upload-image",
                children=html.Div(
                    ["Drag and drop or click to select an image to upload"]
                ),
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
            dl.Map(
                [
                    dl.LayerGroup(
                        id="layer",
                        children=dl.Marker(
                            position=[0, 0],
                            id="marker",
                        ),
                    ),
                    dl.ImageOverlay(
                        id="shown-image", opacity=1, url=image_url, bounds=image_bounds
                    ),
                ],
                bounds=image_bounds,
                id="map",
                style={"width": "500px", "height": "500px"},
            ),
            dcc.Dropdown(
                id="class-dropdown",
                placeholder="Please choose the class to remove after you choose the point",
                value="None",
                multi=True,
                style={"text-align": "center"},
            ),
            html.Div(id="dummy_1", children=0, style=dict(display="none")),
            html.Button(
                id="add_mask",
                children="Add mask",
                style={
                    "display": "inline-block",
                    "textalign": "center",
                },
            ),
            html.Button(
                id="remove_mask",
                children="Remove mask",
                style={
                    "textalign": "center",
                },
            ),
            html.Button(
                id="start_inpainting",
                children="Inpaint",
                style={
                    "textalign": "center",
                },
            ),
            html.Ul(id="results-download", style={"text-align": "center"}),
            html.Img(
                id="inpainted-image", style={"display": "inline-block", "width": 500}
            ),
            dcc.ConfirmDialog(
                id='mask_warning',
                message='',
                ),
            dcc.ConfirmDialog(
                id='mask_warning1',
                message='',
                ),
        ],
        style={"max-width": "500px"},
    )

    @app.callback(
        [Output("layer", "children")],
        [Input("map", "click_lat_lng")],
        prevent_initial_call=True,
    )
    def map_click(click_lat_lng):
        """ Supports interactive clicking on the image """
        return [
            dl.Marker(
                position=click_lat_lng,
                id="marker",
            )
        ]

    @app.callback(
        [
            Output("image-dropdown", "options"),
            Output("image-dropdown", "value"),
        ],
        [Input("upload-image", "filename"), Input("upload-image", "contents")],
        prevent_initial_call=True,
    )
    def upload_image(uploaded_filenames, uploaded_file_contents):
        """
        Uploads an image and run segmentation to detect available masks
        """
        if uploaded_filenames is not None and uploaded_file_contents is not None:
            for name, data in zip(uploaded_filenames, uploaded_file_contents):
                save_file(name, data, UPLOAD_DIRECTORY)
        else:
            return [], "None"

        files = uploaded_files(UPLOAD_DIRECTORY)

        drop_list = [{"label": file, "value": file} for file in files]

        selected_option = files[-1]

        return drop_list, selected_option

    @app.callback(
        [Output("dummy_1", "children")],
        [Input("image-dropdown", "value")],
        prevent_initial_call=True,
    )
    def generate_masks(chosen_image):
        """ Generate masks, if not generated yet, when image is selected """

        img_name, _ = os.path.splitext(chosen_image)
        img = cv2.imread(os.path.join(UPLOAD_DIRECTORY, chosen_image))

        masks_path = os.path.join(SEM_MASKS_DIRECTORY, f"{img_name}_masks.npy")
        labels_path = os.path.join(SEM_MASKS_DIRECTORY, f"{img_name}_labels.npy")
        if (not os.path.isfile(masks_path)) or (not os.path.isfile(labels_path)):
            masks, labels = segmentor.predict(img)
            np.save(masks_path, masks)
            np.save(labels_path, labels)

        return [0]

    @app.callback(
        [
            Output("shown-image", "url"),
            Output("shown-image", "bounds"),
            Output("map", "bounds"),
            Output("mask_warning","displayed"),
            Output("mask_warning","message")
        ],
        [
            Input("image-dropdown", "value"),
            Input("add_mask", "n_clicks"),
            Input("remove_mask", "n_clicks"),
            State("marker", "position"),
            State("class-dropdown", "value"),
            State("shown-image", "bounds"),
        ],
        prevent_initial_call=True,
    )
    def show_image(
        orig_image,
        n_click_add,
        n_click_remove,
        marker_position,
        selected_labels,
        current_bound,
    ):
        img_name, _ = os.path.splitext(orig_image)
        img = cv2.imread(os.path.join(UPLOAD_DIRECTORY, orig_image))
        shown_img_path = os.path.join("/uploaded_images", orig_image)
        bounds = get_bounds(img)

        masks_path = os.path.join(SEM_MASKS_DIRECTORY, f"{img_name}_masks.npy")
        labels_path = os.path.join(SEM_MASKS_DIRECTORY, f"{img_name}_labels.npy")

        if selected_labels and selected_labels != "None":
            x, y = get_img_coordinates(marker_position, current_bound, img)

            ctx = callback_context.triggered
            if ctx[0]["prop_id"] == "add_mask.n_clicks":
                action = "add"
            elif ctx[0]["prop_id"] == "remove_mask.n_clicks":
                action = "remove"
            else:
                return shown_img_path, bounds, bounds, False, ""

            if not os.path.isfile(masks_path) or not os.path.isfile(labels_path):
                return shown_img_path, bounds, bounds, True, "Please wait until masks are generated"
            masks = np.load(masks_path)
            labels = np.load(labels_path)
            new_mask = segmentor.get_mask(masks, labels, selected_labels, x, y)

            if new_mask is None:
                return shown_img_path, bounds, bounds, True, "It seems that this label doesn't exist on the image"

            _, masked_img_name = update_mask(orig_image, new_mask, action, UPLOAD_DIRECTORY, SEM_MASKS_DIRECTORY)

            shown_img_path = os.path.join("/masks", masked_img_name)

        return shown_img_path, bounds, bounds, False, ""

    @app.callback(
        [Output("class-dropdown", "options"), Output("class-dropdown", "value"),            Output("mask_warning1","displayed"),
            Output("mask_warning1","message")],
        [
            Input("marker", "position"),
            State("image-dropdown", "value"),
            State("shown-image", "bounds"),
        ],
        prevent_initial_call=True,
    )
    def search_masks(marker_position, chosen_image, image_bounds):

        img_name, _ = os.path.splitext(chosen_image)
        img = cv2.imread(os.path.join(UPLOAD_DIRECTORY, chosen_image))

        masks_path = os.path.join(SEM_MASKS_DIRECTORY, f"{img_name}_masks.npy")
        labels_path = os.path.join(SEM_MASKS_DIRECTORY, f"{img_name}_labels.npy")

        if not os.path.isfile(masks_path) or not os.path.isfile(labels_path):
            return [], [], True, "Slow down, masks are not generated yet"
        masks = np.load(masks_path)
        labels = np.load(labels_path)

        marker_x_img, marker_y_img = get_img_coordinates(
            marker_position, image_bounds, img
        )

        available_classes = segmentor.detect_labels(
            masks, labels, marker_x_img, marker_y_img
        )

        drop_list = [
            {"label": class_name, "value": class_name}
            for class_name in available_classes
        ]

        return drop_list, available_classes, False, ""

    @app.callback(
        [Output("inpainted-image", "src"), Output("results-download", "children")],
        [
            Input("start_inpainting", "n_clicks"),
            State("image-dropdown", "value"),
        ],
    )
    def run_inpainting(nclicks, image_name):
        input_image = cv2.imread(
            os.path.join(UPLOAD_DIRECTORY, image_name), cv2.IMREAD_COLOR
        )

        image_none_condition = image_name is None or image_name == "None"

        if image_none_condition:
            return "", ""

        filename, ext = os.path.splitext(image_name)

        mask = np.expand_dims(
            cv2.imread(os.path.join(SEM_MASKS_DIRECTORY, f"{filename}_mask.png"), 0), 2
        )

        inpainted_result = inpainter.inpaint(input_image, mask)

        results_name = f"{filename}_removed_{int(time())}.png"
        cv2.imwrite(os.path.join(RESULTS_DIRECTORY, results_name), inpainted_result)

        results_location = f"/results/{urlquote(results_name)}"
        return os.path.join("results", results_name), html.A(
            "Download results", href=results_location
        )

    return app