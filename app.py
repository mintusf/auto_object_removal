import argparse
from flask import Flask
from dash_apps.semseg_app import attach_semseg_app

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["semseg_img", "instseg_img"],
        help="Select which app should be run",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    runmode = args.mode
    if runmode == 'semseg_img':
        attach_app = attach_semseg_app

    server = Flask(__name__)
    server = attach_app(server)

    server.run(debug=True, port=8888)
