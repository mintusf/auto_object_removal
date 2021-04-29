import os
import pytest

from utils.load import parse_config

def test_parse_config():
    default_config_path = os.path.join("config","default.yaml")
    default_config = parse_config(default_config_path)

    assert default_config['input_type'] in ['image', 'video']
    assert isinstance(default_config['semantic_segmentation_cfg'], list)
    assert isinstance(default_config['max_instances'], int) 