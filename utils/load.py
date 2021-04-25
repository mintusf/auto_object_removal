import yaml
import os

def parse_config(path: str) -> dict:
    if os.path.splitext(path)[-1] != ".yaml":
        print(os.path.splitext(path)[-1])
        raise ValueError(f"Provided path is {path}, but only yaml files are supported as config files")
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    return config