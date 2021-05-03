![Pytests](https://github.com/mintusf/auto-object-removal/actions/workflows/pytest.yml/badge.svg?branch=main)

# auto-object-removal
An application to automatically remove selected objects from images and videos.
For automatic mask detection, it uses segmentation.
There are two removel modes:
* all instance removal (semantic segmentation is used)
* single instance removal (instance segmentation is used)

Implemented segmentation models:
* Deeplab (torchvision weigths used)

Implemented image inpainting models:
* CR fill ([this repository](https://github.com/zengxianyu/crfill))

The project is WIP.
