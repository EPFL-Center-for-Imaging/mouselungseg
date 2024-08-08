![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# 🫁 Lungs segmentation in mice CT scans

![screenshot](images/screenshot.png)

We provide a [YoloV8](https://docs.ultralytics.com/) model for the segmentation of the lungs region in mice CT scans. The model was trained on 2D slices and can be applied slice by slice to produce 3D segmentations.

[[`Installation`](#installation)] [[`Model weights`](#model)] [[`Usage`](#usage)]

This project is part of a collaboration between the [EPFL Center for Imaging](https://imaging.epfl.ch/) and the [De Palma Lab](https://www.epfl.ch/labs/depalma-lab/).

## Installation

We recommend performing the installation in a clean Python environment. Install our package from PyPi:

```sh
pip install mouselungseg
```

or from the repository:

```sh
pip install git+https://gitlab.com/center-for-imaging/lungs-segmentation.git
```

or clone the repository and install with:

```sh
git clone git+https://gitlab.com/center-for-imaging/lungs-segmentation.git
cd mouselungseg
pip install -e .
```

## Model weights

The model weights (~6 Mb) are automatically downloaded from [this repository on Zenodo](https://zenodo.org/records/13234710) the first time you run inference. The model files are saved in the user home folder in the `.mousetumornet` directory.

## Usage

**In Napari**

To use our model in Napari, start the viewer with

```sh
napari -w mouselungseg
```

Open an image using `File > Open files` or drag-and-drop an image into the viewer window.

**Sample data**: To test the model, you can run it on our provided sample image. In Napari, open the image from `File > Open Sample > Mouse lung CT scan`.

Next, in the menu bar select `Plugins > Lungs segmentation (mouselungseg)` to start our plugin.

**As a library**

You can run a model in just a few lines of code to produce a segmentation mask from an image (represented as a numpy array).

```py
from mouselungseg import LungsPredictor

lungs_predict = LungsPredictor()

segmentation = lungs_predict.predict(your_image)
```

**As a CLI**

Run inference on an image from the command-line. For example:

```sh
uls_predict_image -i /path/to/folder/image_001.tif
```

The command will save the segmentation next to the image:

```
folder/
    ├── image_001.tif
    ├── image_001_mask.tif
```

To run inference in batch on all images in a folder, use:

```sh
uls_predict_folder -i /path/to/folder/
```

This will produce:

```
folder/
    ├── image_001.tif
    ├── image_001_mask.tif
    ├── image_002.tif
    ├── image_002_mask.tif
```

## Issues

If you encounter any problems, please file an issue along with a detailed description.

## License

This model is licensed under the [BSD-3](LICENSE.txt) license.

## Related projects

- [Mouse Tumor Net](https://gitlab.com/epfl-center-for-imaging/mousetumornet) | 3D U-Net model trained to segment tumor nodules in mice CT scans.