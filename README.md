![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# 🫁 Lungs segmentation in mice CT scans

![screenshot](images/screenshot.png)

We provide a 3D U-Net model for the segmentation of the lungs region in mice CT scans.

[[`Installation`](#installation)] [[`Model weights`](#model)] [[`Usage`](#usage)]

This project is part of a collaboration between the [EPFL Center for Imaging](https://imaging.epfl.ch/) and the [De Palma Lab](https://www.epfl.ch/labs/depalma-lab/).

The implementation, training, and validation of the model were done by **Quentin Chappuis** during the course of his Bachelor Project in the Fall semester of 2023 under the supervision of Mallory Wittwer. The corresponding project that includes training notebooks is available [in this repository](https://gitlab.epfl.ch/center-for-imaging/tumor-lungs).

Related projects:

- [Mouse Tumor Net](https://gitlab.com/epfl-center-for-imaging/mousetumornet)

## Installation

We recommend performing the installation in a clean Python environment.

The code requires `python>=3.9`, as well as `pytorch>=2.0`.

Install our package from the repository:

```sh
pip install git+https://gitlab.com/center-for-imaging/lungs-segmentation.git
```

or clone the repository and install with:

```sh
git clone git+https://gitlab.com/center-for-imaging/lungs-segmentation.git
cd mouselungseg
pip install -e .
```

## Model

**Model weights**
The model weights (~1.1 GB) are automatically downloaded from [this repository on Zenodo](https://zenodo.org/records/10492836) the first time you run inference. The model files are saved in the user home folder in the `.lungsunet` directory.

**Training**
The model was trained using a dataset of `355` images coming from 17 different experiments, 2 different scanners and was validated on `62` images.

## Usage

**In Napari**

To use our model in Napari, start the viewer with

```sh
napari -w mouselungseg
```

Open an image using `File > Open files` or drag-and-drop an image into the viewer window. If you want to open medical image formats such as NIFTI directly, consider installing the [napari-medical-image-formats](https://pypi.org/project/napari-medical-image-formats/) plugin.

**Sample data**: To test the model, you can run it on our provided sample image. In Napari, open the image from `File > Open Sample > Mouse lung CT scan`.

Next, in the menu bar select `Plugins > Lungs segmentation (mouselungseg)` to start our plugin.

**As a library**

You can run a model in just a few lines of code to produce a segmentation mask from an image (represented as a numpy array).

```py
from mouselungseg import LungsPredict

lungs_predict = LungsPredict()
segmentation = lungs_predict.predict(your_image)
mask = lungs_predict.postprocess(segmentation)
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

## Carbon footprint of this project

As per the online tool [*Green algorithms*](http://calculator.green-algorithms.org/), the footprint of training this model was estimated to be around 584 g CO2e.