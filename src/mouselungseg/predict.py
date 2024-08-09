import os
import numpy as np
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import pooch
import scipy.ndimage as ndi
from ultralytics import YOLO

from mouselungseg.postprocess import extract_3d_roi


MODEL_PATH = os.path.expanduser(
    os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".mousetumornet")
)

def keep_biggest_object(lab_int: np.ndarray) -> np.ndarray:
    """Selects only the biggest object of a labels image."""
    labels = ndi.label(lab_int)[0]  # label from scipy
    counts = np.unique(labels, return_counts=1)
    biggestLabel = np.argmax(counts[1][1:]) + 1
    return (labels == biggestLabel).astype(int)

def retreive_model():
    """Downloads the model weights from Zenodo."""
    pooch.retrieve(
        url="https://zenodo.org/records/13268683/files/yolo_seg_mouselungs.pt",
        known_hash="md5:139471da545565d033748dc0d54a2392",
        path=MODEL_PATH,
        progressbar=True,
        fname="yolo_seg_mouselungs.pt",
    )


def to_rgb(arr):
    return np.repeat(arr[..., None], repeats=3, axis=-1)


def handle_2d_predict(image, model, imgsz):
    image = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    image = to_rgb(image)

    results = model.predict(
        source=image,
        conf=0.25,  # Confidence threshold for detections.
        iou=0.5,  # Intersection over union threshold.
        imgsz=imgsz,  # Square resizing
        max_det=2,  # Two detections max
        augment=False,
    )

    mask = np.zeros_like(image, dtype=np.uint16)
    r = results[0]
    if r.masks is not None:
        mask = r.masks.cpu().numpy().data[0]  # First mask only
        mask = resize(mask, image.shape, order=0) == 1
        mask[mask] = 1

        # Keep one of the channels only
        if len(mask.shape) == 3:
            mask = mask[..., 0]

        # Fill-in the mask
        mask = ndi.binary_fill_holes(
            mask, structure=ndi.generate_binary_structure(2, 1)
        )

    if len(mask.shape) == 3:
        mask = mask[..., 0]

    return mask


def handle_3d_predict(image, model, imgsz):
    n_slices = len(image)

    mask_3d = []
    for slice_idx, z_slice in enumerate(image):
        print(f"{slice_idx} / {n_slices}")
        mask_2d = handle_2d_predict(z_slice, model, imgsz)
        mask_3d.append(mask_2d)

    mask_3d = np.stack(mask_3d)

    # Dilate in the Z direcion to suppress missing frames
    mask_3d = ndi.binary_dilation(
        mask_3d, structure=ndi.generate_binary_structure(3, 1), iterations=2
    )

    # Keep the biggest object
    mask_3d = keep_biggest_object(mask_3d)  # Note - this also converts the mask from bool => int64

    return mask_3d


def handle_predict(image, model, imgsz):
    if len(image.shape) == 2:
        mask = handle_2d_predict(image, model, imgsz)
    elif len(image.shape) == 3:
        mask = handle_3d_predict(image, model, imgsz)

    mask = mask.astype(np.uint8)

    return mask


class LungsPredictor:
    def __init__(self):
        retreive_model()

        self.model = YOLO(os.path.join(MODEL_PATH, "yolo_seg_mouselungs.pt"))
        self.imgsz = 640

    def predict(self, image: np.ndarray) -> np.ndarray:
        mask = handle_predict(image, self.model, self.imgsz)
        return mask
    
    def compute_3d_roi(self, image: np.ndarray) -> np.ndarray:
        mask = self.predict(image)
        roi, roi_mask = extract_3d_roi(image, mask)
        return roi, roi_mask
