import os
import numpy as np
from skimage.transform import resize
from skimage.exposure import rescale_intensity

import pooch
import onnxruntime

ONNX_MODEL_PATH = os.path.expanduser(
    os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".lungsunet")
)

def retreive_onnx_model():
    """Downloads the model weights from Zenodo."""
    pooch.retrieve(
        url="https://zenodo.org/records/10529475/files/lungs_segmentation_model.onnx",
        known_hash="md5:6304db46465372f6b2d3bb112e0d5df7",
        path=ONNX_MODEL_PATH,
        progressbar=True,
        fname="lungs_segmentation_model.onnx"
    )

class LungsPredict():
    def __init__(self):
        retreive_onnx_model()

        self.ort_session = onnxruntime.InferenceSession(
            os.path.join(ONNX_MODEL_PATH, "lungs_segmentation_model.onnx"), 
            providers=["CPUExecutionProvider"]
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        image_preprocessed = self.preprocess(image)
        ort_inputs = {self.ort_session.get_inputs()[0].name: image_preprocessed}
        ort_outs = self.ort_session.run(None, ort_inputs)
        out = ort_outs[0]
        out = np.squeeze(out)
        out = resize(out, image.shape, order=0)
        return out

    def preprocess(self, image: np.ndarray):
        image = image.astype(np.float32)
        image = resize(image, (128, 128, 128), order=0)
        image = rescale_intensity(image, out_range=(0, 1))
        image = image[None]
        return image[None]

    def postprocess(self, out: np.ndarray, threshold = 0.5) -> np.ndarray:
        return out > threshold