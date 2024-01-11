import os
import numpy as np
import torch
from UNet_lungs_segmentation.model import UNet
from torchvision.transforms import ToTensor
from skimage.transform import resize
from skimage.exposure import rescale_intensity

import pooch


MODEL_PATH = os.path.expanduser(
    os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".lungsunet")
)


def retreive_model():
    """Downloads the model weights from Zenodo."""
    pooch.retrieve(
        url="https://sandbox.zenodo.org/records/21462/files/model_weights.pt",
        known_hash="md5:323dc96017201776bbd00ce289d93e6e",
        path=MODEL_PATH,
        progressbar=True,
    )


class LungsPredict():
    def __init__(self):
        retreive_model()

        self.device = "cpu"
        self.model = UNet(n_channels=1, n_class=1).to(self.device)
        self.checkpoint = torch.load(f"{MODEL_PATH}/4bfcccbf3653f4bd442c55e242d5af65-model_weights.pt", map_location=torch.device('cpu'))
        self.model.load_state_dict(self.checkpoint["model_state_dict"])

    def predict(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self.preprocess(image).to(self.device)

        out = self.model(image_tensor)
        out = out.cpu().detach().numpy()
        out = np.squeeze(out)
        out = np.transpose(out, axes=(1, 2, 0))
        return resize(out, image.shape, order=0)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32)
        image = resize(image, (128, 128, 128), order=0)
        image = rescale_intensity(image, out_range=(0, 1))
        image_tensor = ToTensor()(image)
        image_tensor = image_tensor[None]
        return image_tensor[None]

    def postprocess(self, out: np.ndarray, threshold = 0.5) -> np.ndarray:
        return out > threshold