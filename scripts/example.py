import napari
import imageio

from mouselungseg import LungsPredict


def predict(image: "np.ndarray") -> "np.ndarray":
    """The main function."""
    lungs_predict = LungsPredict()
    segmentation = lungs_predict.predict(image)
    mask = lungs_predict.postprocess(segmentation)
    return mask


if __name__ == "__main__":
    image = imageio.imread(
        "https://zenodo.org/record/8099852/files/lungs_ct.tif"
    )
    print(image.shape)

    mask = predict(image)
    print(mask.sum())

    viewer = napari.Viewer()
    viewer.add_image(image)
    viewer.add_labels(mask)

    napari.run()
