import imageio

from mouselungseg import LungsPredictor, extract_3d_roi

if __name__ == "__main__":
    image = imageio.imread("https://zenodo.org/record/8099852/files/lungs_ct.tif")
    print(image.shape)

    predictor = LungsPredictor()

    mask = predictor.predict(image)
    print(mask.shape, mask.sum())

    roi, mask_roi = extract_3d_roi(image, mask)
    print(roi.shape, mask_roi.shape, mask_roi.sum())
