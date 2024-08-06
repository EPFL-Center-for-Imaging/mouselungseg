import pandas as pd
from skimage.measure import regionprops_table


def extract_3d_roi(image, labels):
    df = pd.DataFrame(
        regionprops_table(
            labels, 
            intensity_image=image, 
            properties=['bbox', 'image']
        )
    )

    x0 = int(df["bbox-0"].values[0])
    x1 = int(df["bbox-3"].values[0])
    y0 = int(df["bbox-1"].values[0])
    y1 = int(df["bbox-4"].values[0])
    z0 = int(df["bbox-2"].values[0])
    z1 = int(df["bbox-5"].values[0])

    roi = image[x0:x1, y0:y1, z0:z1]
    roi_mask = df['image'][0]

    return roi, roi_mask