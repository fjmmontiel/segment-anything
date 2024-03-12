from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import sys
import time
import os
import json
sys.path.append("..")

#####
sam_checkpoint = "./model_checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'


def filter_data_by_percentile(data, column_name, percentile_value, complementary: bool = False):
    """
    Filter rows in the dataset based on the specified percentile of a given column.

    Parameters:
    - data (list of dicts): The dataset to process, represented as a list of dictionaries.
    - column_name (str): The name of the column to calculate the percentile for.
    - percentile_value (float): The percentile to calculate (e.g., 0.20 for the 20th percentile).

    Returns:
    - list of dicts: The filtered dataset, represented as a list of dictionaries.
    """
    # Convert list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    # Calculate the specified percentile of the given column
    percentile = df[column_name].quantile(percentile_value)
    # Filter the DataFrame to keep rows where the column's value is <= calculated percentile
    if complementary:
        filtered_df = df[df[column_name] >= percentile]
    else:
        filtered_df = df[df[column_name] <= percentile]        
    # Optional: Convert filtered DataFrame back to a list of dictionaries
    filtered_data = filtered_df.to_dict('records')
    # Return the filtered data
    print(f"Total filtered objects are: {len(filtered_df[column_name])}")
    return filtered_data


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

##############


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam)

# , box_nms_thresh=0.3, min_mask_region_area = 10


def load_image(image_path: str):
    # Implement loading the image from disk
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    return image


def get_largest_crops(detections, n_grids):
    # Implement a function to find the n_grids largest detected objects
    # This function should return the bounding boxes of the n_grids largest objects
    print("Running get_largest_crops")
    # We remove the biggest crop of them all
    return sorted(detections, key=lambda x: x['area'], reverse=True)[1:n_grids + 1]


def crop_images(image, crops):
    # Implement a function to crop the largest grids from the original image
    # This function should save the cropped images
    print("Cropping images!")
    cropped_images = []
    for crop in crops:
        x, y, w, h = crop['bbox']
        cropped = image[y:y + h, x:x + w]
        cropped_images.append(cropped)
    return cropped_images


def save_images(images, base_path):
    # Implement a function to save images without losing quality
    print("Saving images!")
    directory = os.path.dirname(base_path)  # Get the directory of the base path
    base_name = os.path.basename(base_path)  # Get the base filename
    base_name = os.path.splitext(base_name)[0]  # Remove the file extension from the base name    
    for idx, image in enumerate(images):
        filename = f"{base_name}_{idx}.png"  # Construct new filename
        file_path = os.path.join(directory, filename)  # Join directory and new filename
        cv2.imwrite(file_path, image)  # PNG for lossless saving
        print(f"Saved {file_path}")


def save_detections(filename, detections):
    mask_without_segmentation = [{key: value for key, value in mask.items() if key != "segmentation"} for mask in detections]
    with open(filename, 'w') as f:
        json.dump(mask_without_segmentation, f, indent=4)


def slice_image(image, num_slices):
    """
    Slice the image into num_slices slices.

    Parameters:
    - image: The image to slice.
    - num_slices: The number of vertical and horizontal slices.

    Returns:
    - list of numpy arrays: The list containing sliced images.
    """
    h, w, _ = image.shape
    slice_height = h // num_slices
    slice_width = w // num_slices
    slices = []

    for i in range(num_slices):
        for j in range(num_slices):
            slice = image[i * slice_height:(i + 1) * slice_height, j * slice_width:(j + 1) * slice_width]
            slices.append(slice)

    return slices


def detect_slice_and_save(sam_model, slice, base_path, slice_index):
    """
    Detect objects in a slice, save the raw slice and the annotated slice.

    Args:
    - sam_model: The SAM model for detection.
    - slice: The image slice to process.
    - base_path: Base path for saving images.
    - slice_index: Index of the slice.
    """
    # Detect objects in the slice
    start = time.time()
    detections = sam_model.generate(slice)
    print(f"Time to make the detection on this slice: {slice_index}, {time.time() - start} seconds")
    # Save the raw slice
    raw_slice_path = f"{base_path}_raw_slice_{slice_index}.png"
    cv2.imwrite(raw_slice_path, slice)
    print(f"Saved raw slice to {raw_slice_path}")

    # Save the annotated slice
    annotated_slice_path = f"{base_path}_annotated_slice_{slice_index}.png"
    save_annotated_image(detections, slice, annotated_slice_path)
    print(f"Saved annotated slice to {annotated_slice_path}")
    save_detections(f"{base_path}_slice_{slice_index}.json", detections)  # Save detections for each slice


def save_annotated_image(anns, raw_slice, file_path):
    """
    Save the annotated image with segmentation masks over the raw slice to a PNG file.

    Args:
    - anns: Annotations containing segmentation masks.
    - raw_slice: The raw image slice to annotate.
    - file_path: Path to save the annotated image.
    """
    # Create an RGBA image for the overlay
    overlay = np.zeros((raw_slice.shape[0], raw_slice.shape[1], 4), dtype=np.uint8)

    for ann in anns:
        m = ann['segmentation']
        color_mask = np.random.random(3) * 255  # Generate a random color
        overlay[m] = np.concatenate([color_mask, [255]])  # Full opacity for the mask

    # Combine the raw slice with the overlay
    annotated_img = raw_slice.copy()
    for c in range(3):  # RGB channels
        annotated_img[:, :, c] = annotated_img[:, :, c] * (255 - overlay[:, :, 3]) / 255 + overlay[:, :, c] * (overlay[:, :, 3] / 255)

    cv2.imwrite(file_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGBA2BGRA))
    print(f"Saved annotated slice to {file_path}")


def process_cropped_images(cropped_images, base_path, n_slices, sam):
    """
    Process cropped images by slicing, detecting, and saving.

    Args:
    - cropped_images: List of cropped images to process.
    - base_path: Base path for saving images.
    - n_slices: The number of slices on one dimension.
    """
    for index, cropped_image in enumerate(cropped_images):
        slices = slice_image(cropped_image, n_slices)
        for slice_index, slice in enumerate(slices):
            detect_slice_and_save(sam, slice, f"{base_path}_{index}", slice_index)


def detect_and_crop(image_path: str, detect_small_objects: bool, n_grids: int, n_slices: int):
    image = load_image(image_path)
    print("Image loaded!")
    if detect_small_objects:
        sam_model = SamAutomaticMaskGenerator(model=sam)
        print("Model initialised!")
        start = time.time()
        detections = sam_model.generate(image)
        print(f"Time to make the masks: {time.time() - start}")
        print("Detections in place!")
        largest_crops = get_largest_crops(detections, n_grids)
        cropped_images = crop_images(image, largest_crops)
        base_path = os.path.splitext(image_path)[0]
        save_images(cropped_images, base_path)  # Assumes image_path has an extension
        process_cropped_images(cropped_images, base_path, n_slices, sam_model)


image_path = 'test_flies/2_grid_flies.png'
detect_small_objects = True  # Set to True if you want to detect small objects
n_grids = 2  # Number of largest areas to crop
num_slices = 4  # Number of slices per cropped grid
detect_and_crop(image_path, detect_small_objects, n_grids, num_slices)
