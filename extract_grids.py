from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
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


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam)


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


def save_detections(image_path, detections):
    directory = os.path.dirname(image_path)  # Get the directory of the base path
    base_name = os.path.basename(image_path)  # Get the base filename
    base_name = os.path.splitext(base_name)[0]
    file_path = os.path.join(directory, 'masks.json')
    mask_without_segmentation = [{key: value for key, value in mask.items() if key != "segmentation"} for mask in detections]
    with open(file_path, 'w') as f:
        json.dump(mask_without_segmentation, f, indent=4)


def detect_and_crop(image_path: str, detect_small_objects: bool, n_grids: int):
    image = load_image(image_path)
    print("Image loaded!")
    if detect_small_objects:
        sam_model = SamAutomaticMaskGenerator(sam)
        print("Model initialised!")
        start = time.time()
        detections = sam_model.generate(image)
        print(f"Time to make the masks: {time.time() - start}")
        save_detections(image_path=image_path, detections=detections)
        print("Detections in place!")
        largest_crops = get_largest_crops(detections, n_grids)
        cropped_images = crop_images(image, largest_crops)
        save_images(cropped_images, image_path)  # Assumes image_path has an extension


# Routine to just detect the grids it shape agnostic
image_path = 'test_flies/2_grid_flies.png'
detect_small_objects = True  # Set to False if you don't want to detect small objects
n_grids = 2  # Number of largest areas to crop
detect_and_crop(image_path, detect_small_objects, n_grids)