from ultralytics import YOLO
import time
import streamlit as st
import cv2
#from pytube import YouTube
import os
import platform
import sys
from pathlib import Path
import numpy as np

import settings

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(ROOT / "weights/best.pt")
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 waste detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected images.   
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


# Function to calculate the area of bounding boxes
def calculate_area(boxes):
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        width = x2 - x1
        height = y2 - y1
        area = width * height
        areas.append(area)
    return areas

# Function to calculate pixel counts from masks
def calculate_pixel_counts(outputs, category_names):
    total_pixels = 0
    category_pixel_counts = {category: 0 for category in category_names}

    for output in outputs:
        masks = output.masks.data.cpu().numpy()  # Assuming 'masks' contains the segmentation masks
        classes = output.boxes.cls.cpu().numpy()  # Assuming 'boxes.cls' contains the predicted classes

        for mask, pred_class in zip(masks, classes):
            mask_np = mask.astype(np.uint8)
            mask_pixels = np.sum(mask_np)
            total_pixels += mask_pixels
            category_name = category_names[int(pred_class)]
            category_pixel_counts[category_name] += mask_pixels

    return total_pixels, category_pixel_counts

# Function to compute percentages and generate summary
def generate_summary(inference_results, category_names):
    summaries = []

    for image_path, outputs in inference_results:
        total_pixels, category_pixel_counts = calculate_pixel_counts(outputs, category_names)

        percentages = {cat: (count / total_pixels) * 100 for cat, count in category_pixel_counts.items()}

        summary = {
            "image_path": image_path,
            "total_pixels": total_pixels,
            "category_pixel_counts": category_pixel_counts,
            "percentages": percentages
        }

        summaries.append(summary)

    return summaries
