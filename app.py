# Python In-built packages
from pathlib import Path
import PIL
import numpy as np

# External packages
import streamlit as st

# Local Modules
import settings
import helper

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

# 37 Categories
category_names = ['Aluminium_foil', 'Background', 'Cardboard', 'Cig_bud', 'Cig_pack', 'Disposable', 'E-Waste', 'Foam Paper', 'Foam cups and plates', 'Garbage', 'Glass_bottle', 'Light bulbs', 'Mask', 'Metal', 'Nylog_sting', 'Nylon_sting', 'Papar_Cup', 'Paper', 'Plastic', 'Plastic_Bag', 'Plastic_Container', 'Plastic_Glass', 'Plastic_Straw', 'Plastic_bottle', 'Plastic_tray', 'Plastic_wraper', 'Rubber', 'Steel_Bottle', 'Tetrapack', 'Thermocol', 'Toothpaste', 'can', 'contaminated_waste', 'diaper_sanitarypad', 'tin_box', 'top_view_waste', 'wood']

# Setting page layout
st.set_page_config(
    page_title="Waste Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Centered title with red color
st.markdown(
    '<p style="text-align:center; color:red; font-size:30px;">Capstone Project</p>',
    unsafe_allow_html=True
)

# Subtitle in bold
st.markdown(
    '<p style="text-align:center; font-size:40px; font-weight:bold;">Waste Detection using YOLOv8</p>',
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png"))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Waste'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes.xyxy
                areas = calculate_area(boxes)
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)

                # Calculate pixel counts and percentages
                total_pixels, category_pixel_counts = calculate_pixel_counts(res, category_names)
                percentages = {cat: (count / total_pixels) * 100 for cat, count in category_pixel_counts.items()}

                # Filter out categories with zero pixel counts
                non_zero_pixel_counts = {cat: count for cat, count in category_pixel_counts.items() if count > 0}
                non_zero_percentages = {cat: percent for cat, percent in percentages.items() if percent > 0}

                st.write(f"Total Pixels: {total_pixels}")
                st.write("Category Pixel Counts:")
                st.write(non_zero_pixel_counts)
                st.write("Percentages:")
                st.write(non_zero_percentages)

                try:
                    with st.expander("Detection Results"):
                        for box, area in zip(boxes, areas):
                            st.write(f"Box: {box}")
                            st.write(f"Area: {area} pixels")
                    # Add balloons after displaying results
                    st.balloons()
                except Exception as ex:
                    st.write("No image is uploaded yet!")
else:
    st.error("Please select a valid source type!")
