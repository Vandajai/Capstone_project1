# Python In-built packages
from pathlib import Path
import PIL
import numpy as np

# External packages
import streamlit as st
import pandas as pd

# Local Modules
import settings
import helper
from helper import calculate_area
from helper import calculate_pixel_counts
from helper import generate_summary

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
                try:
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes.xyxy
                    areas = calculate_area(boxes)

                    # Filter out categories with zero pixel counts
                    total_pixels, category_pixel_counts = calculate_pixel_counts(res, category_names)
                    percentages = {cat: (count / total_pixels) * 100 for cat, count in category_pixel_counts.items()}

                    # Filter out categories with zero pixel counts
                    non_zero_pixel_counts = {cat: count for cat, count in category_pixel_counts.items() if count > 0}
                    non_zero_percentages = {cat: percent for cat, percent in percentages.items()  if percent > 0}

                    # Calculate the total area of the image
                    total_pixels = uploaded_image.width * uploaded_image.height

                    # Calculate the area of each detected category
                    areas = calculate_area(boxes)
                    category_areas = {cat: area for cat, area in zip(non_zero_pixel_counts.keys(), areas)}

                    # Ensure Garbage area is 100% and adjust other categories accordingly
                    garbage_area = max(category_areas.get("Garbage", 1), max(category_areas.values()) * 1.2)  # Ensure garbage area is at least 1 and higher than others
                    adjusted_category_areas = {cat: area / garbage_area * 100 if cat != "Garbage" else 100 for cat, area in category_areas.items()}

                    # Create a list of tuples for each detected category with percentage and adjusted area
                    adjusted_detected_results = []
                    for cat, area in adjusted_category_areas.items():
                        adjusted_detected_results.append((cat, f"{area:.0f}%", f"{area / 100 * total_pixels:.4f}"))

                    # Sort the adjusted results by adjusted
                    # Sort the results by area in descending order
                    adjusted_detected_results.sort(key=lambda x: float(x[2]), reverse=True)

                    # Display the detected image once for all categories
                    detected_image = PIL.Image.fromarray(res[0].plot()[:, :, ::-1])
                    detected_image_path = "detected_image.png"
                    detected_image.save(detected_image_path)
                    st.image(detected_image_path, caption="Detected Image", use_column_width=True)

                    # Display the results in a table format below the image
                    with st.expander("Detection Results"):
                        results_df = pd.DataFrame(adjusted_detected_results, columns=["Metric", "In picture (%)", "Area (cm²)"])
                        st.dataframe(results_df)

                    st.balloons()

                except Exception as ex:
                    st.error("Error occurred during waste detection.")
                    st.error(ex)
