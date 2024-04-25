# <span style="color:deepskyblue"> Real-time waste Detection with YOLOv8 & Streamlit </span>

This repository is an extensive open-source project showcasing the seamless integration of **Waste detection **
 using **YOLOv8** (object detection algorithm), along with **Streamlit** (a popular Python web application framework for creating interactive web apps). The project offers a user-friendly and customizable interface designed to detect and track waste in real-time images.

## Requirements

Python 3.6+
YOLOv8
Streamlit

```bash
pip install ultralytics streamlit 
```

## Installation

- Clone the repository: git clone <https://github.com/Vandajai/Capstone_project1.git>
- Change to the repository directory: `cd Capstone_Project1`
- Create `weights` and `images` directories inside the project.
- Download the pre-trained YOLOv8 weights and save them to the `weights` directory in the same project.

## Usage

- Run the app with the following command: `streamlit run app.py`
- The app should open in a new browser window.

### ML Model Config

- Select task (Detection, Segmentation)
- Select model confidence
- Use the slider to adjust the confidence threshold (25-100) for the model.

One the model config is done, select a source.

### Detection on images

- The default image with its objects-detected image is displayed on the main page.
- Select a source. (radio button selection `Image`).
- Upload an image by clicking on the "Browse files" button.
- Click the "Detect Objects" button to run the object detection algorithm on the uploaded image with the selected confidence threshold.
- The resulting image with objects detected will be displayed on the page. 



## Acknowledgements

This app uses [YOLOv8]for object detection algorithm and [Streamlit] library for the user interface.

### Disclaimer

Please note this project is intended for Capstone projects only.

**Hit star ⭐ if you like this repo!!!**
