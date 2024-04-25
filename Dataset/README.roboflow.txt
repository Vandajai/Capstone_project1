
WasteSegment_CombinedDataset5 - v1 2024-04-12 10:36pm
==============================

This dataset was exported via roboflow.com on April 12, 2024 at 5:11 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2307 images.
Segmentation-M4wG are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Resize to 640x640 (Fit within)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random rotation of between -15 and +15 degrees
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -10 and +10 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels


