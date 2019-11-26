import os
import cv2
from PIL import Image
from util import draw_bounding_box
from ObjectDetector import ObjectDetector
from SemanticSegmenter import Segmenter
from DisparityCalculator import DisparityCalculator

##############################
# COMPUTER VISION COURSEWORK #
##############################
master_path_to_dataset = "./TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

# For each frame we:
#   - Detect objects using YOLO
#   - Utilize semantic segmentation on detected objects to extract areas of interest
#   - Calculate the disparity map using x network
#   - Use the segmentation as a binary mask on the disparity map
#   - Calculate the mean of the area of interest for each detected object and calculate distances

object_detector = ObjectDetector(master_path_to_dataset)
segmenter = Segmenter()
disparity_calculator = DisparityCalculator(max_disparity=192, model_path='finetuned')

left_file_list = sorted(os.listdir(full_path_directory_left))

for left_filename in left_file_list:
    right_filename = left_filename.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, left_filename)
    full_path_filename_right = os.path.join(full_path_directory_right, right_filename)

    if '.png' in left_filename:
        image = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        masked_image = image.copy()

    # Ellipse masks out the car bonnet
    cv2.ellipse(masked_image, (int(image.shape[1] / 2), masked_image.shape[0]), (masked_image.shape[0], 130), 0, 180, 360, (0, 0, 0), -1)

    [class_ids, confidences, boxes] = object_detector.detect_objects(masked_image)
    [draw_bounding_box(
        image, object_detector.get_classes()[class_ids[i]], -1, 
        boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3],
        (255, 178, 50)
    ) for i in range(0, len(class_ids))]

    segmentation_map = segmenter.segment_image(Image.fromarray(masked_image))

    if (os.path.isfile(full_path_filename_right)):
        right_image = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        cv2.ellipse(right_image, (int(image.shape[1] / 2), masked_image.shape[0]), (masked_image.shape[0], 130), 0, 180, 360, (0, 0, 0), -1)

    calculated_disparity = disparity_calculator.calculate_disparity(masked_image, right_image)

    cv2.imshow("segmentation", segmentation_map)
    cv2.imshow("disparity", calculated_disparity)
    cv2.imshow("object detection", image)

    cv2.waitKey()
    break