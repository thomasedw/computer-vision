import os
import cv2
from PIL import Image
from util import draw_bounding_box
from ObjectDetector import ObjectDetector
from SemanticSegmenter import Segmenter
from PyramidDisparityCalculator import DisparityCalculator
import numpy as np
import time
import math

##############################
# COMPUTER VISION COURSEWORK #
##############################
master_path_to_dataset = "C://Users/thoma/development/coursework/software_systems_and_applications/Computer Vision/implementation/TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"
segmentation_colors = {
    "background": (0, 0, 0,),
    "aeroplane": (128, 0, 0),
    "bicycle": (0, 128, 0),
    "bird": (128, 128, 0),
    "boat": (0, 0, 128),
    "bus": (0, 128, 128),
    "car": (128, 128, 128),
    "cat": (64, 0, 0),
    "cow": (64, 128, 0),
    "dog": (64, 128, 128),
    "motorbike": (64, 128, 128),
    "person": (192, 0, 128),
    "train": (128, 192, 0)
}

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

black_threshold = 1

focal_length = 399.9745178222656
baseline = 0.2090607502

# For each frame we:
#   - Detect objects using YOLO
#   - Utilize semantic segmentation on detected objects to extract areas of interest
#   - Calculate the disparity map using x network
#   - Use the segmentation as a binary mask on the disparity map
#   - Calculate the mean of the area of interest for each detected object and calculate distances

object_detector = ObjectDetector(master_path_to_dataset)
object_classes = object_detector.get_classes()
segmenter = Segmenter()
disparity_calculator = DisparityCalculator(max_disparity=192, model_path='C://Users/thoma/development/coursework/software_systems_and_applications/Computer Vision/implementation/pretrained_model_KITTI2015.tar')

left_file_list = sorted(os.listdir(full_path_directory_left))

for left_filename in left_file_list:
    # if left_filename == "1506943061.478682_L.png":
    right_filename = left_filename.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, left_filename)
    full_path_filename_right = os.path.join(full_path_directory_right, right_filename)

    if '.png' in left_filename:
        original_image = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        left_masked_image = original_image.copy()

    start_preprocessing_time = time.time()
    # Ellipse masks out the car bonnet
    cv2.ellipse(left_masked_image, (int(left_masked_image.shape[1] / 2), left_masked_image.shape[0]), (left_masked_image.shape[0], 130), 0, 180, 360, (0, 0, 0), -1)

    left_masked_image_lab = cv2.cvtColor(left_masked_image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(left_masked_image_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    lab_planes[0] = clahe.apply(lab_planes[0])
    left_masked_image_lab = cv2.merge(lab_planes)
    left_masked_image_lab = cv2.cvtColor(left_masked_image_lab, cv2.COLOR_LAB2BGR)

    # Detect objects
    start_object_detection_time = time.time()
    [class_ids, confidences, boxes] = object_detector.detect_objects(left_masked_image)
    end_object_detection_time = time.time()

    # Verify that right_image exists
    if (os.path.isfile(full_path_filename_right)):
        right_masked_image = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        cv2.ellipse(right_masked_image, (int(left_masked_image.shape[1] / 2), left_masked_image.shape[0]), (left_masked_image.shape[0], 130), 0, 180, 360, (0, 0, 0), -1)
        right_masked_image_lab = cv2.cvtColor(right_masked_image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(right_masked_image_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        lab_planes[0] = clahe.apply(lab_planes[0])
        right_masked_image_lab = cv2.merge(lab_planes)
        right_masked_image = cv2.cvtColor(right_masked_image_lab, cv2.COLOR_LAB2BGR)

        # cv2.imshow("corrected_right_masked_image", right_masked_image)
    end_preprocessing_time = time.time()
    # Calculate disparity
    start_disparity_calc_time = time.time()
    calculated_disparity = disparity_calculator.calculate_disparity(left_masked_image, right_masked_image)
    # Zero-out and black areas in the right image (e.g. car bonnet)
    # calculated_disparity[np.any(right_masked_image == [0, 0, 0], axis=-1)] = 0
    # Multiply disparity to resonable (visible) values
    calculated_disparity = (calculated_disparity * 192).astype('uint16')
    cv2.ellipse(calculated_disparity, (int(left_masked_image.shape[1] / 2), left_masked_image.shape[0]), (left_masked_image.shape[0], 130), 0, 180, 360, (0, 0, 0), -1)
    end_disparity_calc_time = time.time()
    average_segmentation_time = []
    
    # Loop through each detected object
    for i in range(0, len(class_ids)):
        segmentation_map = None

        if boxes[i][0] < 0:
            boxes[i][0] = 0
        elif boxes[i][0] > left_masked_image.shape[1]:
            boxes[i][0] = left_masked_image.shape[1]
        
        if boxes[i][1] < 0:
            boxes[i][1] = 0
        elif boxes[i][1] > left_masked_image.shape[0]:
            boxes[i][1] = left_masked_image.shape[0]

        if boxes[i][1] + boxes[i][3] > left_masked_image.shape[0]:
            boxes[i][3] = left_masked_image.shape[0] - boxes[i][1]
        
        if boxes[i][0] + boxes[i][2] > left_masked_image.shape[1]:
            boxes[i][2] = left_masked_image.shape[1] - boxes[i][0]

        if object_classes[class_ids[i]] in segmentation_colors:
            object_colour = segmentation_colors[object_classes[class_ids[i]]]       
            # Segment the detected object
            start_segmentation_time = time.time()
            try:
                segmentation_map = segmenter.segment_image(Image.fromarray(left_masked_image[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]]))
                # Resize segmentation map to original bounding box found from YOLO
                segmentation_map = cv2.resize(segmentation_map, (boxes[i][2], boxes[i][3]), interpolation = cv2.INTER_AREA)
                end_segmentation_time = time.time()
                average_segmentation_time.append(end_segmentation_time - start_segmentation_time)
            except ValueError:
                pass

        # Get disparity map of the YOLO bounding box
        disparity_map_of_object = calculated_disparity[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]]

        # If the segmentation map has failed to pickup objects, we fallback to using the mean of the bouding box of YOLO
        if segmentation_map is None or np.sum(segmentation_map) < black_threshold:
            average_disparity = np.mean(calculated_disparity[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]])    
        else:
            # Select pixels of the disparity which are in the segmentation map
            disparity_segmented_pixels = disparity_map_of_object[np.all(segmentation_map == object_colour, axis=-1)]
            # disparity_segmented_pixels = disparity_map_of_object
            # cv2.imshow("disp_seg_pixels", disparity_segmented_pixels)
            # cv2.waitKey(500)
            disparity_segmented_pixels = disparity_segmented_pixels[abs(disparity_segmented_pixels - np.mean(disparity_segmented_pixels)) < 6 * np.std(disparity_segmented_pixels)]
            original_image[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]][np.all(segmentation_map == object_colour, axis=-1)] = object_colour

            average_disparity = np.mean(disparity_segmented_pixels)

        estimated_distance = focal_length * (baseline / average_disparity) * 100
        # print("average disparity = {}, calculated distance = {}".format(average_disparity, estimated_distance))

        # cv2.imshow("segmentation-object-"+str(i), segmentation_map_of_object)
        if math.isnan(estimated_distance):
            print("NAN")
        else:
            draw_bounding_box(
                original_image, object_classes[class_ids[i]], estimated_distance, 
                boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3],
                (255, 178, 50)
            )
    
    print("Timing overview".center(50, "-"))
    print("| Preprocessing time: {}".format(end_preprocessing_time - start_preprocessing_time))
    print("| Object detection time: {}".format(end_object_detection_time - start_object_detection_time))
    print("| Disparity calculation time: {}".format(end_disparity_calc_time - start_disparity_calc_time))
    if len(average_segmentation_time) > 0:
        print("| Average segmentation time: {}".format(sum(average_segmentation_time) / len(average_segmentation_time)))
    
    # Output disparity
    cv2.namedWindow("calculated_disparity", cv2.WINDOW_NORMAL)
    cv2.imshow("calculated_disparity", calculated_disparity)
    # Output detected objects
    cv2.namedWindow("object_detection", cv2.WINDOW_NORMAL)
    cv2.imshow("object_detection", original_image)
    cv2.waitKey(500)

cv2.destroyAllWindows()