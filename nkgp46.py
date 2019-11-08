import cv2
import argparse
import numpy as np
import os 

master_path_to_dataset = "./TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

left_file_list = sorted(os.listdir(full_path_directory_left))

parser = argparse.ArgumentParser(description='Detect pedestrians and vehicles in image sequences')
parser.add_argument("-fs", "--fullscreen", action="store_true", help="Run in fullscreen mode")
parser.add_argument("-cl", "--class_file", type=str, help="List of classes we can detect", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str, help="Neural network configuration", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str, help="Neural network weights", default='yolov3.weights')

args = parser.parse_args()

# Classes we can detect
classesFile = args.class_file

inpWidth = 1024      # Width of input image
inpHeight = 544     # Height of input image
nmsThreshold = 0.4  # Non-maximum suppression threshold
confThreshold = 0.5 # Confidence threshold

def drawBoundingBox(image, class_name, confidence, left, top, right, bottom, colour):
    # Draw the bounding box
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # Confidence label
    label = "{}:{:.2f}".format(class_name, confidence)

    labelSize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])

    cv2.rectangle(
        image,
        (left, top - round(1.5 * labelSize[1])),
        (left + round(1.5 * labelSize[0]), top + baseline), (255, 255, 255),
        cv2.FILLED
    )

    cv2.putText(
        image,
        label,
        (left, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        1
    )

# Get the names of output layers in the Convolutional Neural Network
def getOutputLayerNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def post_process(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Keep only boxes with high confidence scores
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    classIds_nms = []
    confidence_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidence_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    return (classIds_nms, confidence_nms, boxes_nms)

def detectObjects():
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip("\n").split("\n")

    net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights_file)
    output_layer_names = getOutputLayerNames(net)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    windowName = "nkgp46 coursework submission"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    
    keep_processing = True

    capture = cv2.VideoCapture()

    
    for filename_left in left_file_list:

        full_path_filename_left = os.path.join(full_path_directory_left, filename_left);

        if '.png' in filename_left:
            frame = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)

        tensor = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        net.setInput(tensor)

        results = net.forward(output_layer_names)

        classIDs, confidences, boxes = post_process(frame, results, confThreshold, nmsThreshold)

        for detected_object in range(0, len(boxes)):
            if (classes[classIDs[detected_object]] in ['car', 'truck', 'person', 'bicycle']):
                box = boxes[detected_object]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                print(classes[classIDs[detected_object]])

                drawBoundingBox(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50))
        
        cv2.imshow(windowName, frame)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN & args.fullscreen)

        key = cv2.waitKey(2) & 0xFF

        if key == ord('x'):
            keep_processing = False
        elif key == ord('f'):
            args.fullscreen = not(args.fullscreen)

    
    cv2.destroyAllWindows()

detectObjects()