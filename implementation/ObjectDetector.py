import cv2
import os
import numpy as np

class ObjectDetector:
    def __init__(self, dataset_path, config_file_path="./yolov3.cfg", weights_file_path="./yolov3.weights", classes_file_path="./coco.names", cycle_left_path="left-images", cycle_right_path="right-images", non_max_threshold=0.4, confidence_threshold=0.5, input_width=1024, input_height=544):
        super().__init__()
        self.dataset_path = dataset_path
        self.cycle_left_path = cycle_left_path
        self.cycle_right_path = cycle_right_path
        self.non_max_threshold = non_max_threshold
        self.confidence_threshold = confidence_threshold
        self.input_width = input_width
        self.input_height = input_height

        # Check config/weight/class files exist
        if not os.path.exists(config_file_path):
            raise FileNotFoundError("config_file_path: File not found")

        if not os.path.exists(weights_file_path):
            raise FileNotFoundError("weights_file_path: File not found")
        
        if not os.path.exists(classes_file_path):
            raise FileNotFoundError("classes_file_path: File not found")
        else:
            # Read classes and set them
            with open(classes_file_path, 'rt') as class_file:
                classes = class_file.read().rstrip("\n").split("\n")

            self.classes = classes

        self.net = self.get_network(config_file_path=config_file_path, weights_file_path=weights_file_path)
        self.output_layer_names = self.get_output_layers()

    def get_classes(self):
        return self.classes

    def get_network(self, **kwargs):
        config_file_path = kwargs.get("config_file_path")
        weights_file_path = kwargs.get("weights_file_path")
        net = cv2.dnn.readNetFromDarknet(config_file_path, weights_file_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return net

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        return [
            layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]

    def post_process(self, frame, net_result):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        class_ids, confidences, boxes = [], [], []
        
        # Filter results to only keep those with high confidence
        for result in net_result:
            for potential_detection in result:
                scores = potential_detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    detection_center_x = int(potential_detection[0] * frame_width)
                    detection_center_y = int(potential_detection[1] * frame_height)
                    detection_width = int(potential_detection[2] * frame_width)
                    detection_height = int(potential_detection[3] * frame_height)
                    detection_left = int(detection_center_x - detection_width / 2)
                    detection_top = int(detection_center_y - detection_height / 2)
                    
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([
                        detection_left,
                        detection_top,
                        detection_width,
                        detection_height
                    ])

        class_ids_nms, confidence_nms, boxes_nms = [], [], []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.non_max_threshold)

        for i in indices:
            i = i[0]
            class_ids_nms.append(class_ids[i])
            confidence_nms.append(confidences[i])
            boxes_nms.append(boxes[i])
            
        return (class_ids_nms, confidence_nms, boxes_nms)

    def detect_objects(self, frame):
        frame_tensor = cv2.dnn.blobFromImage(frame, 1/255, (self.input_width, self.input_height), [0, 0, 0], 1, crop=False)
        self.net.setInput(frame_tensor)
        net_result = self.net.forward(self.output_layer_names)

        class_ids, confidences, boxes = self.post_process(frame, net_result)
        return [class_ids, confidences, boxes]
