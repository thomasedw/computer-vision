import cv2
import math

def draw_bounding_box(image, class_name, estimated_distance, left, top, right, bottom, colour):
    if class_name == "person":
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3)
    else:
        cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    if math.isinf(estimated_distance):
        estimated_distance = ""

    if estimated_distance != "":
        label = "{}:{:.2f}m".format(class_name, estimated_distance)
        fontScale = min(1 / (estimated_distance) ** 1.001, 1) + 0.3
    else:
        label = "{}".format(class_name)
        fontScale = 1.3

    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fontScale, 1)
    top = max(top, label_size[1])

    cv2.rectangle(
        image, (left, bottom - round(1.5 * label_size[1])),
        (left + round(1.5 * label_size[0]), bottom + baseline),
        (255, 255, 255), cv2.FILLED
    )

    cv2.putText(
        image, label, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX,
        0.75 * fontScale, (0, 0, 0), 1
    )
    