import cv2
import numpy as np

class StandardDisparityCalculator():
    def __init__(self, max_disparity):
        super().__init__()
        self.max_disparity = max_disparity
        self.stereo_processor = cv2.StereoSGBM_create(0, max_disparity, 21)

    def calculate_disparity(self, left_image, right_image):
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        gray_left = np.power(gray_left, 0.75).astype('uint8')
        gray_right = np.power(gray_right, 0.75).astype('uint8')

        disparity = self.stereo_processor.compute(gray_left, gray_right)
        disparity_noise_filter = 5
        cv2.filterSpeckles(disparity, 0, 4000, self.max_disparity - disparity_noise_filter)

        _, disparity = cv2.threshold(disparity, 0, self.max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16).astype(np.uint8)

        return disparity_scaled