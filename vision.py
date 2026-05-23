import cv2
import numpy as np

def process_hoof_image(input_path, output_path):
    img = cv2.imread(input_path)

    if img is None:
        return False, "Could not read image.", None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([5, 20, 50])
    upper_yellow = np.array([60, 255, 255])

    tape_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    cv2.imwrite(output_path, tape_mask)

    white_pixels = cv2.countNonZero(tape_mask)

    # TEMP DEMO MEASUREMENTS
    # Later, image detection will calculate these.
    measured_length = 125
    measured_width = 130

    message = f"Yellow mask test. White pixels: {white_pixels}"

    return True, message, measured_length, measured_width