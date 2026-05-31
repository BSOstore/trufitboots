import cv2
import numpy as np


def process_hoof_image(input_path, output_path):
    img = cv2.imread(input_path)

    if img is None:
        return False, "Could not read image.", None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, "No hoof/boot shape detected.", None, None

    # Ignore tiny noise contours
    image_area = img.shape[0] * img.shape[1]

    valid_contours = [c for c in contours if cv2.contourArea(c) > image_area * 0.01]

    if not valid_contours:
        return False, "No large hoof/boot shape detected.", None, None

    largest_contour = max(valid_contours, key=cv2.contourArea)
    print("Total contours found:", len(contours))
    print("Valid contours:", len(valid_contours))
    print("Largest contour area:", cv2.contourArea(largest_contour))

    x, y, w, h = cv2.boundingRect(largest_contour)

    output_img = img.copy()

    cv2.drawContours(output_img, [largest_contour], -1, (0, 255, 0), 2)

    cv2.imwrite(output_path, output_img)

    # TEMP SCALE FACTOR
    # This is only for prototype testing until we add ruler/reference calibration.
    pixels_per_mm = 4.0

    measured_width = round(w / pixels_per_mm)
    measured_length = round(h / pixels_per_mm)

    message = f"Detected object box: {w}px wide x {h}px tall"

    return True, message, measured_length, measured_width
