import cv2
import numpy as np


def process_hoof_image(input_path, output_path):
    img = cv2.imread(input_path)

    if img is None:
        return False, "Could not read image.", None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    print("Contours found:", len(contours))

    if not contours:
        return False, "No hoof/boot shape detected.", None, None

    largest_contour = max(contours, key=cv2.contourArea)

    print("Largest contour area:", cv2.contourArea(largest_contour))

    x, y, w, h = cv2.boundingRect(largest_contour)

    output_img = img.copy()

    cv2.rectangle(
        output_img,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        4,
    )

    cv2.drawContours(
        output_img,
        [largest_contour],
        -1,
        (0, 0, 255),
        6,
    )

    cv2.imwrite(output_path, output_img)

    pixels_per_mm = 4.0

    measured_width = round(w / pixels_per_mm)
    measured_length = round(h / pixels_per_mm)

    message = f"Detected object box: {w}px wide x {h}px tall"

    return True, message, measured_length, measured_width
