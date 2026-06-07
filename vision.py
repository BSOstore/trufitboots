import cv2
import numpy as np


def process_hoof_image(input_path, output_path):
    img = cv2.imread(input_path)

    if img is None:
        return False, "Could not read image.", None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([40, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((15, 15), np.uint8)

    yellow_mask = cv2.GaussianBlur(yellow_mask, (9, 9), 0)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return False, "No hoof/boot shape detected.", None, None

    print(f"Contours found: {len(contours)}")

    valid_contours = []
    img_h, img_w = img.shape[:2]

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h else 999

        print(i, "area:", area, "w:", w, "h:", h, "aspect:", aspect)

        if area < 5000:
            continue

        # Ignore tape-measure-like shapes
        if aspect > 2.0:
            continue

        # Ignore very flat/wide junk
        if h < img_h * 0.20:
            continue

        valid_contours.append(c)

    if not valid_contours:
        return False, "No valid hoof/boot shape detected.", None, None

    largest_contour = max(valid_contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    print("Selected contour area:", cv2.contourArea(largest_contour))
    print("Selected box:", w, "x", h)

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

    pixels_per_mm = w / 140

    measured_width = round(w / pixels_per_mm)
    measured_length = round(h / pixels_per_mm)

    message = f"Detected object box: {w}px wide x {h}px tall"

    return True, message, measured_length, measured_width
